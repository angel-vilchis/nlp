import torch
from torch import nn 
import torch.nn.functional as F
from config import SOS_TOKEN, MAX_LEN_SEQ_A, MAX_LEN_SEQ_B, NUM_LAYERS, DEVICE
from src.attention import BahdanauAttention

TEACHER_FORCE_EPS = torch.tensor(1.0)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_count, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.vocab_count = vocab_count
        self.token_embedding = nn.Embedding(vocab_count, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2*hidden_size, hidden_size, batch_first=True, num_layers=NUM_LAYERS)
        self.out = nn.Linear(hidden_size, vocab_count)
        self.dropout = nn.Dropout(dropout_p)
        self.epsilon = TEACHER_FORCE_EPS

    def forward(self, encoder_outputs, encoder_hidden, targets):
        '''
        Forward pass for training
        '''
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(size=(batch_size, 1), dtype=torch.long, device=DEVICE).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        
        for i in range(MAX_LEN_SEQ_B):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if torch.rand(()) < self.epsilon:
                decoder_input = targets[:, i].unsqueeze(1)
                
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
                
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, tokens, hidden, encoder_outputs):
        '''
        Single forward step for training or inference
        '''
        output = self.token_embedding(tokens)
        output = self.dropout(output)

        query = hidden.permute(1, 0, 2) # decoder hidden state
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((output, context.repeat(1, output.shape[1], 1)), dim=2)
        output, hidden = self.gru(input_gru, hidden)

        attn_weights = F.pad(attn_weights.squeeze(1), pad=(0, MAX_LEN_SEQ_A-attn_weights.size(-1)), value=0)
        output = self.out(output)
        return output, hidden, attn_weights
    
    def predict(self, encoder_outputs, encoder_hidden, sampler):
        '''
        Inference for support of different algorithms for sampling
        '''
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(size=(batch_size, 1), dtype=torch.long, device=DEVICE).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        seqs = []
        attn = []
        probs = []

        for i in range(MAX_LEN_SEQ_B):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            
            # Sample next token
            batch_tokens, batch_probs = sampler.sample(logits=decoder_output)
            decoder_input = batch_tokens
            
            # Update outputs
            seqs.append(batch_tokens)
            attn.append(attn_weights)
            probs.append(batch_probs)
        
        attn = torch.hstack(attn).reshape(batch_size, MAX_LEN_SEQ_B, MAX_LEN_SEQ_A)
        seqs = torch.cat(seqs, dim=1)
        
        probs = torch.cat(probs, dim=1)
        return seqs, attn, probs

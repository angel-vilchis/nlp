import torch
from torch import nn 
from src.encoders import EncoderRNN
from src.decoders import DecoderRNN
from config import SOS_TOKEN, DEVICE, MAX_LEN_SEQ_A, MAX_LEN_SEQ_B
from src.samplers import GreedySampler
from src.beam_search import beam_decode

class EncoderDecoderRNN(nn.Module): 
    def __init__(self, A_vocab_count, hidden_size, B_vocab_count, category_count=0, category_hidden_size=0, dropout_p=0.1, cat_drop_prob=0.1):
        super(EncoderDecoderRNN, self).__init__()
        self.encoder = EncoderRNN(A_vocab_count, hidden_size, category_count=category_count, 
                                  category_hidden_size=category_hidden_size, 
                                  dropout_p=dropout_p, cat_drop_prob=cat_drop_prob)
        self.decoder = DecoderRNN(hidden_size, B_vocab_count, dropout_p=dropout_p)
    
    def forward(self, A_tokens, targets, categories=None):
        out, hidden = self.encoder(A_tokens, categories=categories)
        out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        return self.decoder(out, hidden, targets)
    
    def predict(self, A_tokens, categories=None, sampler=GreedySampler(), num_return_sequences=1, beam_width=1): 
        out, hidden = self.encoder(A_tokens, categories=categories)
        out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        if num_return_sequences > 1:
            return beam_decode(self.decoder, out, hidden, sampler=sampler, num_return_sequences=num_return_sequences, beam_width=beam_width)
        return self.decoder.predict(out, hidden, sampler=sampler)
    
    
class Byt5SentencePieceTransformer(nn.Module): 
    def __init__(self, d_model, nheads, num_encoder_layers, num_decoder_layers, 
                 input_vocab_count, output_vocab_count, category_count=0, dropout_p=0.1, cat_drop_prob=0.1):
        super(Byt5SentencePieceTransformer, self).__init__()
        self.cat_drop_prob = cat_drop_prob
        self.__cat_drop_prob_train = cat_drop_prob
        
        self.input_tok_embed = nn.Embedding(input_vocab_count, 1472)
        self.compress = nn.Linear(1472, d_model)
        self.input_cat_embed = nn.Embedding(category_count, d_model)
        self.input_pos_embed = nn.Embedding(MAX_LEN_SEQ_A+1, d_model) # +1 for category "token"

        self.output_tok_embed = nn.Embedding(output_vocab_count, d_model)
        self.output_pos_embed = nn.Embedding(MAX_LEN_SEQ_B+1, d_model) # +1 for SOS token
        
        self.transformer = nn.Transformer(d_model=d_model, nhead=nheads, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, batch_first=True)
        
        self.linear = nn.Linear(d_model, output_vocab_count)
        self.dropout = nn.Dropout(dropout_p)
        
    def train(self, mode=True):
        super().train(mode)
        self.cat_drop_prob = self.__cat_drop_prob_train if mode else 0.0
        
    def get_input_embed(self, in_tokens, in_len, categories):
        in_tok_embeds = self.compress(self.input_tok_embed(in_tokens))
        if categories == None or (torch.rand(()) < self.cat_drop_prob): # Randomly drop categories
            categories = torch.zeros(size=(in_tokens.size(0), 1), dtype=int, device=DEVICE)
        in_cat_embeds = self.input_cat_embed(categories)
        in_tok_pos = self.input_pos_embed(torch.arange(0, in_len).to(DEVICE))
        in_embeds = self.dropout(torch.hstack((in_cat_embeds, in_tok_embeds+in_tok_pos)))
        return in_embeds
    
    def get_output_embed_training(self, out_tokens, out_len):
        out_tok_embeds = self.output_tok_embed(out_tokens)
        out_tok_pos = self.output_pos_embed(torch.arange(0, out_len).to(DEVICE))
        out_embed = self.dropout(out_tok_embeds+out_tok_pos)
        return out_embed
    
    def get_output_embed_decoding(self, out_tokens, i):
        out_tok_embeds = self.output_tok_embed(out_tokens[:, :i])
        out_tok_pos = self.output_pos_embed(torch.arange(0, i).to(DEVICE))
        out_embed = self.dropout(out_tok_embeds+out_tok_pos)
        return out_embed
    
    def forward(self, in_tokens, out_tokens, categories=None):
        '''
            Training forward pass entire instance
        '''
        out_tokens = torch.hstack((torch.full(size=(out_tokens.size(0), 1), fill_value=SOS_TOKEN, device=DEVICE), out_tokens[:, :-1]))
        in_batch_size, in_len = in_tokens.shape
        out_batch_size, out_len = out_tokens.shape

        in_embed = self.get_input_embed(in_tokens, in_len, categories)
        out_embed = self.get_output_embed_training(out_tokens, out_len)
        input_mask = torch.zeros(in_embed.shape[:-1]).to(DEVICE) #  (in_tokens == PAD_TOKEN)
        output_mask = self.transformer.generate_square_subsequent_mask(out_len).to(DEVICE)

        out = self.transformer(
            in_embed, 
            out_embed, 
            src_key_padding_mask=input_mask, 
            tgt_mask=output_mask,
        )
        out = self.linear(out)
        return out, None, None
        
    def predict(self, in_tokens, categories=None, sampler=GreedySampler(), num_return_sequences=1, beam_width=10):
        '''
            Inference, decode token by token
        '''
        in_batch_size, in_len = in_tokens.shape
        
        in_embed = self.get_input_embed(in_tokens, in_len, categories)
        encoder_outputs = self.transformer.encoder(in_embed)

        if num_return_sequences > 1:
            return beam_decode(self, encoder_outputs, None, num_return_sequences=num_return_sequences, beam_width=beam_width, sampler=sampler, trans=True)

        out_tokens = torch.ones(size=(in_tokens.size(0), MAX_LEN_SEQ_B+1), dtype=torch.long, device=DEVICE) * SOS_TOKEN
        probs = []
        for i in range(1, MAX_LEN_SEQ_B+1):
            out_embed = self.get_output_embed_decoding(out_tokens, i)
            output_mask = self.transformer.generate_square_subsequent_mask(i).to(DEVICE)

            out = self.transformer.decoder(out_embed, 
                                           encoder_outputs, 
                                           tgt_mask=output_mask)
            out = self.linear(out)[:, [-1], :]
            
            # Sample next token
            batch_tokens, batch_probs = sampler.sample(logits=out)
            probs.append(batch_probs)
            
            out_tokens[:, i] = batch_tokens.squeeze()
        
        probs = torch.cat(probs, dim=1)
        return out_tokens[:, 1:], None, probs
            
        
    def predict_temp(self, in_tokens, categories=None, sampler=GreedySampler(), startswith=[]):
        '''
            Testing endswith/startswith idea
        '''
        out_tokens = torch.ones(size=(in_tokens.size(0), MAX_LEN_SEQ_B+1), dtype=torch.long, device=DEVICE) * SOS_TOKEN
        in_batch_size, in_len = in_tokens.shape
        
        in_embed = self.get_input_embed(in_tokens, in_len, categories)
        encoder_outputs = self.transformer.encoder(in_embed)

        if startswith != []:
            out_tokens[:, 1:1+startswith.shape[-1]] = startswith
            
        for i in range(1+startswith.shape[-1], MAX_LEN_SEQ_B+1):
            out_embed = self.get_output_embed_decoding(out_tokens, i)
            output_mask = self.transformer.generate_square_subsequent_mask(i).to(DEVICE)

            out = self.transformer.decoder(out_embed, 
                                           encoder_outputs, 
                                           tgt_mask=output_mask)
            out = self.linear(out)[:, -1, :]
            
            # Sample next token
            new_tokens, probs = sampler.sample(logits=out)
            out_tokens[:, i] = new_tokens.squeeze()
            
        return out_tokens[:, 1:], None
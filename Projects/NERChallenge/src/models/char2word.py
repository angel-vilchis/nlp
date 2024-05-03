from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from configs.model.char2word import NUM_LAYERS_CHAR_RNN, CHAR_HIDDEN_SIZE, CHAR_EMBEDDING_SIZE

class Char2WordModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=NUM_LAYERS_CHAR_RNN):
        super(Char2WordModel, self).__init__()
        self.vocab_size = 2 + vocab_size # 0 for padding, 1 for OOV char
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=0)
        self.gru = nn.LSTM(input_size=self.embedding_dim, hidden_size=CHAR_HIDDEN_SIZE // 2, num_layers=NUM_LAYERS_CHAR_RNN, batch_first=True, bidirectional=True)
    
    def forward(self, char_ids):
        batch_size, num_words, num_chars = char_ids.shape
        char_embeds = self.embedding(char_ids)
        char_embeds = char_embeds.view(-1, num_chars, CHAR_EMBEDDING_SIZE)
        seq_lens = (char_ids.view(-1, num_chars) != 0).sum(axis=1)
        seq_lens[seq_lens == 0] = 1 # These words will be ignored anyways
        packed_input = pack_padded_sequence(char_embeds, seq_lens.tolist(), batch_first=True, enforce_sorted=False)
        _, (final_hidden_states, _) = self.gru(packed_input)
        
        embeds_by_word = final_hidden_states.permute(1, 2, 0).reshape(batch_size, num_words, -1)
        return embeds_by_word
import torch
from torch import nn
from configs.model.biaffine import TAG_HIDDEN_SIZE, WIDTH_HIDDEN_SIZE
from configs.model.word2vec import MAX_WORD_LEN

class BiaffineModule(nn.Module):
    def __init__(self, hidden_size, num_tags, num_features, dropout=0.0, seq_len=MAX_WORD_LEN):
        super(BiaffineModule, self).__init__()
        self.num_features = num_features
        self.start_hidden = nn.Linear(hidden_size + num_features, TAG_HIDDEN_SIZE)
        self.end_hidden = nn.Linear(hidden_size + num_features, TAG_HIDDEN_SIZE)
        self.dropout = nn.Dropout(dropout)
        
        self.width_hidden = nn.Parameter(torch.rand((seq_len, seq_len, WIDTH_HIDDEN_SIZE)))
        self.linear = nn.Linear(2*TAG_HIDDEN_SIZE + WIDTH_HIDDEN_SIZE, num_tags)
        self.U = nn.Parameter(torch.rand((TAG_HIDDEN_SIZE, num_tags, TAG_HIDDEN_SIZE)))
        self.seq_len = seq_len
        nn.init.kaiming_uniform_(self.U)
        nn.init.uniform_(self.width_hidden)
        
    def forward(self, final_hidden, feature_vecs=None):
        start_hidden = self._get_start_hidden(final_hidden, feature_vecs)
        end_hidden = self._get_end_hidden(final_hidden, feature_vecs)
        
        pre_linear = torch.einsum('bsci,bei->bsec', torch.einsum('bsh,hci->bsci', start_hidden, self.U), end_hidden)   
        start_hidden_expanded = start_hidden.unsqueeze(2).expand(-1, -1, self.seq_len, -1)
        end_hidden_expanded = end_hidden.unsqueeze(1).expand(-1, self.seq_len, -1, -1)
        ffn_input = torch.cat((start_hidden_expanded, end_hidden_expanded), dim=-1)
        ffn_input = torch.cat((ffn_input, self.width_hidden.repeat((ffn_input.size(0), 1, 1, 1))), dim=-1)
        return pre_linear + self.linear(ffn_input)
    
    def _get_start_hidden(self, final_hidden, feature_vecs):
        start_hidden = self.start_hidden(torch.cat((final_hidden, feature_vecs), axis=-1)) if self.num_features > 0 else self.start_hidden(final_hidden)
        return self.dropout(start_hidden)
    
    def _get_end_hidden(self, final_hidden, feature_vecs):
        end_hidden = self.end_hidden(torch.cat((final_hidden, feature_vecs), axis=-1)) if self.num_features > 0 else self.end_hidden(final_hidden)
        return self.dropout(end_hidden)
        
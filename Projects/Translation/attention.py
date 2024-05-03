import torch
from torch import nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    '''
        Attention for RNN
    '''
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        '''
            query: decoder hidden state
            keys: encoder hidden states
        '''
        query_projected = self.Wa(query) 
        keys_projected = self.Ua(keys)
        scores = self.Va(torch.tanh(query_projected + keys_projected))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
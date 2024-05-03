import torch
from torch import nn 

class LinearModule(nn.Module):
    def __init__(self, input_size, output_size, num_features):
        super(LinearModule, self).__init__()
        self.num_features = num_features
        self.linear = nn.Linear(input_size + num_features, output_size)
        
    def forward(self, final_embeddings, feature_vecs):
        return self.linear(torch.cat((final_embeddings, feature_vecs), axis=-1)) if self.num_features > 0 else self.linear(final_embeddings)
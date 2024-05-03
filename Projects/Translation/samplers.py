import torch
from torch import nn 
import torch.nn.functional as F
from config import DEVICE

class Sampler: # Base sampler class
    def __call__(self, logits):
        return self.sample(logits)


class GreedySampler(Sampler):
    def __str__(self):
        return "Greedy"
    
    def sample(self, logits, num_samples=1):
        probs = F.softmax(logits, dim=-1) # probs = F.log_softmax(logits, dim=-1)
        topn_probs, topn_indices = probs.topk(num_samples)
        tokens = topn_indices.squeeze(-1).detach()
        probs = topn_probs.squeeze(-1).detach()
        return tokens, probs


class TemperatureSampler(Sampler): # or Topk or hybrid
    def __init__(self, temp=1.0, topk=None):
        self.temp = temp
        self.topk = topk

    def __str__(self):
        if self.temp != 1.0 and self.topk != None: 
            return f"Temp&Topk(t={self.temp}, k={self.topk})"
        
        if self.temp != 1.0: 
            return f"Temp({self.temp})"
        
        if self.topk != None:
            return f"Topk({self.topk})" 
        
        return "Random"

    def sample(self, logits, num_samples=1):
        if self.topk:
            logits, topk_indices = logits.topk(self.topk) # only use topk logits to sample

        probs = F.softmax(logits/self.temp, dim=-1).squeeze(1)
        tokens = torch.multinomial(probs, num_samples=num_samples)
        probs = probs[torch.arange(tokens.size(0)), tokens.squeeze(1)].unsqueeze(-1)

        if self.topk:
            if topk_indices.dim() == 2:
                tokens = topk_indices[torch.arange(tokens.size(0)), tokens.squeeze(1)]
            else:
                tokens = topk_indices[torch.arange(tokens.size(0)), :, tokens.squeeze(1)] # update ids to reflect actual vocab ids

        return tokens, probs


class NucleusSampler(Sampler):
    def __init__(self, p, temp=1.0):
        self.p = p
        self.temp = 1.0

    def __str__(self):
        if self.temp != 1.0:
            return f"Nucl&Temp(p={self.p}, t={self.temp})"
        return f"Nucl({self.p})"
    
    def sample(self, logits, num_samples=1):
        probs = F.softmax(logits / self.temp, dim=-1).squeeze(1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        topks = (cumsum_probs < self.p).min(axis=-1).indices.unsqueeze(-1) # topk for each instance depending on p

        nucleus = torch.arange(sorted_probs.size(-1), device=DEVICE) < topks
        nucleus[:, 0] = True # If first token already has higher probability, then greedy for that case
         
        sorted_probs.masked_fill_(~nucleus, float("-inf"))
        sorted_probs = F.softmax(sorted_probs, dim=-1) # new probs

        tokens = torch.multinomial(sorted_probs, num_samples=num_samples)
        tokens = sorted_indices[torch.arange(tokens.size(0)), tokens.squeeze(1), None] # update ids to reflect actual vocab ids
        return tokens, torch.zeros(size=(logits.size(0), 1))

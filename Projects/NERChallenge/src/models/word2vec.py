import torch
from torch import nn
from configs.model.shared import DEVICE
from configs.model.word2vec import MAX_WORD_LEN
from helpers.losses import ce_loss
from task_helpers.encodings import get_id_from_word, get_word_from_id

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.vocab_size = 2 + vocab_size # 0 for padding, 1 for OOV words
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=0)
        self.linear = nn.Linear(in_features=self.embedding_dim, out_features=self.vocab_size)
    
    def forward(self, word_ids):
        loss = 0
        for i in range(MAX_WORD_LEN):
            curr_word_ids = word_ids.clone()
            labels = curr_word_ids[:, i].clone()
            curr_word_ids[:, i] = 0
            if labels.eq(0).all().item():
                break
            h = self.embedding(curr_word_ids)
            h = h.sum(axis=1) / (curr_word_ids != 0).sum(axis=1).unsqueeze(-1)
            logits = self.linear(h)
            curr_loss = ce_loss(logits, labels, (labels != 0), self.vocab_size)
            loss += curr_loss
        return None, loss / i
    
    def predict(self, context_words):
        word_ids = torch.tensor([get_id_from_word(word) for word in context_words], device=DEVICE)
        h = self.embedding(word_ids)
        h = h.mean(axis=0)
        logits = self.linear(h)
        probs = nn.functional.softmax(logits, dim=-1)
        return probs, get_word_from_id(probs.argmax(-1).item())
    
    
# class Word2VecModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim):
#         super(Word2VecModel, self).__init__()
#         self.vocab_size = 2 + vocab_size # 0 for padding, 1 for OOV words
#         self.embedding_dim = embedding_dim
#         self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=0)
#         self.left = nn.Linear(in_features=self.embedding_dim, out_features=self.vocab_size)
#         self.right = nn.Linear(in_features=self.embedding_dim, out_features=self.vocab_size)
    
#     def forward(self, word_ids):
#         left_labels = torch.hstack((torch.zeros((word_ids.size(0),1), dtype=torch.long, device=DEVICE), word_ids[:,:-1]))
#         right_labels = torch.hstack((word_ids[:,1:], torch.zeros((word_ids.size(0),1), dtype=torch.long, device=DEVICE)))
#         embeddings = self.embedding(word_ids)
#         left_logits = self.left(embeddings)
#         right_logits = self.right(embeddings)
#         left_loss = ce_loss(left_logits, left_labels, (left_labels != 0), self.vocab_size)
#         right_loss = ce_loss(right_logits, right_labels, (right_labels != 0), self.vocab_size)
#         loss = (left_loss + right_loss) / 2
#         return None, loss
    
#     def predict(self, words):
#         word_ids = torch.tensor([get_id_from_word(word) for word in words], device=DEVICE)
#         h = self.embedding(word_ids)
#         left_logits = self.left(h)
#         right_logits = self.right(h)
#         left_probs = nn.functional.softmax(left_logits, dim=-1)
#         right_probs = nn.functional.softmax(right_logits, dim=-1)
        return left_probs, right_probs
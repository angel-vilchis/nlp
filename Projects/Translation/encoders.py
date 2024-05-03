import torch
from torch import nn
from config import NUM_LAYERS, DEVICE, PAD_TOKEN

CAT_ERROR = "Categories must have shape (Batch Size, 1). Fill no categories with zeros."
PRETRAINED_HIDDEN_SIZE = 1472

class EncoderRNN(nn.Module):
    def __init__(self, vocab_count, hidden_size, category_count=0, category_hidden_size=0, dropout_p=0.1, cat_drop_prob=0.1):
        super(EncoderRNN, self).__init__()
        valid_cat_args_provided = (category_count == 0 and category_hidden_size == 0) or (category_count > 0 and category_hidden_size > 0)
        assert valid_cat_args_provided, f"# of Categories={category_count} and category hidden size={category_hidden_size} must both be valid if provided."
        self.cat_drop_prob = cat_drop_prob
        self.__cat_drop_prob_train = cat_drop_prob
        self.category_count = category_count
        self.category_hidden_size = category_hidden_size
        
        self.token_embedding = nn.Embedding(vocab_count, PRETRAINED_HIDDEN_SIZE, padding_idx=PAD_TOKEN)
        self.compress = nn.Linear(PRETRAINED_HIDDEN_SIZE, hidden_size-category_hidden_size)
        if self.category_count:
            self.category_embedding = nn.Embedding(category_count, category_hidden_size, padding_idx=0)
        
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=NUM_LAYERS)
        self.dropout = nn.Dropout(dropout_p)
        
    def train(self, mode=True):
        super().train(mode)
        self.cat_drop_prob = self.__cat_drop_prob_train if mode else 0.0

    def forward(self, tokens, categories=None):
        if categories != None:
            assert categories.shape == (tokens.shape[0], 1), CAT_ERROR
            
        seq_lengths = (tokens != PAD_TOKEN).sum(-1).cpu().numpy()
        token_embeddings = self.token_embedding(tokens)
        token_embeddings = self.compress(token_embeddings)
        token_embeddings = self.dropout(token_embeddings) 
        embeddings = token_embeddings
        
        if self.category_count:
            batch_size, num_tokens, _ = token_embeddings.shape
            if categories == None or (torch.rand(()) < self.cat_drop_prob):
                categories = torch.zeros(size=(batch_size, 1), dtype=int, device=DEVICE)
                
            category_embeddings = self.category_embedding(categories) 
            category_embeddings = category_embeddings.repeat(1, num_tokens, 1)
            embeddings = torch.cat((token_embeddings, category_embeddings), dim=-1)
        
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, seq_lengths, batch_first=True, enforce_sorted=False)
        batch_size, _, embed_size = embeddings.shape
        h_0 = torch.rand(size=(1, batch_size, embed_size), device=DEVICE)
        
        output, hidden = self.gru(packed_embeddings, h_0)
        return output, hidden
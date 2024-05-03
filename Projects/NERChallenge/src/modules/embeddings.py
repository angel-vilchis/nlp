import torch
from torch import nn 
from transformers import AutoModel
from configs.model.char2word import CHAR_EMBEDDING_SIZE, CHAR_HIDDEN_SIZE
from configs.model.transformer import MODEL_NAME
from configs.model.word2vec import WORD_EMBEDDING_SIZE, MAX_WORD_LEN
from ..models.char2word import Char2WordModel
from ..models.word2vec import Word2VecModel
from task_helpers.encodings import word_counts, char_counts

class EmbeddingsModule(nn.Module):
    def __init__(self, transformer, word2vec, char2word):
        super(EmbeddingsModule, self).__init__()
        self.has_word2vec = word2vec
        self.has_char2word = char2word
        self.has_transformer = transformer
        self.hidden_size = 0
        if self.has_word2vec:
            self.word2vec = Word2VecModel(len(word_counts), WORD_EMBEDDING_SIZE).embedding
            self.hidden_size += WORD_EMBEDDING_SIZE
        if self.has_char2word:
            self.char2word = Char2WordModel(len(char_counts), CHAR_EMBEDDING_SIZE)
            self.hidden_size += CHAR_HIDDEN_SIZE
        if self.has_transformer:
            self.transformer = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False)
            self.hidden_size += self.transformer.config.hidden_size
            
    def forward(self, input_ids, attention_mask, token_type_ids, token_ids, word_ids, char_ids):
        embeds = []
        if self.has_transformer:
            transformer_hidden = self._get_transformer_hidden(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, token_ids=token_ids)
            embeds.append(transformer_hidden)
        if self.has_word2vec:
            word_hidden = self._get_word_hidden(word_ids)
            embeds.append(word_hidden)
        if self.has_char2word:
            char_hidden = self._get_char_hidden(char_ids)
            embeds.append(char_hidden)
        final_hidden = torch.cat(embeds, dim=-1)
        return final_hidden
    
    def _get_transformer_hidden(self, input_ids=None, attention_mask=None, token_type_ids=None, token_ids=None):
        o = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # TODO: XLMRoberta doesn't take in token type ids?, can import from an import in another file? 
        o = o.last_hidden_state
        
        transformer_hidden_by_word = [[o[i][token_ids[i] == token_id].mean(axis=0) for token_id in range(token_ids[i].max()+1)] for i in range(o.size(0))]
        
        for i in range(len(transformer_hidden_by_word)):
            n = len(transformer_hidden_by_word[i])
            assert n <= MAX_WORD_LEN
            padding_len = MAX_WORD_LEN - n
            transformer_hidden_by_word[i].extend([torch.zeros_like(transformer_hidden_by_word[i][0])] * padding_len)
            transformer_hidden_by_word[i] = torch.vstack(transformer_hidden_by_word[i])
            
        transformer_hidden = torch.stack(transformer_hidden_by_word)
        return transformer_hidden
    
    def _get_word_hidden(self, word_ids):
        return self.word2vec(word_ids)
    
    def _get_char_hidden(self, char_ids):
        return self.char2word(char_ids)
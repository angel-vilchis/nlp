import torch
from torch import nn 
from helpers.losses import ce_loss, soft_jaccard_loss_seg
from ..modules.embeddings import EmbeddingsModule
from ..modules.biaffine import BiaffineModule
from ..modules.crf import CRFModule
from ..modules.linear import LinearModule
from ..modules.rnn import RNNModule

CRF_WITH_NO_SEQUENCE_LABELER_ERROR = "Cannot have CRF for model that doesn't label each word."

class NERModel(nn.Module):
    def __init__(self, end2end_config):
        super(NERModel, self).__init__()
        self.num_tags = end2end_config["num_tags"]
        self.is_sequence_labeler = end2end_config["is_sequence_labeler"]
        self.has_rnn = end2end_config["has_rnn"]
        self.has_crf = end2end_config["has_crf"]
        self.model_type = end2end_config["model_type"]
        assert not (self.has_crf and not self.is_sequence_labeler), CRF_WITH_NO_SEQUENCE_LABELER_ERROR
        
        self.embedding_module = EmbeddingsModule(end2end_config["has_transformer"], end2end_config["has_word2vec"], end2end_config["has_char2word"])
        self.embedding_dropout = nn.Dropout(end2end_config["embedding_dropout"])
        head_input_size = self.embedding_module.hidden_size
        if self.has_rnn:
            self.rnn = RNNModule(self.embedding_module.hidden_size)
            self.rnn_dropout = nn.Dropout(end2end_config["rnn_dropout"])
            head_input_size = self.rnn.hidden_size
        if self.is_sequence_labeler:
            self.logits = LinearModule(head_input_size, self.num_tags, num_features=end2end_config["num_features"])
            if self.has_crf: self.crf = CRFModule(end2end_config["num_tags"])
        else:
            self.biaffine = BiaffineModule(head_input_size, self.num_tags, num_features=end2end_config["num_features"], dropout=end2end_config["boundary_dropout"])
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, token_ids=None, 
                word_ids=None, word_mask=None, char_ids=None, feature_vecs=None, 
                labels=None, biaffine_prob_labels=None, biaffine_labels=None):            
        embeddings = self.embedding_module(input_ids, attention_mask, token_type_ids, token_ids, word_ids, char_ids)
        embeddings = self.embedding_dropout(embeddings)
        if self.has_rnn:
            embeddings = self.rnn(embeddings, word_mask[:,0])
            embeddings = self.rnn_dropout(embeddings)
        if self.is_sequence_labeler:
            logits = self.logits(embeddings, feature_vecs)
            if self.has_crf:
                loss = self.crf.loss(logits, labels, mask=word_mask[:,0].to(bool), reduction="sum")
            else:
                loss = ce_loss(logits, labels, word_mask[:,0], self.num_tags)
            return logits, loss
        
        else:
            logits = self.biaffine(embeddings, feature_vecs)
            loss = ce_loss(logits, biaffine_labels, word_mask, self.num_tags)
            # loss = soft_jaccard_loss_seg(logits, biaffine_labels, word_mask, self.num_tags)
            return logits, loss
            
            
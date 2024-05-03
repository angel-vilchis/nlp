from torch import nn
from transformers import BertForMaskedLM, RobertaForMaskedLM, XLMRobertaForMaskedLM
from configs.model.transformer import MODEL_NAME, MODEL_TYPE
from helpers.losses import ce_loss
    
class eBayModelForMaskedLM(nn.Module):
    def __init__(self):
        super(eBayModelForMaskedLM, self).__init__()
        if MODEL_TYPE == "bert":
            self.transformer = BertForMaskedLM.from_pretrained(MODEL_NAME)
        if MODEL_TYPE == "roberta":
            self.transformer = RobertaForMaskedLM.from_pretrained(MODEL_NAME)
        if MODEL_TYPE == "xlmroberta":
            self.transformer = XLMRobertaForMaskedLM.from_pretrained(MODEL_NAME)

        self.vocab_size = self.transformer.config.vocab_size

    def forward(self, input_ids, attention_mask, token_type_ids, labels, masked):
        if MODEL_TYPE != "xlmroberta":
            logits = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        else:
            logits = self.transformer(input_ids, attention_mask=attention_mask).logits
            
        attention_mask *= masked
        loss = ce_loss(logits, labels, attention_mask, self.vocab_size)
        return logits, loss

import torch
from configs.model.char2word import MAX_CHAR_LEN
from configs.model.dataset import BOUNDARY_EPS, BOUNDARY_SMOOTHING, LABEL_SMOOTHING
from configs.model.transformer import MAX_BERT_LEN, TOKENIZER, MASK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from configs.model.word2vec import MAX_WORD_LEN
from configs.config import MODEL_CONFIG
from task_helpers.encodings import get_id_from_word, get_id_from_char, normalize
from task_helpers.features import token_features
from task_helpers.tags import ALONE_MARKING, NO_TAG, SPAN_TAGS, tag2id, get_raw_tag_id, id_marking_is_start, id_marking_is_end

TOKEN_TAG_LEN_ERROR = "Tokens and Tags lengths don't match."

class eBayNERD:
    def __init__(self, titles, tags):
        self.titles = titles
        self.tags = tags
    
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, index):
        tokens = self.titles[index].split()
        if self.tags:
            tags = self.tags[index]
            assert len(tokens) == len(tags), TOKEN_TAG_LEN_ERROR
        else:
            tags = [f"{ALONE_MARKING}_" + NO_TAG for _ in tokens]
        
        input_ids = []
        labels = []
        token_ids = []
        word_ids = []
        word_mask = torch.zeros((MAX_WORD_LEN, MAX_WORD_LEN), dtype=torch.long)
        char_ids = []
        feature_vecs = []
        biaffine_prob_labels = torch.tensor([[[1.0] + [0 for _ in range(len(SPAN_TAGS)-1)] for _ in range(MAX_WORD_LEN)] for _ in range(MAX_WORD_LEN)], dtype=torch.float32)
        biaffine_labels = torch.tensor([[0 for _ in range(MAX_WORD_LEN)] for _ in range(MAX_WORD_LEN)], dtype=torch.long)
        seg_labels = torch.tensor([[-1 for _ in range(MAX_WORD_LEN)] for _ in range(MAX_WORD_LEN)], dtype=torch.long)

        # Encode each token and extend target to each segment
        curr_start = None
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            # TRANSFORMER
            enc_token = TOKENIZER.encode(token, add_special_tokens=False)
            enc_token_len = len(enc_token)
            input_ids.extend(enc_token)
            token_ids.extend([i] * enc_token_len)
            labels += [tag2id(tag)]
            
            # WORD
            word_ids.extend([get_id_from_word(normalize(token))])
            
            # CHAR
            curr_char_ids = [get_id_from_char(char) for char in token]
            curr_char_ids = curr_char_ids[:MAX_CHAR_LEN]
            padding_len = MAX_CHAR_LEN - len(curr_char_ids)
            curr_char_ids.extend([0] * padding_len)
            char_ids.extend([curr_char_ids])
            
            # ENGINEERED FEATURES
            feature_vec, feature_id = token_features(token)
            feature_vecs.extend([feature_vec])
            
            # LABELS
            id = tag2id(tag)
            raw_id = get_raw_tag_id(id)
            if id_marking_is_start(id):
                curr_start = i
            if id_marking_is_end(id):
                budget = 1.0
                if BOUNDARY_SMOOTHING:
                    each_amount = BOUNDARY_EPS
                    if LABEL_SMOOTHING:
                        other_ids = get_label_smoothing_ids(raw_id)
                    if (curr_start - 1) >= 0:
                        biaffine_prob_labels[curr_start-1, i, 0] -= each_amount
                        biaffine_prob_labels[curr_start-1, i, raw_id] += each_amount
                        # budget -= each_amount
                    if (curr_start + 1) < len(tokens) and (curr_start + 1) <= i:
                        biaffine_prob_labels[curr_start+1, i, 0] -= each_amount
                        biaffine_prob_labels[curr_start+1, i, raw_id] += each_amount
                        # budget -= each_amount
                    if (i-1) >= 0 and curr_start <= i-1:
                        biaffine_prob_labels[curr_start, i-1, 0] -= each_amount
                        biaffine_prob_labels[curr_start, i-1, raw_id] += each_amount
                        # budget -= each_amount
                    if (i+1) < len(tokens):
                        biaffine_prob_labels[curr_start, i+1, 0] -= each_amount
                        biaffine_prob_labels[curr_start, i+1, raw_id] += each_amount
                        # budget -= each_amount
                biaffine_prob_labels[curr_start, i, 0] -= budget
                biaffine_prob_labels[curr_start, i, raw_id] += budget
                biaffine_labels[curr_start][i] = raw_id
                seg_labels[curr_start][i] = raw_id
                curr_start = None
        seg_labels[seg_labels != -1] = 1
        seg_labels[seg_labels == -1] = 0
        word_mask[:i+1, :i+1] = torch.triu(torch.ones_like(word_mask[:i+1, :i+1]), diagonal=0) # ignore where start > end, and where longer than actual tokens
        
        # TRUNCATE
        input_ids = input_ids[:MAX_BERT_LEN-2]
        token_ids = token_ids[:MAX_BERT_LEN-2]
        labels = labels[:MAX_WORD_LEN]
        word_ids = word_ids[:MAX_WORD_LEN]
        char_ids = char_ids[:MAX_WORD_LEN]
        
        # SOS/EOS
        input_ids = [SOS_TOKEN] + input_ids + [EOS_TOKEN]
        token_ids = [-1] + token_ids + [-1]
        
        # TRANSFORMER PARAMS
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        
        # PADDING
        transformer_padding_len = MAX_BERT_LEN - len(input_ids)
        input_ids += ([PAD_TOKEN] * transformer_padding_len)
        token_ids += ([-1] * transformer_padding_len)
        attention_mask += ([0] * transformer_padding_len)
        token_type_ids += ([0] * transformer_padding_len)

        word_padding_len = MAX_WORD_LEN - len(word_ids)
        word_ids += ([0] * word_padding_len)
        char_ids += ([[0] * MAX_CHAR_LEN] * word_padding_len)
        labels += ([tag2id(f"{ALONE_MARKING}_" + NO_TAG)] * word_padding_len)
        feature_vec, feature_id = token_features(None); feature_vecs += ([feature_vec] * word_padding_len)
        
        out = {}
        if MODEL_CONFIG["has_transformer"]:
            out["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
            out["token_ids"] = torch.tensor(token_ids, dtype=torch.int8)
            out["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
            out["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
        if MODEL_CONFIG["has_word2vec"]:
            out["word_ids"] = torch.tensor(word_ids, dtype=torch.long)
            out["word_mask"] = word_mask
        if MODEL_CONFIG["has_char2word"]:
            out["char_ids"] = torch.tensor(char_ids, dtype=torch.long)
            out["word_mask"] = word_mask
        if MODEL_CONFIG["num_features"]:
            out["feature_vecs"] = torch.tensor(feature_vecs, dtype=torch.float16)
            
        if MODEL_CONFIG["is_sequence_labeler"]:
            out["labels"] = torch.tensor(labels, dtype=torch.long)
            if MODEL_CONFIG["model_type"] == "tag":
                out["labels"] = torch.tensor([get_raw_tag_id(id) for id in labels], dtype=torch.long)
        else:
            if MODEL_CONFIG["model_type"] == "segment":
                out["biaffine_labels"] = seg_labels
            else:
                out["biaffine_labels"] = biaffine_labels
                # out["biaffine_prob_labels"] = biaffine_prob_labels
        return out
        
        
class eBayW2V:
    def __init__(self, titles):
        self.titles = titles
    
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, index):
        title = self.titles[index]
        tokens = title.split()
        
        # Encoding
        word_ids = [get_id_from_word(normalize(token)) for token in tokens]
        
        # Truncation 
        word_ids = word_ids[:MAX_WORD_LEN]

        # Padding
        padding_len = MAX_WORD_LEN - len(word_ids)
        word_ids += ([0] * padding_len)

        return {
                "word_ids": torch.tensor(word_ids, dtype=torch.long),
               }
        
        
class eBayMLM:
    def __init__(self, titles, mask_proportion=0.15):
        self.titles = titles
        self.mask_proportion = mask_proportion
    
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, index):
        tokens = self.titles[index].split()

        input_ids = []
        labels = []
        masked = []

        for i, token in enumerate(tokens):
            enc_token = TOKENIZER.encode(token, add_special_tokens=False)
            enc_token_len = len(enc_token)
            labels.extend(enc_token)
            
            # TODO: Bert 
            if torch.rand(()).item() < self.mask_proportion:
                masked.extend([1] * enc_token_len)
                choice = torch.rand(()).item()
                if choice < 0.8:
                    input_ids.extend([MASK_TOKEN] * enc_token_len)
                elif choice < 0.9:
                    input_ids.extend(torch.randint(TOKENIZER.vocab_size, (enc_token_len,)).tolist())
                else:
                    input_ids.extend(enc_token)
            else:
                masked.extend([0] * enc_token_len)
                input_ids.extend(enc_token)
            
            
        # Add Start and End of Sequences Tokens
        input_ids = input_ids[:MAX_BERT_LEN-2]
        input_ids = [SOS_TOKEN] + input_ids + [EOS_TOKEN]
        
        masked = masked[:MAX_BERT_LEN-2]
        masked = [0] + masked + [0]
        
        labels = labels[:MAX_BERT_LEN-2]
        labels = [SOS_TOKEN] + labels + [EOS_TOKEN]

        # Params for BERT
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # If padding necessary
        padding_len = MAX_BERT_LEN - len(input_ids)
        input_ids += ([PAD_TOKEN] * padding_len)
        labels += ([PAD_TOKEN] * padding_len)
        attention_mask += ([0] * padding_len)
        token_type_ids += ([0] * padding_len)
        masked += ([0] * padding_len)
        
        return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "masked": torch.tensor(masked, dtype=torch.long),
               }
        
def get_label_smoothing_ids(tag_id):
    pass
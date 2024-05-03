import torch
from torch import optim
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from helpers.ckpt import replace_keys
from task_helpers.features import ADD_FEATURES
from task_helpers.tags import SPAN_TAGS, TOKEN_TAGS

# CUSTOM
####################################################################################
transformer_init_ckpt_path = "ckpts/pretrain/bert_dbmdz_finetuned_best_val.pth"
if transformer_init_ckpt_path:
    transformer_init_ckpt = torch.load(transformer_init_ckpt_path)
    transformer_init_ckpt = replace_keys(transformer_init_ckpt["model_state_dict"], "transformer.bert.", "")
    # transformer_init_ckpt = replace_keys(transformer_init_ckpt["model_state_dict"], "MLM.bert.", "")
else:
    transformer_init_ckpt = transformer_init_ckpt_path
    
word2vec_init_ckpt_path = "ckpts/pretrain/mlm_full_title_best_val.pth"
if word2vec_init_ckpt_path:
    word2vec_init_ckpt = torch.load(word2vec_init_ckpt_path)
    word2vec_init_ckpt = replace_keys(word2vec_init_ckpt["model_state_dict"], "embedding.", "")
else:
    word2vec_init_ckpt = word2vec_init_ckpt_path

num_features = sum(ADD_FEATURES.values())

TRAIN_CONFIG = {
    "ckpt_name": "random",
    "BATCH_SIZE": 64,
    "NUM_EPOCHS": 30,
    "LR": 1e-3,
    "PRETRAINED_LR": 1e-3,
    "LR_END": 1e-10, 
    "transformer_init_ckpt": "",
    "word2vec_init_ckpt": word2vec_init_ckpt,
    "optim": optim.AdamW,
    "scheduler_type": get_linear_schedule_with_warmup,
    "measure_train_f1": True,
}

is_sequence_labeler = True # False
num_tags = len(TOKEN_TAGS) if is_sequence_labeler else SPAN_TAGS

MODEL_CONFIG = {
    "num_tags": 2,
    "is_sequence_labeler": False,
    "model_type": "segment", # segment, tag, none
    "has_transformer": False,
    "has_word2vec": True,
    "has_char2word": False,
    "has_rnn": True,
    "has_crf": False,
    "num_features": num_features,
    "embedding_dropout": 0.1,
    "rnn_dropout": 0.0,
    "boundary_dropout": 0.0,
}
# ####################################################################################


# BEST BERT + SOFTMAX
####################################################################################
# transformer_init_ckpt_path = "ckpts/pretrain/bert_swmlm_3mil_best_val.pth"
# transformer_init_ckpt = torch.load(transformer_init_ckpt_path)
# transformer_init_ckpt = replace_keys(transformer_init_ckpt["model_state_dict"], "MLM.bert.", "")

# num_features = sum(ADD_FEATURES.values())

# TRAIN_CONFIG = {
#     "ckpt_name": "bert_sm",
#     "BATCH_SIZE": 64,
#     "NUM_EPOCHS": 30,
#     "LR": 1e-4,
#     "PRETRAINED_LR": 1e-4,
#     "LR_END": 1e-10, 
#     "transformer_init_ckpt": transformer_init_ckpt,
#     "word2vec_init_ckpt": "",
#     "optim": optim.AdamW,
#     "scheduler_type": get_linear_schedule_with_warmup,
#     "measure_train_f1": False,
# }

# MODEL_CONFIG = {
#     "num_tags": len(TOKEN_TAGS),
#     "is_sequence_labeler": True,
#     "has_transformer": True,
#     "has_word2vec": False,
#     "has_char2word": False,
#     "has_rnn": False,
#     "has_crf": False,
#     "num_features": num_features,
#     "embedding_dropout": 0.1,
#     "rnn_dropout": 0.0,
#     "boundary_dropout": 0.0,
# }
####################################################################################






# BEST WORD + CHAR + GRU + Biaffine
####################################################################################
# word2vec_init_ckpt_path = "ckpts/pretrain/mlm_full_title_best_val.pth"
# if word2vec_init_ckpt_path:
#     word2vec_init_ckpt = torch.load(word2vec_init_ckpt_path)
# else:
#     word2vec_init_ckpt = word2vec_init_ckpt_path

# num_features = sum(ADD_FEATURES.values())

# TRAIN_CONFIG = {
#     "ckpt_name": "biGRU_biaffine",
#     "BATCH_SIZE": 16,
#     "NUM_EPOCHS": 15,
#     "LR": 4e-3,
#     "PRETRAINED_LR": 4e-3,
#     "LR_END": 1e-10, 
#     "transformer_init_ckpt": "",
#     "word2vec_init_ckpt": word2vec_init_ckpt,
#     "optim": optim.AdamW,
#     "scheduler_type": get_linear_schedule_with_warmup,
#     "measure_train_f1": True,
# }

# MODEL_CONFIG = {
#     "num_tags": len(SPAN_TAGS),
#     "is_sequence_labeler": False,
#     "has_transformer": False,
#     "has_word2vec": True,
#     "has_char2word": True,
#     "has_rnn": True,
#     "has_crf": False,
#     "num_features": num_features,
#     "embedding_dropout": 0.2,
#     "rnn_dropout": 0.1,
#     "boundary_dropout": 0.1
# }
####################################################################################
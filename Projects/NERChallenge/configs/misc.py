import torch
from torch import optim
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

init_ckpt_path = "ckpts/pretrain/boundary_embeddings_optimized_best_val.pth"
if init_ckpt_path:
    init_ckpt = torch.load(init_ckpt_path)["model_state_dict"]
else:
    init_ckpt = init_ckpt_path
    
MLM_CONFIG = {
    "init_ckpt": init_ckpt,
    "ckpt_name": "random",
    "BATCH_SIZE": 64,
    "LR": 3e-5,
    "NUM_EPOCHS": 1,
    "MASK_PROPORTION": 0.15,
    "optim": optim.AdamW,
    "scheduler_type": get_cosine_schedule_with_warmup,
    "TEST_SIZE": 5000,
    "LOADER_SIZE": 5000,
}

W2V_CONFIG = {
    "init_ckpt": init_ckpt,
    "ckpt_name": "random",
    "BATCH_SIZE": 16,
    "LR": 5e-3,
    "NUM_EPOCHS": 1,
    "optim": optim.AdamW,
    "scheduler_type": get_cosine_schedule_with_warmup,
    "TEST_SIZE": 5000,
    "LOADER_SIZE": 5000,
}
import sys; import os; sys.path.append(os.path.dirname("./"))
from datetime import datetime; timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import os 
import argparse
import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from helpers.train import train_fn, eval_fn
from src.models.mlm import eBayModelForMaskedLM
from src.dataset import eBayMLM
from configs.model.transformer import TOKENIZER
from configs.misc import MLM_CONFIG
from configs.model.shared import DEVICE, DATA_DATASET_PATH, RANDOM_STATE

print(f"Cuda available: {torch.cuda.is_available()}\nDevice: {DEVICE}")

# Tokenizer
print(f"Tokenizer: {TOKENIZER.__class__.__name__}\n")

# Train validation split
data_valid_dataset = datasets.Dataset.load_from_disk(DATA_DATASET_PATH)
data_dataset, valid_dataset = train_test_split(data_valid_dataset, test_size=MLM_CONFIG["TEST_SIZE"], random_state=RANDOM_STATE)
data_dataset, valid_dataset = datasets.Dataset.from_dict(data_dataset), datasets.Dataset.from_dict(valid_dataset)
print(f"dataset_path = {DATA_DATASET_PATH}\n\nData: {data_dataset} Valid: {valid_dataset}")
valid_mld = eBayMLM(titles=valid_dataset["Clean_Title"], mask_proportion=MLM_CONFIG["MASK_PROPORTION"])

# Model
model = eBayModelForMaskedLM().to(DEVICE)
if not isinstance(MLM_CONFIG["init_ckpt"], str):
    model.load_state_dict(MLM_CONFIG["init_ckpt"])

# Other hyperparms
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
OPTIMIZER = MLM_CONFIG["optim"](optimizer_parameters, lr=MLM_CONFIG["LR"])
num_train_steps = int(len(data_dataset) / MLM_CONFIG["BATCH_SIZE"])
scheduler = MLM_CONFIG["scheduler_type"](
        OPTIMIZER, num_warmup_steps=0, num_training_steps=num_train_steps
    )

# Training loop
log_dir = f"./tensorboard_events/{MLM_CONFIG['ckpt_name']}_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
val_dataloader = DataLoader(valid_mld, batch_size=MLM_CONFIG["BATCH_SIZE"])

best_loss = float("inf")
delete_keys = []

start_step = 0
end_step = len(data_dataset)
for epoch in range(MLM_CONFIG["NUM_EPOCHS"]):
    for loader_step in range(start_step, end_step+MLM_CONFIG["LOADER_SIZE"], MLM_CONFIG["LOADER_SIZE"]): 
        curr_lr = OPTIMIZER.param_groups[0]['lr']
        random_indices = np.random.randint(0, end_step, MLM_CONFIG["LOADER_SIZE"])
        data_mld = eBayMLM(titles=data_dataset.select(random_indices)["Clean_Title"], mask_proportion=MLM_CONFIG["MASK_PROPORTION"])
        train_dataloader = DataLoader(data_mld, batch_size=MLM_CONFIG["BATCH_SIZE"], shuffle=True)
        train_loss = train_fn(train_dataloader, model, OPTIMIZER, DEVICE, scheduler=scheduler, delete_keys=delete_keys)
        with torch.no_grad():
            valid_loss = eval_fn(val_dataloader, model, DEVICE, delete_keys=delete_keys)
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": OPTIMIZER.state_dict(), "loader_step": loader_step}, 
                        f"ckpts/pretrain/{MLM_CONFIG['ckpt_name']}_best_val.pth")

        writer.add_scalars("training_measurements", {"train": train_loss, "val": valid_loss, "lr": curr_lr}, epoch*end_step + loader_step)
        print(f"Epoch {epoch}, Loader Step {loader_step}-{loader_step+MLM_CONFIG['LOADER_SIZE']}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, LR: {curr_lr:.5f}")
    start_step = 0

# Save ckpt where model ended
torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": OPTIMIZER.state_dict(), "loader_step": loader_step},
           f"ckpts/pretrain/{MLM_CONFIG['ckpt_name']}_final.pth")
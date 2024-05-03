import sys; import os; sys.path.append(os.path.dirname("./"))
from datetime import datetime; timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import os 
import argparse
import datasets
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from configs.model.word2vec import WORD_EMBEDDING_SIZE
from configs.misc import W2V_CONFIG
from src.models.word2vec import Word2VecModel
from task_helpers.encodings import word_counts
from src.dataset import eBayW2V
from configs.model.shared import DEVICE, DATA_DATASET_PATH, RANDOM_STATE
from helpers.train import train_fn, eval_fn

print(f"Cuda available: {torch.cuda.is_available()}\nDevice: {DEVICE}")

# Train validation split
from configs.model.shared import TRAIN_DATASET_PATH
data_valid_dataset = datasets.Dataset.load_from_disk(TRAIN_DATASET_PATH)
data_dataset, valid_dataset = train_test_split(data_valid_dataset, test_size=100, random_state=RANDOM_STATE)
data_dataset, valid_dataset = datasets.Dataset.from_dict(data_dataset), datasets.Dataset.from_dict(valid_dataset)
print(f"dataset_path = {DATA_DATASET_PATH}\n\nData: {data_dataset} Valid: {valid_dataset}")
valid_w2v = eBayW2V(titles=valid_dataset["Title"])

model = Word2VecModel(vocab_size=len(word_counts), embedding_dim=WORD_EMBEDDING_SIZE).to(DEVICE)
if not isinstance(W2V_CONFIG["init_ckpt"], str):
    model.load_state_dict(W2V_CONFIG["init_ckpt"])

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
OPTIMIZER = W2V_CONFIG["optim"](optimizer_parameters, lr=W2V_CONFIG["LR"])
num_train_steps = int(len(data_dataset) / W2V_CONFIG['BATCH_SIZE'])
scheduler = W2V_CONFIG["scheduler_type"](
        OPTIMIZER, num_warmup_steps=0, num_training_steps=num_train_steps
    )

# Training loop
log_dir = f"./tensorboard_events/{W2V_CONFIG['ckpt_name']}_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
val_dataloader = DataLoader(valid_w2v, batch_size=W2V_CONFIG['BATCH_SIZE'])

best_loss = float("inf")
delete_keys = []

start_step = 0
end_step = len(data_dataset)
for epoch in range(W2V_CONFIG['NUM_EPOCHS']):
    for loader_step in range(start_step, end_step+W2V_CONFIG['LOADER_SIZE'], W2V_CONFIG['LOADER_SIZE']): 
        curr_lr = OPTIMIZER.param_groups[0]['lr']
        random_indices = np.random.randint(0, end_step, W2V_CONFIG['LOADER_SIZE'])
        data_w2v = eBayW2V(titles=data_dataset.select(random_indices)["Clean_Title"])
            
        train_dataloader = DataLoader(data_w2v, batch_size=W2V_CONFIG['BATCH_SIZE'], shuffle=True)
        train_loss = train_fn(train_dataloader, model, OPTIMIZER, DEVICE, scheduler=scheduler, delete_keys=delete_keys)
        with torch.no_grad():
            valid_loss = eval_fn(val_dataloader, model, DEVICE, delete_keys=delete_keys)
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": OPTIMIZER.state_dict(), "loader_step": loader_step}, 
                        f"ckpts/pretrain/{W2V_CONFIG['ckpt_name']}_best_val.pth")

        writer.add_scalars("training_measurements", {"train": train_loss, "val": valid_loss, "lr": curr_lr}, epoch*end_step + loader_step)
        print(f"Epoch {epoch}, Loader Step {loader_step}-{loader_step+W2V_CONFIG['LOADER_SIZE']}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, LR: {curr_lr:.5f}")
    
    start_step = 0
    # Save ckpt where model ended
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": OPTIMIZER.state_dict(), "loader_step": loader_step},
            f"ckpts/pretrain/{W2V_CONFIG['ckpt_name']}_final.pth")
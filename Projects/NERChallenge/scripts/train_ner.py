import sys; import os; sys.path.append(os.path.dirname("./"))
from datetime import datetime; timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import os 
import argparse
import datasets
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from configs.model.shared import DEVICE, TRAIN_DATASET_PATH, VALID_SPLIT, RANDOM_STATE
from configs.model.transformer import TOKENIZER
from configs.config import MODEL_CONFIG, TRAIN_CONFIG
from helpers.train import train_fn, eval_fn
from src.models.end2end import NERModel
from src.dataset import eBayNERD
from task_helpers.data_split import stratify_train_test_split
from src.inference import sequence_get_data, biaffine_get_data, tag_get_data, segment_get_data
from task_helpers.evaluation import sequence_get_tag_stats, biaffine_get_tag_stats, tag_get_tag_stats, segment_get_tag_stats

if MODEL_CONFIG["is_sequence_labeler"]:
    get_data = sequence_get_data
    get_tag_stats = sequence_get_tag_stats
else:
    get_data = biaffine_get_data
    get_tag_stats = biaffine_get_tag_stats
    
if MODEL_CONFIG["model_type"] == "segment":
    get_data = segment_get_data
    get_tag_stats = segment_get_tag_stats
elif MODEL_CONFIG["model_type"] == "tag":
    get_data = tag_get_data
    get_tag_stats = tag_get_tag_stats

print(f"Cuda available: {torch.cuda.is_available()}\nDevice: {DEVICE}")
print(f"Tokenizer: {TOKENIZER.__class__.__name__}\n")

# Train validation split
train_valid_dataset = datasets.Dataset.load_from_disk(TRAIN_DATASET_PATH)
train_dataset, valid_dataset, ent_weights = stratify_train_test_split(train_valid_dataset, test_size=VALID_SPLIT, random_state=RANDOM_STATE)
print(f"dataset_path = {TRAIN_DATASET_PATH}\n\nTrain: {train_dataset} Valid: {valid_dataset}")
# TODO: Training data augmentation
train_nerd = eBayNERD(titles=train_dataset["Clean_Title"], tags=train_dataset["Tags"])
valid_nerd = eBayNERD(titles=valid_dataset["Clean_Title"], tags=valid_dataset["Tags"])

# Model
model = NERModel(MODEL_CONFIG).to(DEVICE)

if not isinstance(TRAIN_CONFIG["transformer_init_ckpt"], str):
    model.embedding_module.transformer.load_state_dict(TRAIN_CONFIG["transformer_init_ckpt"], strict=False)
    
if not isinstance(TRAIN_CONFIG["word2vec_init_ckpt"], str):
    model.embedding_module.word2vec.load_state_dict(TRAIN_CONFIG["word2vec_init_ckpt"], strict=False)
    
# for param in model.transformer.parameters():
#     param.requires_grad = False
    
# for param in model.parameters():
#     param.requires_grad = False
    
# for param in model.transformer.encoder.layer[-8:].parameters():
#     param.requires_grad = True

# for param in model.crf.parameters():
#     param.requires_grad = True

# Other hyperparams
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
pretrained = ["transformer", "word2vec"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and any([k in n for k in pretrained])
        ],
        "weight_decay": 0.001,
        "lr": TRAIN_CONFIG["PRETRAINED_LR"],
    },
    {
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay) and any([k in n for k in pretrained])
        ],
        "weight_decay": 0.0,
        "lr": TRAIN_CONFIG["PRETRAINED_LR"],
    },
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any([k in n for k in pretrained])
        ],
        "weight_decay": 0.001,
        "lr": TRAIN_CONFIG["LR"],
    },
    {
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any([k in n for k in pretrained])
        ],
        "weight_decay": 0.0,
        "lr": TRAIN_CONFIG["LR"],
    },
]
OPTIMIZER = TRAIN_CONFIG["optim"](optimizer_parameters)
num_train_steps = int(len(train_nerd) / TRAIN_CONFIG["BATCH_SIZE"] * TRAIN_CONFIG["NUM_EPOCHS"])
scheduler = TRAIN_CONFIG["scheduler_type"](OPTIMIZER, num_warmup_steps=0, num_training_steps=num_train_steps)

# Training loop
log_dir = f"./tensorboard_events/{TRAIN_CONFIG['ckpt_name']}_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

train_dataloader = DataLoader(train_nerd, batch_size=TRAIN_CONFIG["BATCH_SIZE"], shuffle=True)
val_dataloader = DataLoader(valid_nerd, batch_size=TRAIN_CONFIG["BATCH_SIZE"])

best_loss = float("inf")
best_wf1 = 0
delete_keys = []
for epoch in range(TRAIN_CONFIG["NUM_EPOCHS"]): 
    curr_lr_pretrained = OPTIMIZER.param_groups[0]["lr"]
    curr_lr_initialized = OPTIMIZER.param_groups[-1]["lr"]
    # Load dataloaders and progress bar each epoch
    train_loss = train_fn(train_dataloader, model, OPTIMIZER, DEVICE, scheduler=scheduler, delete_keys=delete_keys)

    with torch.no_grad():
        valid_loss = eval_fn(val_dataloader, model, DEVICE, delete_keys=delete_keys)

    with torch.no_grad():
        if TRAIN_CONFIG["measure_train_f1"]:
            data = get_data(model, train_dataset, delete_keys)
            tag_stats, train_f1, train_wf1 = get_tag_stats(data)
        else:
            train_f1 = train_wf1 = 0.00
        
        data = get_data(model, valid_dataset, delete_keys)
        tag_stats, valid_f1, valid_wf1 = get_tag_stats(data)
        
    if valid_wf1 > best_wf1:
        best_wf1 = valid_wf1
        torch.save(model.state_dict(), f"ckpts/end2end/{TRAIN_CONFIG['ckpt_name']}_best_val.pth")

    writer.add_scalars("training_measurements", {"train": train_loss, 
                                                 "val": valid_loss, 
                                                 "train_f1": train_f1,
                                                 "val_f1": valid_f1, 
                                                 "train_wf1": train_wf1,
                                                 "val_wf1": valid_wf1}, epoch)
    
    print(f"Epoch {epoch+1}/{TRAIN_CONFIG['NUM_EPOCHS']}, curr_lr_pre: {curr_lr_pretrained:.6f}, curr_lr_init: {curr_lr_initialized:.6f}")
    print(f"    train/valid loss: {train_loss:.4f}/{valid_loss:.4f}")
    print(f"    train/valid f1:   {train_f1:.4f}/{valid_f1:.4f}")
    print(f"    train/valid wf1:  {train_wf1:.4f}/{valid_wf1:.4f}")
    print()

    # Save ckpt where model ended
    torch.save(model.state_dict(), f"ckpts/end2end/{TRAIN_CONFIG['ckpt_name']}_final.pth")
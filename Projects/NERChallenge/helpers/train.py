from tqdm import tqdm

def train_fn(dataloader, model, optimizer, device, scheduler=None, delete_keys=[], final_lr=0) -> float:
    model.train()
    final_loss = 0
    progress_bar = tqdm(total=len(dataloader))
    for data in dataloader:
        for k, v in list(data.items()):
            if k in delete_keys:
                del data[k]
            else:
                data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        if scheduler and scheduler.get_last_lr()[0] > final_lr:
            scheduler.step()
        final_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
    return final_loss / len(dataloader)

def eval_fn(dataloader, model, device, delete_keys=[]) -> float:
    model.eval()
    final_loss = 0
    progress_bar = tqdm(total=len(dataloader))
    for data in dataloader:
        for k, v in list(data.items()):
            if k in delete_keys:
                del data[k]
            else:
                data[k] = v.to(device)
        _, loss = model(**data)
        
        final_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
    return final_loss / len(dataloader)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from task_helpers.tags import id2tag, SPAN_TAGS
from src.dataset import eBayNERD
from configs.model.shared import DEVICE 
from configs.model.dataset import NESTED_NER
from configs.model.inference import VALID_BATCH_SIZE, TEST_BATCH_SIZE
from configs.model.word2vec import MAX_WORD_LEN
from configs.config import MODEL_CONFIG
from scipy.stats import mode
from task_helpers.evaluation import get_final_tags_span

INVALID_INPUT_ERROR = "Must provide either titles or model inputs"

def biaffine_infer_entities(model, titles=None, model_inputs=None):
    assert titles or model_inputs, INVALID_INPUT_ERROR
    if isinstance(titles, str):
        titles = [titles]

    if not model_inputs:
        tokens = [title.split() for title in titles]
        for model_inputs in DataLoader(eBayNERD(tokens=tokens), batch_size=TEST_BATCH_SIZE):
            for k, v in list(model_inputs.items()):
                model_inputs[k] = v.to(DEVICE)

    word_mask = model_inputs["word_mask"]
    logits, loss = model(**model_inputs)
    span_probs = torch.nn.functional.softmax(logits, dim=-1)
    total_entities = []
    for i in range(span_probs.size(0)):
        entities = []
        spans = (span_probs[i] * word_mask[i].unsqueeze(-1)).cpu()
        span_classes = spans.argmax(-1)
        span_scores = spans[torch.arange(MAX_WORD_LEN), torch.arange(MAX_WORD_LEN), span_classes]
        values, indices = (((span_classes != 0) * span_scores).view(-1)).sort(descending=True) # Ignore No Tags
        indices = indices[values > 0]
        values = values[values > 0]
        for index, value in zip(indices, values):
            start, end = np.unravel_index(index, (MAX_WORD_LEN, MAX_WORD_LEN))
            category = span_classes[start,end].item()
            for (other_start, other_end, _) in entities: 
                if (start < other_start <= end < other_end) or (other_start < start <= other_end < end): # NESTED NER
                    break
                if NESTED_NER:
                    continue
                if (start <= other_start <= other_end <= end) or (other_start <= start <= end <= other_end): # FLAT NER
                    break
            else:
                entities.append((start, end, category))
        entities.sort()
        total_entities.append(entities)
    return span_probs, loss, total_entities

def biaffine_get_data(model, labeled_dataset, delete_keys=[]):
    dataloader = DataLoader(eBayNERD(titles=labeled_dataset["Clean_Title"], tags=labeled_dataset["Tags"]), batch_size=VALID_BATCH_SIZE)
    title_entities = biaffine_get_title_preds(model, dataloader, delete_keys)
    labeled_entities = []
    for i in range(len(labeled_dataset)):
        labeled_entities.append(get_final_tags_span(labeled_dataset[i]["Tags"]))
    return (labeled_entities, title_entities)

def biaffine_get_title_preds(model, dataloader, delete_keys=[]):
    title_entities = []
    entity_probs = []
    losses = []
    for model_inputs in tqdm(dataloader, total=len(dataloader)):
        for k, v in list(model_inputs.items()):
            if k in delete_keys:
                del model_inputs[k]
                continue
            model_inputs[k] = v.to(DEVICE)
        span_probs, loss, curr_title_entities = biaffine_infer_entities(model, model_inputs=model_inputs)
        losses.append(loss.item())
        entity_probs.append(span_probs)
        title_entities += curr_title_entities
    entity_probs = torch.vstack(entity_probs)
    losses = torch.tensor(losses)
    return title_entities

def sequence_infer_entities(model, titles=None, model_inputs=None):
    assert titles or model_inputs, INVALID_INPUT_ERROR
    if isinstance(titles, str):
        titles = [titles]

    if not model_inputs:
        tokens = [title.split() for title in titles]
        for model_inputs in DataLoader(eBayNERD(titles=tokens), batch_size=TEST_BATCH_SIZE):
            for k, v in list(model_inputs.items()):
                model_inputs[k] = v.to(DEVICE)
                
    word_mask = model_inputs["word_mask"]
    logits, loss = model(**model_inputs)
    if MODEL_CONFIG["has_crf"]:
        logits, loss, preds = model(**model_inputs, eval=True)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs, preds, loss
    else:
        logits, loss = model(**model_inputs)
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    instances = logits.size(0)
    indices = probs.argmax(dim=-1)
    preds = [[id2tag(indices[i][j]) for j in range(word_mask[i][0].sum())] for i in range(instances)]
    return probs, preds, loss

def sequence_get_data(model, labeled_dataset, delete_keys=[], quiz=False):
    titles = labeled_dataset["Clean_Title"]
    if quiz: 
       tags = [["%_No Tag" for _ in range(len(elem.split()))] for elem in titles]
    else:
        tags = labeled_dataset["Tags"]
        
    dataloader = DataLoader(eBayNERD(titles=titles, tags=tags), batch_size=VALID_BATCH_SIZE)
    title_preds = sequence_get_title_preds(model, dataloader, delete_keys)

    data = []
    for i, instance in enumerate(labeled_dataset):
        title_pred = title_preds[i]
        padding_len = len(tags[i]) - len(title_pred)
        title_pred.extend(["No Tag"] * padding_len)
        data.append((instance["Clean_Title"].split(), tags[i], title_pred))
    return data

def sequence_get_title_preds(model, dataloader, delete_keys=[]):
    title_preds = []
    for model_inputs in tqdm(dataloader, total=len(dataloader)):
        for k, v in list(model_inputs.items()):
            if k in delete_keys:
                del model_inputs[k]
                continue
            model_inputs[k] = v.to(DEVICE)
        _, curr_title_preds, _ = sequence_infer_entities(model, model_inputs=model_inputs)
        title_preds += curr_title_preds
    return title_preds

def tag_get_data(model, labeled_dataset, delete_keys=[], quiz=False):
    pass

def segment_get_data(model, labeled_dataset, delete_keys=[]):
    dataloader = DataLoader(eBayNERD(titles=labeled_dataset["Clean_Title"], tags=labeled_dataset["Tags"]), batch_size=VALID_BATCH_SIZE)
    title_entities = segment_get_title_preds(model, dataloader, delete_keys)
    labeled_entities = []
    for i in range(len(labeled_dataset)):
        labeled_entities.append(get_final_tags_span(labeled_dataset[i]["Tags"]))
    return (labeled_entities, title_entities)

def segment_get_title_preds(model, dataloader, delete_keys=[]):
    title_entities = []
    entity_probs = []
    losses = []
    for model_inputs in tqdm(dataloader, total=len(dataloader)):
        for k, v in list(model_inputs.items()):
            if k in delete_keys:
                del model_inputs[k]
                continue
            model_inputs[k] = v.to(DEVICE)
        span_probs, loss, curr_title_entities = biaffine_infer_entities(model, model_inputs=model_inputs)
        losses.append(loss.item())
        entity_probs.append(span_probs)
        title_entities += curr_title_entities
    entity_probs = torch.vstack(entity_probs)
    losses = torch.tensor(losses)
    return title_entities

def segment_infer_entities(model, titles=None, model_inputs=None):
    assert titles or model_inputs, INVALID_INPUT_ERROR
    if isinstance(titles, str):
        titles = [titles]

    if not model_inputs:
        tokens = [title.split() for title in titles]
        for model_inputs in DataLoader(eBayNERD(tokens=tokens), batch_size=TEST_BATCH_SIZE):
            for k, v in list(model_inputs.items()):
                model_inputs[k] = v.to(DEVICE)

    word_mask = model_inputs["word_mask"]
    logits, loss = model(**model_inputs)
    span_probs = torch.nn.functional.sigmoid(logits.squeeze(-1))
    total_entities = []
    thres = 0.5
    for i in range(span_probs.size(0)):
        entities = []
        spans = (span_probs[i] * word_mask[i]).cpu()
        span_classes = (spans > thres).to(int)
        values, indices = (((span_classes != 0) * spans).view(-1)).sort(descending=True) # Ignore No Tags
        indices = indices[values > 0]
        values = values[values > 0]
        for index, value in zip(indices, values):
            start, end = np.unravel_index(index, (MAX_WORD_LEN, MAX_WORD_LEN))
            for (other_start, other_end) in entities: 
                if (start < other_start <= end < other_end) or (other_start < start <= other_end < end): # NESTED NER
                    break
                if NESTED_NER:
                    continue
                if (start <= other_start <= other_end <= end) or (other_start <= start <= end <= other_end): # FLAT NER
                    break
            else:
                entities.append((start, end))
        entities.sort()
        total_entities.append(entities)
    return span_probs, loss, total_entities
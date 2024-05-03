import torch
from torch import nn 

def ce_loss(logits, labels, masks, num_classes):
    loss = nn.CrossEntropyLoss()
    active_loss = masks.reshape(-1) == 1
    active_logits = logits.reshape(-1, num_classes)
    active_labels = torch.where(
        active_loss,
        labels.view(-1),
        torch.tensor(loss.ignore_index).type_as(labels)
    )
    return loss(active_logits, active_labels)

def ce_loss_from_probs(probs, labels, masks, num_classes):
    loss = nn.NLLLoss()
    active_loss = masks.reshape(-1) == 1
    active_log_probs = torch.log(probs).reshape(-1, num_classes)
    active_labels = torch.where(
        active_loss,
        labels.view(-1),
        torch.tensor(loss.ignore_index).type_as(labels)
    )
    return loss(active_log_probs, active_labels)

def ce_loss_from_probs_labels(probs, labels, masks, num_classes, class_weights=None, focal_loss=False):
    active_loss = masks.reshape(-1) == 1
    active_probs = probs.view(-1, num_classes)[active_loss]
    active_labels = labels.view(-1, num_classes)[active_loss]
    if focal_loss:
        return -((1-active_probs.view(-1, 36)) * (active_labels.view(-1, 36) * torch.log(active_probs.view(-1, 36)) + 1e-10)).sum() / active_loss.sum()
    return -((active_labels.view(-1, 36) * torch.log(active_probs.view(-1, 36)) + 1e-10)).sum() / active_loss.sum()

def soft_f1_loss(probs, labels, masks, num_classes, class_weights=None):
    def f1_loss_class(class_probs, binary_labels): 
        tp, tn = (class_probs * binary_labels).sum(), ((1-class_probs) * (1-binary_labels)).sum()
        fp, fn = (class_probs * (1-binary_labels)).sum(), ((1-class_probs) * binary_labels).sum()
        p, r = tp / (tp + fp + 1e-10), tp / (tp + fn + 1e-10)
        f1 = (2 * p * r) / (p + r + 1e-10)
        return f1.mean()
    
    score = 0
    for i in range(0, num_classes):
        p_i = (probs[:,:,:,i] * masks)
        y_i = (labels[:,:,:,i] * masks).to(float)
        score += (class_weights[i] if class_weights != None else 1/(num_classes)) * f1_loss_class(p_i, y_i)
    return 1.0 - score

def soft_jaccard_loss(probs, labels, masks, num_classes, class_weights=None):
    def jaccard_loss_class(class_probs, binary_labels): 
        intersection = (class_probs * binary_labels).sum()
        union = class_probs.sum() + binary_labels.sum()
        return 2 * intersection / (union + 1e-10)
    
    score = 0
    for i in range(1, num_classes):
        p_i = (probs[:,:,:,i] * masks)
        y_i = (labels[:,:,:,i] * masks).to(float)
        score += (class_weights[i] if class_weights is not None else 1/(num_classes)) * jaccard_loss_class(p_i, y_i)
    return 1.0 - score

def soft_jaccard_loss_seg(logits, labels, masks, num_classes, class_weights=None):
    def jaccard_loss_class(class_probs, binary_labels): 
        intersection = (class_probs * binary_labels).sum()
        union = class_probs.sum() + binary_labels.sum()
        return 2 * intersection / (union + 1e-10)
    probs = torch.nn.functional.sigmoid(logits.squeeze(-1))
    score = jaccard_loss_class(probs*masks, labels*masks)
    return 1 - score
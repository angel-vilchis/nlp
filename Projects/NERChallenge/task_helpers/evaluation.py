from collections import defaultdict
import numpy as np
from task_helpers.tags import ASPECT_NAMES, ASPECT_WEIGHTS, CONTINUE_MARKING, END_MARKING, ALONE_MARKING, tag2id, id2tag, id2spantag, get_raw_tag_id, id_marking_is_start, id_marking_is_end

def biaffine_get_tag_stats(final_tags):
    tag_stats = defaultdict(lambda: {"predicted": 0, "labeled": 0, "predicted_and_labeled": 0})
    labels, preds = final_tags
    for i in range(len(labels)):
        final_labels, final_preds = labels[i], preds[i]
        for (_, _, tag_id) in final_labels:
            tag = id2spantag(tag_id)
            tag_stats[tag]["labeled"] += 1
        for (_, _, tag_id) in final_preds:
            tag = id2spantag(tag_id)
            tag_stats[tag]["predicted"] += 1
        for final_label in final_labels:
            if final_label in final_preds:
                tag_id = final_label[-1] 
                tag = id2spantag(tag_id)
                tag_stats[tag]["predicted_and_labeled"] += 1
                final_preds.pop(final_preds.index(final_label))
    return metrics(tag_stats)

def get_final_tags_span(tags):
    entities = []
    for i, tag in enumerate(tags):
        id = tag2id(tag)
        raw_id = get_raw_tag_id(id)
        if id_marking_is_start(id):
            curr_start = i
        if id_marking_is_end(id):
            if raw_id != 0:
                curr_end = i
                entities.append((curr_start, curr_end, raw_id))
            curr_start = None
    return entities

def sequence_get_tag_stats(data):
    tag_stats = defaultdict(lambda: {"predicted": 0, "labeled": 0, "predicted_and_labeled": 0}) 
    for words, labels, preds in data:
        assert len(words) == len(labels) == len(preds), "There is an error, data lengths are different."
        final_labels = get_final_tags_word(words, labels)
        final_preds = get_final_tags_word(words, preds)
        for (tag, _) in final_labels:
            tag_stats[tag]["labeled"] += 1
        for (tag, _) in final_preds:
            tag_stats[tag]["predicted"] += 1
        for final_label in final_labels:
            if final_label in final_preds: 
                tag = final_label[0]
                tag_stats[tag]["predicted_and_labeled"] += 1
                final_preds.pop(final_preds.index(final_label))
    return metrics(tag_stats)

def get_final_tags_word(words, tags, reverse=False, both=False):
    final = []
    for i in range(len(tags)):
        if (tags[i][0] == CONTINUE_MARKING or tags[i][0] == END_MARKING) and len(final):
            final[-1][1] += f" {words[i]}"
        elif (tags[i][0] == ALONE_MARKING):
            final.append([tags[i][2:], words[i]])
        else:
            final.append([tags[i][2:], words[i]])
            
    # Reverse
    final_reverse = []
    for i in range(len(tags)):
        if (tags[-(i+1)][0] == CONTINUE_MARKING or tags[-(i+1)][1] != "_") and len(final_reverse):
            final_reverse[-1][1] =  f"{words[-(i+1)]} " + final_reverse[-1][1]
        elif (tags[-(i+1)][0] == ALONE_MARKING):
            final_reverse.append([tags[-(i+1)][2:], words[-(i+1)]])
        else:
            final_reverse.append([tags[-(i+1)][2:], words[-(i+1)]])
            
    if reverse:
        final.reverse()
        return final_reverse
    if both:
        return final, final_reverse
            
    return final

def metrics(tag_stats):
    total_count = sum([tag_stats[tag]["labeled"] for tag in ASPECT_NAMES]) 
    for tag in tag_stats.keys():
        tag_stats[tag]["precision"] = precision = tag_stats[tag]["predicted_and_labeled"] / tag_stats[tag]["predicted"] if tag_stats[tag]["predicted"] else 0
        tag_stats[tag]["recall"] = recall = tag_stats[tag]["predicted_and_labeled"] / tag_stats[tag]["labeled"] if tag_stats[tag]["labeled"] else 0
        tag_stats[tag]["f1_score"] = f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0 
        
        if tag in ASPECT_NAMES:
            weight = ASPECT_WEIGHTS[tag]
            # weight = tag_stats[tag]["labeled"] / total_count
            tag_stats[tag]["weight"] = weight
            tag_stats[tag]["weighted_f1_score"] = (f1 * weight)

    fold_f1_score = np.mean([tag_stats[tag]["f1_score"] for tag in ASPECT_NAMES])
    fold_weighted_f1_score = sum([tag_stats[tag]["weighted_f1_score"] for tag in ASPECT_NAMES])
    return tag_stats, fold_f1_score, fold_weighted_f1_score

def tag_get_tag_stats():
    pass

def segment_get_tag_stats(final_tags):
    tag_stats = {"predicted": 0, "labeled": 0, "predicted_and_labeled": 0}
    labels, preds = final_tags
    for i in range(len(labels)):
        final_labels, final_preds = labels[i], preds[i]
        final_labels = [(elem[0], elem[1]) for elem in final_labels]
        final_preds = [(elem[0], elem[1]) for elem in final_preds]
        tag_stats["labeled"] += len(final_labels)
        tag_stats["predicted"] += len(final_preds)
        for final_label in final_labels:
            if final_label in final_preds:
                tag_stats["predicted_and_labeled"] += 1
                final_preds.pop(final_preds.index(final_label))
    return segment_metrics(tag_stats)

def segment_metrics(tag_stats):
    tag_stats["precision"] = precision = tag_stats["predicted_and_labeled"] / tag_stats["predicted"] if tag_stats["predicted"] else 0
    tag_stats["recall"] = recall = tag_stats["predicted_and_labeled"] / tag_stats["labeled"] if tag_stats["labeled"] else 0
    tag_stats["f1_score"] = f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0 
    return tag_stats, 0.00, f1
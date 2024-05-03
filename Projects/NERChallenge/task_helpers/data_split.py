import numpy as np 
from .tags import SPAN_TAGS, tag2id, get_raw_tag_id, id_marking_is_start

UNSUCCESSFUL_STRATIFY_ERROR = "Stratify unsuccessful"

def stratify_train_test_split(dataset, test_size, random_state):
    random_state = np.random.RandomState(seed=random_state)
    entity_counts_per_title = [[0 for _ in range(len(SPAN_TAGS))] for _ in range(len(dataset))]
    for i in range(len(dataset)):
        for tag in dataset[i]["Tags"]:
            id = tag2id(tag)
            raw_id = get_raw_tag_id(id)
            
            if id_marking_is_start(id):
                entity_counts_per_title[i][raw_id] += 1
    entity_counts_per_title = np.array(entity_counts_per_title)
    target_dist = entity_counts_per_title.sum(axis=0) / entity_counts_per_title.sum()

    split_train_indices = []
    split_train_counts = []
    
    split_valid_indices = []
    split_valid_counts = []
    
    sorted_entities = np.argsort(target_dist)
    for i in sorted_entities:
        indices = np.nonzero(entity_counts_per_title[:,i] != 0)[0]
        random_state.shuffle(indices)
        values = entity_counts_per_title[indices]
        if random_state.rand() >= 0.5:
            method = np.ceil
        else:
            method = np.floor
        split_valid_indices += indices[:int(method(len(indices) * test_size))].tolist()
        split_valid_counts += values[:int(method(len(indices) * test_size))].tolist()
        split_train_indices += indices[int(method(len(indices) * test_size)):].tolist()
        split_train_counts += values[int(method(len(indices) * test_size)):].tolist()
        entity_counts_per_title[indices] = 0
        
    split_valid_counts = np.array(split_valid_counts)
    split_train_counts = np.array(split_train_counts)
    
    valid_dist = split_valid_counts.sum(axis=0) / split_valid_counts.sum()
    train_dist = split_train_counts.sum(axis=0) / split_train_counts.sum()
    assert np.linalg.norm(valid_dist-target_dist) < 0.007, UNSUCCESSFUL_STRATIFY_ERROR
    assert np.linalg.norm(train_dist-target_dist) < 0.002, UNSUCCESSFUL_STRATIFY_ERROR
    
    return dataset.select(split_train_indices), dataset.select(split_valid_indices), target_dist
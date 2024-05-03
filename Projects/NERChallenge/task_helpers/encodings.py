import json

WORD_PAD_TOKEN = "<PAD>"
WORD_OOV_TOKEN = "<RARE>"
WORD_OOV_ID = 1

CHAR_PAD_TOKEN = ""
CHAR_OOV_TOKEN = ""
CHAR_OOV_ID = 1

with open('data/own/word_counts.json', 'r') as json_file:
    word_counts = json.load(json_file)
    
with open('data/own/word2id.json', 'r') as json_file:
    word2id = json.load(json_file)
    id2word = {v: k for k, v in word2id.items()}
    
with open('data/own/char_counts.json', 'r') as json_file:
    char_counts = json.load(json_file)
    
with open('data/own/char2id.json', 'r') as json_file:
    char2id = json.load(json_file)
    id2char = {v: k for k, v in char2id.items()}

def normalize(word):
    return word.lower()

def get_id_from_word(word):
    return word2id[word] if word in word2id else WORD_OOV_ID

def get_word_from_id(word_id):
    if word_id == 0:
        return WORD_PAD_TOKEN
    if word_id == 1 or word_id not in id2word: 
        return WORD_OOV_TOKEN
    return id2word[word_id]

def get_words_from_ids(word_ids):
    return [get_word_from_id(id) for id in word_ids]

def get_id_from_char(char):
    return char2id[char] if char in char2id else CHAR_OOV_ID

def get_char_from_id(char_id):
    if char_id == 0:
        return CHAR_PAD_TOKEN
    if char_id == 1 or char_id not in id2char: 
        return CHAR_OOV_TOKEN
    return id2char[char_id]
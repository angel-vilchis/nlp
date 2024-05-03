from transformers import AutoTokenizer

MAX_BERT_LEN = 50

transformers = ["bert-base-german-dbmdz-uncased", 
                "bert-base-german-dbmdz-cased", 
                "deepset/gbert-large",
                "xlm-roberta-large-finetuned-conll03-german",  
                "xlm-roberta-base",
                "xlm-roberta-large",
                "mschiesser/ner-bert-german", 
                "dbmdz/german-gpt2",
                "./ckpts/tokenizer/tokenizer-trained.json", 
                "Davlan/xlm-roberta-base-ner-hrl",
                "Davlan/xlm-roberta-large-ner-hrl",
                "Davlan/xlm-roberta-base-wikiann-ner", 
                "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
                "flair/ner-german-large"]

types = ["bert", "roberta", "xlmroberta", "gpt"]

MODEL_NAME = transformers[0]
MODEL_TYPE = types[0]

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

if MODEL_TYPE != "xlmroberta":
    SOS_TOKEN = TOKENIZER.cls_token_id
    EOS_TOKEN = TOKENIZER.sep_token_id
    PAD_TOKEN = TOKENIZER.pad_token_id
    MASK_TOKEN = TOKENIZER.mask_token_id 
else:
    SOS_TOKEN = TOKENIZER.bos_token_id
    EOS_TOKEN = TOKENIZER.eos_token_id
    PAD_TOKEN = TOKENIZER.pad_token_id
    MASK_TOKEN = TOKENIZER.mask_token_id 
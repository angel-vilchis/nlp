from tags import NO_TAG, ALONE_MARKING
from configs.model.features import GAZETEER

NO_TAG_SYMBOLS = "~!@#$%^*()_-+=[]{}|;',.<>?/"

def title_rules(title: str) -> str:
    pass

def token_rules(token: str) -> str:
    if (token in set(char for char in NO_TAG_SYMBOLS)):
        return ALONE_MARKING + "_" + NO_TAG
    for key in GAZETEER:
        if token.lower() in GAZETEER[key]:
            return ALONE_MARKING + "_" + key
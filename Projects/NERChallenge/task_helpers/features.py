from configs.model.features import ADD_FEATURES
from helpers.cleaning import is_numeric

NUM_FEATURES = sum(ADD_FEATURES.values())

def token_features(token):
    if token == None:
        return [0.0 for _ in range(NUM_FEATURES)], NUM_FEATURES
    vec = []
    is_alpha, is_char = token.isalpha(), len(token) <= 1
    is_uppercased, is_lowercased, is_titlecased = token.isupper() and is_alpha, token.islower() and is_alpha, token.istitle() and is_alpha and (not is_char)
    is_mixedcased = (not is_uppercased) and (not is_lowercased) and (not is_titlecased) and (is_alpha)
    is_number = token.isnumeric() or (is_numeric(token))
    is_alnum = (not is_number) & (not is_alpha) & (token.isalnum())
    not_alnum = (not is_number) & (not is_alpha) & (not is_alnum) & (not is_char)
    misc = (not is_number) & (not is_alpha) & (not is_alnum) & (is_char)
    i = 0
    id = -1
    for k in ADD_FEATURES.keys():
        if ADD_FEATURES[k]:
            if id == -1 and eval(f"{k}"):
                id = i
            else:
                i += 1
            eval(f"vec.append({k})")
    if id == -1:
        id = NUM_FEATURES
    return vec, id
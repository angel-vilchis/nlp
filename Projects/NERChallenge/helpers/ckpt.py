def replace_keys(model_state_dict, substring, new_substring):
    new_state_dict = {}
    for key, value in model_state_dict.items():
        new_key = key.replace(substring, new_substring)
        new_state_dict[new_key] = value
    return new_state_dict

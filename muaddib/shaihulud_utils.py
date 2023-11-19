import json

import keras_core


def load_file(path):
    with open(path) as f:
        return f.read()


def load_json_dict(path):
    return json.loads(load_file(path))


def write_dict_to_file(dict_to_save, path):
    with open(path, "w") as f:
        json.dump(dict_to_save, f)
    return

def read_model_conf(path):
    return keras_core.models.model_from_json(load_file(path))

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


def open_model(path, custom_objects=None, compile=True, safe_mode=True):
    if path.endswith("json"):
        return read_model_conf(path)
    elif path.endswith("keras"):
        return keras_core.models.load_model(
            path,
            custom_objects=custom_objects,
            compile=compile,
            safe_mode=safe_mode,
        )


def get_target_dict(target_variable):
    final_targets = {}
    multiple_target = target_variable.split("|")
    for tag in multiple_target:
        sorteg_tag = sorted(tag.split(";"))
        if len(sorteg_tag) > 1:
            tag_name = "".join([f.title()[:3] for f in sorteg_tag])
        else:
            tag_name = sorteg_tag[0]
        if tag_name not in final_targets:
            final_targets[tag_name] = sorteg_tag

    return final_targets

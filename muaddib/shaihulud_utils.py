import json

import keras_core


def is_jsonable(x):
    if isinstance(x, list):
        return all(is_jsonable(item) for item in x)
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


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
    return keras_core.models.model_from_json(load_json_dict(path))


def open_model(path, custom_objects=None, compile_arg=True, safe_mode=True):
    if path.endswith("json"):
        return read_model_conf(path)
    elif path.endswith("keras"):
        return keras_core.models.load_model(
            path,
            custom_objects=custom_objects,
            compile=compile_arg,
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


def get_mirror_weight_loss(loss_name):
    loss_name2 = loss_name.replace("reversed_", "")
    loss_used = loss_name2.replace("mirror_weights_", "")
    loss_used = loss_used.replace("mirror_loss_", "")
    loss_used = loss_used.replace("mirror_percentage_", "")
    loss_used = loss_used.replace("mirror_normalized_", "")

    from alquitable.advanced_losses import MirrorWeights, MirrorLoss,MirrorPercentage,MirrorNormalized
    from alquitable.losses import ALL_LOSSES_DICT

    if "mirror" in loss_name:
        weight_on_surplus = True
        if "reversed" in loss_name:
            weight_on_surplus = False
        words = loss_used.split("_")
        words = [w.title() for w in words]
        loss_used = "".join(words)
    if len(loss_used.split("_")) > 1:
        words = loss_used.split("_")
        words = [f.title() for f in words]
        loss_used = "".join(words)
    loss_used_fn = ALL_LOSSES_DICT.get(loss_used, None)
    if loss_used_fn is None:
        from keras_core.src.losses import ALL_OBJECTS_DICT

        loss_used_fn = ALL_OBJECTS_DICT.get(loss_used, None)
    if loss_used_fn is None:
        print("loss not found")
        print(loss_name)
        print(loss_used)
        print("------------")
        return
    if "mirror" in loss_name:
        if "weights" in loss_name:
            loss_used_fn = MirrorWeights(
                loss_to_use=loss_used_fn(), weight_on_surplus=weight_on_surplus
            )
        if "mirror_loss" in loss_name:
            loss_used_fn = MirrorLoss(
                loss_to_use=loss_used_fn(), weight_on_surplus=weight_on_surplus
            )
        if "mirror_percentage" in loss_name:
            loss_used_fn = MirrorPercentage(
                loss_to_use=loss_used_fn(), weight_on_surplus=weight_on_surplus
            )
        if "mirror_normalized" in loss_name:
            loss_used_fn = MirrorNormalized(
                loss_to_use=loss_used_fn(), weight_on_surplus=weight_on_surplus
            )
    else:
        loss_used_fn = loss_used_fn()

    return loss_used_fn

import json

import keras


def flatten_extend(lst):
   flat_list = []
   for i in lst:
       if isinstance(i, list):
           flat_list.extend(flatten_extend(i))
       else:
           flat_list.append(i)
   return flat_list


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
    from alquitable.layers import Time2Vec
    mod= keras.models.model_from_json(load_json_dict(path), custom_objects={"Time2Vec": Time2Vec})
    # mod.summary()
    # import numpy as np
    # mod.predict(np.ones((1,*mod.input_shape[1:])))
    return mod


def open_model(path, custom_objects=None, compile_arg=True, safe_mode=True):
    if path.endswith("json"):
        return read_model_conf(path)
    elif path.endswith("keras"):
        return keras.models.load_model(
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
    loss_used = loss_name2.replace("mirror_weights_oratio7_", "")
    loss_used = loss_used.replace("mirror_weights_oratio_e3", "")
    loss_used = loss_used.replace("mirror_weights_oratios_s3", "")
    loss_used = loss_used.replace("mirror_weights_oratio_u15", "")
    loss_used = loss_used.replace("mirror_weights_oratio_h22", "")


    loss_used = loss_used.replace("mirror_weights_", "")
    loss_used = loss_used.replace("mirror_loss_norm_min_max", "")
    
    loss_used = loss_used.replace("mirror_loss_norm_r73", "")
    loss_used = loss_used.replace("mirror_loss_norm_s37", "")

    loss_used = loss_used.replace("mirror_loss_norm", "")
    loss_used = loss_used.replace("mirror_loss_", "")
    loss_used = loss_used.replace("mirror_percentage_", "")
    loss_used = loss_used.replace("mirror_normalized_", "")

    from alquitable.advanced_losses import (
        MirrorLoss,
        MirrorLossNorm,
        MirrorLossNormMinMax,
        MirrorNormalized,
        MirrorPercentage,
        MirrorWeights,
    )
    from alquitable.losses import ALL_LOSSES_DICT

    if "mirror" in loss_name:
        weight_on_surplus = True
        weigth_args = {}
        if "reversed" in loss_name:
            weigth_args["weight_on_surplus"] = False
        if "ratio7" in loss_name:
            weigth_args["ratio"] = 0.7
            weigth_args["ratio_on_weigths"] = True
        if "ratio_e3" in loss_name:
            weigth_args["ratio"] = 0.3
            weigth_args["ratio_on_weigths"] = True
        if "ratio_h22" in loss_name:
            weigth_args["ratio"] = 0.225
            weigth_args["ratio_on_weigths"] = True
        if "ratio_u15" in loss_name:
            weigth_args["ratio"] = 0.15
            weigth_args["ratio_on_weigths"] = True
        if "ratios_s3" in loss_name:
            weigth_args["ratio"] = -0.3
            weigth_args["ratio_on_weigths"] = True

        words = loss_used.split("_")
        words = [w.title() for w in words]
        loss_used = "".join(words)
    if len(loss_used.split("_")) > 1:
        words = loss_used.split("_")
        words = [f.title() for f in words]
        loss_used = "".join(words)
    loss_used_fn = ALL_LOSSES_DICT.get(loss_used, None)
    if loss_used_fn is None:
        from keras.src.losses import ALL_OBJECTS_DICT

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
                loss_to_use=loss_used_fn(), **weigth_args
            )
        if "mirror_loss" in loss_name:
            if "min_max" in loss_name:
                extra_args={}
                if "73" in loss_name:
                    extra_args["ratio_m"]=0.7
                    extra_args["ratio_s"]=0.3
                if "37" in loss_name:
                    extra_args["ratio_m"]=0.3
                    extra_args["ratio_s"]=0.7

                loss_used_fn = MirrorLossNormMinMax(
                    loss_to_use=loss_used_fn(), **extra_args
                )
            elif "norm" in loss_name:
                extra_args={}
                if "73" in loss_name:
                    extra_args["ratio_m"]=0.7
                    extra_args["ratio_s"]=0.3
                if "37" in loss_name:
                    extra_args["ratio_m"]=0.3
                    extra_args["ratio_s"]=0.7

                loss_used_fn = MirrorLossNorm(
                    loss_to_use=loss_used_fn(), **extra_args
                )
            else:
                loss_used_fn = MirrorLoss(
                    loss_to_use=loss_used_fn(), 
                )
        if "mirror_percentage" in loss_name:
            loss_used_fn = MirrorPercentage(
                loss_to_use=loss_used_fn(), 
            )
        if "mirror_normalized" in loss_name:
            loss_used_fn = MirrorNormalized(
                loss_to_use=loss_used_fn(), 
            )
    else:
        loss_used_fn = loss_used_fn()

    return loss_used_fn

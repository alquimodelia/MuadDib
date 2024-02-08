import ast
import importlib
import pathlib
from typing import Callable

import alquitable
import pandas as pd
from keras.callbacks import Callback
from keras.losses import Loss
from tinydb_serialization import SerializationMiddleware, Serializer


class PosixPathSerializer(Serializer):
    OBJ_CLASS = pathlib.Path

    def encode(self, obj):
        return str(obj)

    def decode(self, s):
        return pathlib.Path(s)

class PandasSerializer(Serializer):
    OBJ_CLASS = pd.DataFrame

    def encode(self, obj):
        return str(obj.to_json())

    def decode(self, s):
        return pd.DataFrame(ast.literal_eval(s))

class KerasLossSerializer(Serializer):
    OBJ_CLASS = Loss

    def encode(self, obj):
        params = obj.get_config()
        for key in params.keys():
            val = params[key]
            if isinstance(val, Loss):
                val = KerasLossSerializer().encode(val)
                params[key] = ast.literal_eval(val)
        return str({"loss":obj.__class__.__name__.split(".")[-1], "params":params, "module":obj.__module__})

    def decode(self, s):
        if not isinstance(s, dict):
            s = ast.literal_eval(s)
        params =s["params"]
        if "loss_to_use" in params:
            val = params["loss_to_use"]
            val = KerasLossSerializer().decode(val)
            params["loss_to_use"]=val
            s["params"] = params


        module = importlib.import_module(s["module"])
        loss_func = getattr(module, s["loss"])
        return loss_func(**s["params"])


class KerasCallbackSerializer(Serializer):
    OBJ_CLASS = Callback

    def encode(self, obj):
        return str(obj).split(".")[-1].replace("'>", "")

    def decode(self, s):
        return getattr(alquitable.callbacks, s)

class FunctionSerializer(Serializer):
    OBJ_CLASS = Callable

    def encode(self, obj):
        return f"{obj.__module__}:{obj.__name__}"

    def decode(self, s):
        module_name, function_name = s.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
 
def create_class_serializer(cls):
    class class_serializer(Serializer):
        OBJ_CLASS = cls

        def encode(self, obj):
            return obj.conf_file

        def decode(self, s):
            return cls(conf_file=s)

    return class_serializer()

# class ShaiHuludSerializer(Serializer):
#     OBJ_CLASS = ShaiHulud

#     def encode(self, obj):
#         return f"{obj.__module__}:{obj.__name__}"

#     def decode(self, s):
#         module_name, function_name = s.split(":")
#         module = importlib.import_module(module_name)
#         return getattr(module, function_name)

serialization = SerializationMiddleware()
serialization.register_serializer(PosixPathSerializer(), "PosixPath")
serialization.register_serializer(KerasLossSerializer(), "KerasLoss")
serialization.register_serializer(KerasCallbackSerializer(), "KerasCallback")
serialization.register_serializer(FunctionSerializer(), "Function")
serialization.register_serializer(PandasSerializer(), "PandasDataFrame")

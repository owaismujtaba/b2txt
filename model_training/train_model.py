'''from omegaconf import OmegaConf'''
from rnn_trainer import BrainToTextDecoder_Trainer
import pathlib
import os
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)

def load(file_: Union[str, pathlib.Path, IO[Any]]) -> Union[DictConfig, ListConfig]:
        from ._utils import get_yaml_loader

        if isinstance(file_, (str, pathlib.Path)):
            with io.open(os.path.abspath(file_), "r", encoding="utf-8") as f:
                obj = yaml.load(f, Loader=get_yaml_loader())
        elif getattr(file_, "read", None):
            obj = yaml.load(file_, Loader=get_yaml_loader())
        else:
            raise TypeError("Unexpected file type")

        if obj is not None and not isinstance(obj, (list, dict, str)):
            raise IOError(  # pragma: no cover
                f"Invalid loaded object type: {type(obj).__name__}"
            )

        ret: Union[DictConfig, ListConfig]
        if obj is None:
            ret = OmegaConf.create()
        else:
            ret = OmegaConf.create(obj)
        return ret

args = load('model_training/rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()

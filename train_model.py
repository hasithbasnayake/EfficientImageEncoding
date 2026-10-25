# Define train_model which instantiates the model and trains it given the set hyperparameters
import json
import dataclasses
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import NamedTuple

class Kernel(NamedTuple):
    x: int
    y: int

class STDPParameters(NamedTuple):
    A_plus: float
    A_minus: float
    tau: float
    mu_plus: float
    mu_minus: float

class WTAParameters(NamedTuple):
    num: int
    performWTA: bool

@dataclass(config=ConfigDict(strict=True))
class Config:
    train_size: int
    test_size: int
    seed: int
    num_epochs: int
    eval_every: int
    LGN_preprocessing: bool
    preprocessing_threshold: float
    kernel: Kernel
    num_steps: int
    WTA: bool
    wta_parameters: WTAParameters
    stdp_parameters: STDPParameters

def train_model(config_json_path):

    hyperparameters = gen_config(config_json_path)

    return hyperparameters

def gen_config(config_json_path):

    with open(config_json_path, "r") as f:
        data = json.load(f)

    required_keys = set([field.name for field in dataclasses.fields(Config)])

    missing_keys = required_keys - data.keys()
    extra_keys = data.keys() - required_keys

    if missing_keys or extra_keys:
        raise ValueError(f"Bad config. \n Missing: {sorted(missing_keys) or ''} \n Unexpected: {sorted(extra_keys) or ''}.")

    return Config(**data)

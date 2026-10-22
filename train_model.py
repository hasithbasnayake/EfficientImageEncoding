# Define train_model which instantiates the model and trains it given the set hyperparameters
import json
import dataclasses
from dataclasses import dataclass
from xxlimited_35 import Null


@dataclass
class Config:
    train_size: int
    test_size: int
    seed: int
    num_epochs: int
    eval_every: int
    LGN_preprocessing: bool
    preprocessing_threshold: float
    kernel_size: list[int]
    num_steps: int
    WTA: bool
    WTA_parameters: list
    A_plus: float
    A_minus: float
    tau: float
    mu_plus: float
    mu_minus: float
    #
    # def __post_init__(self):
    #     output = [isinstance(getattr(self, field.name), field.type) for field in dataclasses.fields(self)]
    #     pass

def train_model(config_json_path):

    hyperparameters = gen_config(config_json_path)

    print(dataclasses.fields(hyperparameters))
    print([field.name for field in dataclasses.fields(hyperparameters)])

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

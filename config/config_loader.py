import configparser
import json


mandatory = {}
class DefaultSetter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __setitem__(self, key, value):
        if key not in self.dictionary:
            if value is mandatory:
                raise ValueError(f" Argument --> {key} was mandatory but is not there")
            else:
                self.dictionary[key] = value

    def __getitem__(self, key):
        if key not in self.dictionary:
            self.dictionary[key] = {}
        return DefaultSetter(self.dictionary[key])


def load_json_config(config_path):

    file = open(config_path) 
    config = json.load(file)

    default_setter = DefaultSetter(config)
    default_setter["name"] = mandatory
    default_setter["train"]["lr"] = mandatory
    default_setter["train"]["dataset"] = mandatory
    default_setter["train"]["num_steps"] = mandatory
    default_setter["train"]["batch_size"] = mandatory
    default_setter["train"]["image_size"] = mandatory
    default_setter["train"]["validation"] = mandatory
    default_setter["train"]["restore_ckpt"] = None
    #default_setter.__getitem__("train").__setitem__("iter", 12)
    default_setter["train"]["iters"] = 12
    default_setter["train"]["gamma"] = mandatory
    #default_setter.__setitem__("wdecay", 0.0005)
    #default_setter["wdecay"] = 0.0005
    default_setter["train"]["wdecay"] = mandatory
    
    default_setter["position_only"] = mandatory
    default_setter["position_and_content"] = mandatory
    default_setter["num_heads"] = mandatory

    default_setter["val_freq"] = 10000
    default_setter["print_freq"] = 100
    default_setter["mixed_precision"] = False
    default_setter["gpus"] = [0]
    default_setter["epsilon"] = 1e-8
    default_setter["add_noise"] = False
    default_setter["clip"] = 1.0
    default_setter["dropout"] = 0.0
    default_setter["small"] = False
    default_setter["current_phase"] = 0
    default_setter["current_steps"] = -1

    # parameters related to training schedule are not mandatory. Also no default values are needed.

    return config


def cpy_args_to_config(args):
    config = {}
    config["name"] = args.name
    config["epsilon"] = args.epsilon
    config["clip"] = args.clip
    config["dropout"] = args.dropout
    config["small"] = args.small
    config["gpus"] = args.gpus
    config["add_noise"] = args.add_noise
    config["mixed_precision"] = args.mixed_precision
    config["val_freq"] = args.val_freq
    config["print_freq"] = args.print_freq
    config["position_only"] = args.position_only
    config["position_and_content"] = args.position_and_content
    config["num_heads"] = args.num_heads

    config["train"] = {}
    config["train"]["gamma"] = [args.gamma]
    config["train"]["wdecay"] = [args.wdecay]
    config["train"]["validation"] = [args.validation]
    config["train"]["num_steps"] = [args.num_steps]
    config["train"]["lr"] = [args.lr]
    config["train"]["image_size"] = [args.image_size]
    config["train"]["batch_size"] = [args.batch_size]
    config["train"]["dataset"] = [args.dataset]
    config["train"]["restore_ckpt"] = args.restore_ckpt
    config["train"]["iters"] = args.iters

    config["current_phase"] = args.current_phase
    config["current_steps"] = args.current_steps

    return config

def cpy_eval_args_to_config(args):
    config = {}
    config["model"] = args.model
    config["small"] = args.small
    config["dataset"] = args.dataset
    config["mixed_precision"] = args.mixed_precision
    config["alternate_corr"] = args.alternate_corr
    
    return config

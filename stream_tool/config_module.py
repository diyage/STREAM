import shutil
import os
import numpy as np


class MixConfig:
    def __init__(self, params: dict = {}):
        """config for global"""

        """this is used for model"""
        self.num_hidden = int(params.get("num_hidden", 64))
        """this is used for training"""
        self.use_checkpointing = bool(params.get("use_checkpointing", False))

        self.global_seed = int(params.get("global_seed", 1234))
        self.data_seed = int(params.get("data_seed", 1234))
        self.batch_size = int(params.get("batch_size", 32))
        self.max_epoch = int(params.get("max_epoch", 200))
        self.lr = float(params.get("lr", 0.001))
        self.weight_decay = float(params.get("weight_decay", 0.05))
        self.now_rank_total_batch_num = params.get("now_rank_total_batch_num", None)
        self.gradient_clip = float(params.get("gradient_clip", 32.0))

        self.reload_checkpoint_from = params.get("reload_checkpoint_from", "")
        """this is used for reload checkpoint"""

        """this is used for logging"""
        self.log_frequency: int = params.get("log_frequency", 2)

        """this is used for eval"""
        self.just_evaluation = int(params.get("just_evaluation", 0))
        self.evaluation_max_time_steps = params.get("evaluation_max_time_steps", None)
        if self.evaluation_max_time_steps is not None:
            self.evaluation_max_time_steps = int(self.evaluation_max_time_steps)

        """this is used for data"""
        self.root_offset_ind = int(params.get("root_offset_ind", -1))
        self.max_direct_upstream_distance = params.get("max_direct_upstream_distance", None)
        self.time_stamps_frequency = params.get("time_stamps_frequency", "6h")

        self.data_mean_frequency = int(params.get("data_mean_frequency", 1))

        self.history_time_step_num = int(params.get("history_time_step_num", 24))
        self.future_time_step_num = int(params.get("future_time_step_num", 7))
        self.normalize = params.get("normalize", True)

        self.num_workers = int(params.get("num_workers", 2))
        """this is used for other"""

        self.experiment_name = str(params.get("experiment_name", ""))
        assert self.experiment_name != ""
        save_dir = str(params.get("save_dir", ""))
        assert save_dir != ""
        self.save_dir = os.path.join(save_dir, self.experiment_name)

    def update(self, params: dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"\n>>> Skipping: {key} does not exist in the class", flush=True)

    def get_print_params(self) -> str:
        result = ""
        for k1, v1 in vars(self).items():
            result += f"{k1}: {v1}\n"
        return result

    def get_all_var_names(self):
        return list(vars(self).keys())

    def check_params(self):
        for k1, v1 in vars(self).items():
            if isinstance(v1, np.ndarray):
                continue
            if v1 in [None, ""]:
                print(f"\n>>> warn param {k1} is: {v1}", flush=True)


def build_default_args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--experiment_yaml_file', type=str)
    parser.add_argument('--just_evaluation', type=int, default=0)
    parser.add_argument('--reload_checkpoint_from', type=str, default='')
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--time_stamps_frequency', type=str, default='6h')
    parser.add_argument('--global_seed', type=int, default=1234)
    parser.add_argument('--data_seed', type=int, default=1234)
    return parser


def build_config_from_args(args) -> MixConfig:
    import yaml
    with open(args.experiment_yaml_file, mode='r') as f:
        config_dict = yaml.safe_load(f)

    for key, value in vars(args).items():
        if key in ["local_rank", "experiment_yaml_file"]:
            pass
        else:
            config_dict.update({key: value})

    return MixConfig(config_dict)

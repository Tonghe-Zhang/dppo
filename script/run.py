"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.
"""
import os
import sys
import pretty_errors
import logging

import math
import hydra
from omegaconf import OmegaConf
import gdown
from download_url import (
    get_dataset_download_url,
    get_normalization_download_url,
    get_checkpoint_download_url,
)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

@hydra.main(
    version_base=None,
    config_path=os.path.join(
        os.getcwd(), "cfg"
    ),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # For pre-training: download dataset if needed
    log.info(f"is train_dataset specified in cfg? {'train_dataset_patt' in cfg}")
    if 'train_dataset_path' in cfg:
        log.info(f"cfg.train_dataset_path={cfg.train_dataset_path}")
        log.info(f"does it exists? {os.path.exists(cfg.train_dataset_path)}")
        if not os.path.exists(cfg.train_dataset_path):
            download_url = get_dataset_download_url(cfg)
            download_target = os.path.dirname(cfg.train_dataset_path)
            log.info(f"Train_dataset_path does not exist. Downloading dataset from {download_url} to {download_target}")
            gdown.download_folder(url=download_url, output=download_target)
        else:
            log.info(f"Loading train_dataset from {cfg.train_dataset_path}.")

    # For fine-tuning: download normalization if needed
    if "normalization_path" in cfg:
        if not os.path.exists(cfg.normalization_path):
            download_url = get_normalization_download_url(cfg)
            download_target = cfg.normalization_path
            dir_name = os.path.dirname(download_target)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            log.info(
                f"Normalization statistics does not exist.\
                Downloading from {download_url} to {download_target}"
            )
            gdown.download(url=download_url, output=download_target, fuzzy=True)
        else:
            log.info(f"Loading Normalization statistics from {cfg.normalization_path}.")

    # For fine-tuning: download checkpoint if needed
    if "base_policy_path" in cfg:
        if not os.path.exists(cfg.base_policy_path):
            log.info(f"cfg.base_policy_path={cfg.base_policy_path} does not exists. ")
            download_url = get_checkpoint_download_url(cfg)
            if download_url is None:
                raise ValueError(
                    f"Unknown checkpoint path for fine-tuning. \
                    Did you specify the correct path to the policy you trained?"
                )
            download_target = cfg.base_policy_path
            dir_name = os.path.dirname(download_target)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            log.info(f"No checkpoint for fine-tuning found. Downloading checkpoint from {download_url} to {download_target}")
            gdown.download(url=download_url, output=download_target, fuzzy=True)
        else:
            log.info(f"Loading checkpoint for fine-tuning from {cfg.base_policy_path}. ")

    # Deal with isaacgym needs to be imported before torch
    if "env" in cfg and "env_type" in cfg.env and cfg.env.env_type == "furniture":
        import furniture_bench

    # run agent
    class_ = hydra.utils.get_class(cfg._target_)
    agent = class_(cfg)
    agent.run()

if __name__ == "__main__":
    main()

import basic_tools.utils as utils
import basic_tools.logger as logger
import basic_tools.checkpoint as checkpoint

import sys
import os
from omegaconf import DictConfig, OmegaConf

def start(args):
    # checkpoint.init_checkpoint()

    # sys.stdout = logger.Logger("./log.log", mode="w")
    # sys.stderr = logger.Logger("./log.err", mode="w")

    cmd_line = " ".join(sys.argv)
    print(f"{cmd_line}")
    print(f"Working dir: {os.getcwd()}")

    print(OmegaConf.to_yaml(args))
    return OmegaConf.to_yaml(args)

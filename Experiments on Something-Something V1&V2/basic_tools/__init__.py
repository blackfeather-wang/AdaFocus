import basic_tools.utils as utils
import basic_tools.logger as logger
import basic_tools.checkpoint as checkpoint

import sys
import os

def start(args):
    cmd_line = " ".join(sys.argv)
    print(f"{cmd_line}")
    print(f"Working dir: {os.getcwd()}")
    utils.set_all_seeds(args.seed)

    print(args)
    return args

import time
import signal
import os
import sys
import torch
import socket 


'''
Usage:

    init_checkpoint()

    if exist_checkpoint():
        any_object = load_checkpoint()

    save_checkpoint(any_object)
'''

CHECKPOINT_filename = 'checkpoint.pth.tar'
CHECKPOINT_tempfile = 'checkpoint.temp'
SIGNAL_RECEIVED = False

def SIGTERMHandler(a, b):
    print('received sigterm')
    pass


def signalHandler(a, b):
    global SIGNAL_RECEIVED
    print('Signal received', a, time.time(), flush=True)
    SIGNAL_RECEIVED = True

    print("caught signal", a)
    print(socket.gethostname(), "USR1 signal caught.")
    # do other stuff to cleanup here
    print('requeuing job ' + os.environ['SLURM_JOB_ID'])
    os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    sys.exit(-1)


def init_checkpoint():
    signal.signal(signal.SIGUSR1, signalHandler)
    signal.signal(signal.SIGTERM, SIGTERMHandler)
    print('Signal handler installed', flush=True)

def save_checkpoint(state):
    global CHECKPOINT_filename, CHECKPOINT_tempfile
    torch.save(state, CHECKPOINT_tempfile)
    if os.path.isfile(CHECKPOINT_tempfile):
        os.rename(CHECKPOINT_tempfile, CHECKPOINT_filename)
    print("Checkpoint done")

def save_checkpoint_if_signal(state):
    global SIGNAL_RECEIVED
    if SIGNAL_RECEIVED:
        save_checkpoint(state)

def exist_checkpoint():
    global CHECKPOINT_filename
    return os.path.isfile(CHECKPOINT_filename)

def load_checkpoint(filename=None):
    global CHECKPOINT_filename
    if filename is None:
        filename = CHECKPOINT_filename

    # optionally resume from a checkpoint
    # if args.resume:
        #if os.path.isfile(args.resume):
    # To make the script simple to understand, we do resume whenever there is
    # a checkpoint file
    if os.path.isfile(filename):
        print(f"=> loading checkpoint {filename}")
        checkpoint = torch.load(filename)
        print(f"=> loaded checkpoint {filename}")
        return checkpoint
    else:
        raise RuntimeError(f"=> no checkpoint found at '{filename}'")


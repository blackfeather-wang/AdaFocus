import torch
import sys
import random
import numpy as np
import os
import subprocess

from torch import optim

def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def to_cpu(x):
    if isinstance(x, dict):
        return { k : to_cpu(v) for k, v in x.items() }
    elif isinstance(x, list):
        return [ to_cpu(v) for v in x ]
    elif isinstance(x, torch.Tensor):
        return x.cpu()
    else:
        return x

def model2numpy(model):
    return { k : v.cpu().numpy() for k, v in model.state_dict().items() }

def activation2numpy(output):
    if isinstance(output, dict):
        return { k : activation2numpy(v) for k, v in output.items() }
    elif isinstance(output, list):
        return [ activation2numpy(v) for v in output ]
    elif isinstance(output, Variable):
        return output.data.cpu().numpy()

def count_size(x):
    if isinstance(x, dict):
        return sum([ count_size(v) for k, v in x.items() ])
    elif isinstance(x, list) or isinstance(x, tuple):
        return sum([ count_size(v) for v in x ])
    elif isinstance(x, torch.Tensor):
        return x.nelement() * x.element_size()
    else:
        return sys.getsizeof(x)

def mem2str(num_bytes):
    assert num_bytes >= 0
    if num_bytes >= 2 ** 30:  # GB
        val = float(num_bytes) / (2 ** 30)
        result = "%.3f GB" % val
    elif num_bytes >= 2 ** 20:  # MB
        val = float(num_bytes) / (2 ** 20)
        result = "%.3f MB" % val
    elif num_bytes >= 2 ** 10:  # KB
        val = float(num_bytes) / (2 ** 10)
        result = "%.3f KB" % val
    else:
        result = "%d bytes" % num_bytes
    return result

def get_mem_usage():
    import psutil

    mem = psutil.virtual_memory()
    result = ""
    result += "available: %s\t" % (mem2str(mem.available))
    result += "used: %s\t" % (mem2str(mem.used))
    result += "free: %s\t" % (mem2str(mem.free))
    # result += "active: %s\t" % (mem2str(mem.active))
    # result += "inactive: %s\t" % (mem2str(mem.inactive))
    # result += "buffers: %s\t" % (mem2str(mem.buffers))
    # result += "cached: %s\t" % (mem2str(mem.cached))
    # result += "shared: %s\t" % (mem2str(mem.shared))
    # result += "slab: %s\t" % (mem2str(mem.slab))
    return result

def get_github_string():
    _, output = subprocess.getstatusoutput("git -C ./ log --pretty=format:'%H' -n 1")
    ret, _ = subprocess.getstatusoutput("git -C ./ diff-index --quiet HEAD --")
    return f"Githash: {output}, unstaged: {ret}"


def accumulate(all_y, y):
    if all_y is None:
        all_y = dict()
        for k, v in y.items():
            if isinstance(v, list):
                all_y[k] = [ [vv] for vv in v ]
            else:
                all_y[k] = [v]
    else:
        for k, v in all_y.items():
            if isinstance(y[k], list):
                for vv, yy in zip(v, y[k]):
                    vv.append(yy)
            else:
                v.append(y[k])

    return all_y

def combine(all_y):
    output = dict()
    for k, v in all_y.items():
        if isinstance(v[0], list):
            output[k] = [ torch.cat(vv) for vv in v ]
        else:
            output[k] = torch.cat(v)

    return output

def concatOutput(loader, nets, condition=None):
    outputs = [None] * len(nets)

    use_cnn = nets[0].use_cnn

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if not use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()

            outputs = [ accumulate(output, to_cpu(net(x))) for net, output in zip(nets, outputs) ]
            if condition is not None and not condition(i):
               break

    return [ combine(output) for output in outputs ]


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lrs = args.lr_steps.split('-')
    lr_steps = [int(lr) for lr in lrs]
    if args.lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        backbone_lr = args.backbone_lr * decay
        fc_lr = args.fc_lr * decay
        decay = args.weight_decay
    elif args.lr_type == 'cos':
        import math
        backbone_lr = 0.5 * args.backbone_lr * (1 + math.cos(math.pi * epoch / args.epochs))
        fc_lr = 0.5 * args.fc_lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError

    if args.train_stage == 0:
        optimizer.param_groups[0]['lr'] = backbone_lr # Glancer
        optimizer.param_groups[1]['lr'] = backbone_lr # Focuser
        optimizer.param_groups[2]['lr'] = fc_lr # GRU
    elif args.train_stage == 1:
        optimizer.param_groups[0]['lr'] = backbone_lr # Focuser
        optimizer.param_groups[1]['lr'] = fc_lr # GRU
    elif args.train_stage == 2:
        pass
    elif args.train_stage == 3:
        optimizer.param_groups[0]['lr'] = fc_lr # GRU

    for param_group in optimizer.param_groups:
        # param_group['lr'] = lr
        param_group['weight_decay'] = decay

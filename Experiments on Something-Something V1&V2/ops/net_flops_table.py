import sys

sys.path.insert(0, "../")

import torch
import torchvision
from torch import nn
from thop import profile

feat_dim_dict = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "mobilenet_v2": 1280,
    "efficientnet-b0": 1280,
    "efficientnet-b1": 1280,
    "efficientnet-b2": 1408,
    "efficientnet-b3": 1536,
    "efficientnet-b4": 1792,
    "efficientnet-b5": 2048,
}

prior_dict = {
    "efficientnet-b0": (0.39, 5.3),
    "efficientnet-b1": (0.70, 7.8),
    "efficientnet-b2": (1.00, 9.2),
    "efficientnet-b3": (1.80, 12),
    "efficientnet-b4": (4.20, 19),
    "efficientnet-b5": (9.90, 30),
}


def get_gflops_params(model_name, resolution, num_classes, seg_len=-1, pretrained=True):
    if model_name in prior_dict:
        gflops, params = prior_dict[model_name]
        gflops = gflops / 224 / 224 * resolution * resolution
        return gflops, params

    if "resnet" in model_name:
        model = getattr(torchvision.models, model_name)(pretrained)
        last_layer = "fc"
    elif model_name == "mobilenet_v2":
        model = getattr(torchvision.models, model_name)(pretrained)
        last_layer = "classifier"
    else:
        exit("I don't know what is %s" % model_name)
    feat_dim = feat_dim_dict[model_name]

    setattr(model, last_layer, nn.Linear(feat_dim, num_classes))

    if seg_len == -1:
        dummy_data = torch.randn(1, 3, resolution, resolution)
    else:
        dummy_data = torch.randn(1, 3, seg_len, resolution, resolution)

    hooks = {}
    flops, params = profile(model, inputs=(dummy_data,), custom_ops=hooks)
    gflops = flops / 1e9
    params = params / 1e6

    return gflops, params


if __name__ == "__main__":
    str_list = []
    for s in str_list:
        print(s)

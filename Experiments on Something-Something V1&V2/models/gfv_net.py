import torch
from torch import nn
import torchvision
from PIL.Image import Image
from .ppo import PPO, Memory
from .ppo_continuous import PPO_Continuous
from .utils import random_crop, get_patch
from ops.transforms import *
from ops.basic_ops import ConsensusModule
from ops.temporal_shift import TemporalShift

from .tsn import TSN
from .mobilenetv2 import mobilenet_v2, InvertedResidual


class GFV(nn.Module):
    """
    top class for adaptive inference on video
    """

    def __init__(self, args):
        super(GFV, self).__init__()
        self.num_segments_glancer = args.num_segments_glancer
        self.num_segments_focuser = args.num_segments_focuser
        self.num_class = args.num_classes
        self.glancer = None
        self.focuser = None
        self.classifier = None
        self.input_size = 224
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.with_glancer = args.with_glancer
        self.glancer = Glancer(args)
        state_dim = args.feature_map_channels * (args.num_segments_glancer // args.video_div) \
                    * math.ceil(args.glance_size / 32) * math.ceil(args.glance_size / 32)
        policy_params = {
            'feature_dim': args.feature_map_channels * (args.num_segments_glancer // args.video_div),
            'state_dim': state_dim,
            'action_dim': args.action_dim,
            'hidden_state_dim': args.hidden_state_dim,
            'policy_conv': args.policy_conv,
            'gpu': args.gpu,
            'ppo_continuous': args.ppo_continuous,
            'gamma': args.gamma,
            'policy_lr': args.policy_lr,
            'action_std': args.action_std,
            'with_bn': args.actorcritic_with_bn
        }
        focuser_base_model_params = {
            'num_segments': args.num_segments_focuser,
            'modality': args.modality,
            'base_model': args.base_model,
            'partial_bn': args.partial_bn,
            'pretrain': args.pretrain,
            'is_shift': args.is_shift,
            'shift_div': args.shift_div,
            'shift_place': args.shift_place,
            'fc_lr5': args.fc_lr5,
            'temporal_pool': args.temporal_pool,
            'non_local': args.non_local
        }
        self.focuser = Focuser(args.patch_size, args.random_patch, policy_params, focuser_base_model_params)
        self.dropout = nn.Dropout(p=args.dropout)
        feat_dim = self.focuser.feature_dim
        self.classifier = nn.Linear(in_features=feat_dim, out_features=args.num_classes)
        self.consensus = ConsensusModule(consensus_type='avg')
        self.down = torchvision.transforms.Resize((args.patch_size, args.patch_size), interpolation=Image.BILINEAR)

    def train(self, mode=True):
        super(GFV, self).train(mode)
        return

    def forward(self, *argv, **kwargs):
        focuser_input = kwargs["input"]
        glancer_input = kwargs["scan"]  # imgs that scaled down to 96x96
        _b, _tc, _h, _w = glancer_input.shape
        _t, _c = _tc // 3, 3
        glancer_input = glancer_input.view(_b * _t, _c, _h, _w)

        with torch.no_grad():
            global_feat_map, global_feat_logit = self.glancer(
                glancer_input)  # global_feat_map is feature map before avgpool, size: b*t, c, h, w;

        _b, _tc, _h, _w = focuser_input.shape
        _t, _c = _tc // 3, 3
        local_patch = self.focuser(input=focuser_input, state=global_feat_map, restart_batch=True,
                                   training=kwargs["training"]).view(_b * _t, _c, self.patch_size, self.patch_size)
        local_feat = self.focuser.net(local_patch, no_reshape=True)

        local_feat = self.dropout(local_feat)
        local_feat_logit = self.classifier(local_feat)

        local_feat_logit = local_feat_logit.view(_b, self.num_segments_focuser, -1)
        local_feat_logit = self.consensus(local_feat_logit).squeeze(1)
        global_feat_logit = global_feat_logit.view(_b, self.num_segments_glancer, -1)
        global_feat_logit = self.consensus(global_feat_logit).squeeze(1)
        return local_feat_logit + global_feat_logit

    def glance(self, input_prime):
        _b, _tc, _h, _w = input_prime.shape
        _t, _c = _tc // 3, 3
        downs_2d = input_prime.view(_b * _t, _c, _h, _w)
        global_feat_map, global_feat = self.glancer(downs_2d)
        _, _featc, _feath, _featw = global_feat_map.shape
        return global_feat_map.view(_b, _t, _featc, _feath, _featw), global_feat.view(_b, _t, -1)

    def adjust_patch_size(self, patch_size):
        self.focuser.patch_size = patch_size
        self.focuser.patch_sampler.size = patch_size
        print("patch size adjusted to ", patch_size)

    def reset(self):
        self.classifier._reset(self.batch_size)

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    @property
    def crop_size(self):
        return self.input_size

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])

    def get_patch_augmentation(self):
        return torchvision.transforms.Compose([GroupScale(self.patch_size), GroupCenterCrop(self.patch_size)])

    def action_stage2(self, focuser_image, global_feat_map, global_feat_logit, focus_time_step, args,
                      prev_local_patch=None, training=True):
        num_frame_in_each_div_glancer = args.num_segments_glancer // args.video_div
        num_frame_in_each_div_focuser = args.num_segments_focuser // args.video_div
        _b, _t, _c, _h, _w = focuser_image.shape
        cur_image = focuser_image[:, focus_time_step * num_frame_in_each_div_focuser:
                                     (focus_time_step + 1) * num_frame_in_each_div_focuser, :, :, :].view(_b, -1, _h,
                                                                                                          _w)
        _b, _t, _c, _h, _w = global_feat_map.shape
        cur_global_feat_map = global_feat_map[:, focus_time_step * num_frame_in_each_div_glancer:
                                                 (focus_time_step + 1) * num_frame_in_each_div_glancer, :, :, :].view(
            _b, -1, _h, _w)
        cur_local_patch = self.focuser(input=cur_image, state=cur_global_feat_map, training=training,
                                       restart_batch=True if focus_time_step == 0 else False).view(_b,
                                                                                                   num_frame_in_each_div_focuser,
                                                                                                   3, args.patch_size,
                                                                                                   args.patch_size)
        cur_random_local_patch = self.focuser.random_patching(cur_image).view(_b, num_frame_in_each_div_focuser, 3,
                                                                              args.patch_size, args.patch_size)
        if prev_local_patch is not None:
            local_patch = torch.cat([prev_local_patch, cur_local_patch], dim=1)
            baseline_patch = torch.cat([prev_local_patch, cur_random_local_patch], dim=1)
        else:
            local_patch = cur_local_patch
            baseline_patch = cur_random_local_patch

        with torch.no_grad():
            total_local_patch = local_patch.view(-1, 3, args.patch_size, args.patch_size)
            total_local_feat = self.focuser.net(total_local_patch, no_reshape=True)
            total_local_feat = self.dropout(total_local_feat)
            total_local_feat_logit = self.classifier(total_local_feat)
            total_local_feat_logit = total_local_feat_logit.view(_b,
                                                                 num_frame_in_each_div_focuser * (focus_time_step + 1),
                                                                 -1)
            if self.with_glancer:
                total_logit = self.consensus(global_feat_logit).squeeze(1) + self.consensus(
                    total_local_feat_logit).squeeze(1)
            else:
                total_logit = self.consensus(total_local_feat_logit).squeeze(1)

        with torch.no_grad():
            baseline_patch = baseline_patch.view(-1, 3, args.patch_size, args.patch_size)
            baseline_feat = self.focuser.net(baseline_patch, no_reshape=True)
            baseline_feat = self.dropout(baseline_feat)
            baseline_logit = self.classifier(baseline_feat)
            baseline_logit = baseline_logit.view(_b, num_frame_in_each_div_focuser * (focus_time_step + 1), -1)
            if self.with_glancer:
                baseline_logit = self.consensus(global_feat_logit).squeeze(1) + self.consensus(baseline_logit).squeeze(
                    1)
            else:
                baseline_logit = self.consensus(baseline_logit).squeeze(1)

        return total_logit, baseline_logit, local_patch

    def action_stage3(self, focuser_image, global_feat_map, global_feat_logit, focus_time_step, args,
                      prev_local_patch=None):
        num_frame_in_each_div_glancer = args.num_segments_glancer // args.video_div
        num_frame_in_each_div_focuser = args.num_segments_focuser // args.video_div
        _b, _t, _c, _h, _w = focuser_image.shape
        cur_image = focuser_image[:, focus_time_step * num_frame_in_each_div_focuser:
                                     (focus_time_step + 1) * num_frame_in_each_div_focuser, :, :, :].view(_b, -1, _h,
                                                                                                          _w)
        _b, _t, _c, _h, _w = global_feat_map.shape
        cur_global_feat_map = global_feat_map[:, focus_time_step * num_frame_in_each_div_glancer:
                                                 (focus_time_step + 1) * num_frame_in_each_div_glancer, :, :, :].view(
            _b, -1, _h, _w)
        cur_local_patch = self.focuser(input=cur_image, state=cur_global_feat_map, training=False,
                                       restart_batch=True if focus_time_step == 0 else False).view(_b,
                                                                                                   num_frame_in_each_div_focuser,
                                                                                                   3,
                                                                                                   args.patch_size,
                                                                                                   args.patch_size)
        if prev_local_patch is not None:
            local_patch = torch.cat([prev_local_patch, cur_local_patch], dim=1)
        else:
            local_patch = cur_local_patch

        total_local_patch = local_patch.view(-1, 3, args.patch_size, args.patch_size)
        total_local_feat = self.focuser.net(total_local_patch, no_reshape=True)
        total_local_feat = self.dropout(total_local_feat)
        total_local_feat_logit = self.classifier(total_local_feat)
        total_local_feat_logit = total_local_feat_logit.view(
            _b, num_frame_in_each_div_focuser * (focus_time_step + 1), -1)
        if self.with_glancer:
            total_logit = self.consensus(global_feat_logit).squeeze(1) + self.consensus(
                total_local_feat_logit).squeeze(1)
        else:
            total_logit = self.consensus(total_local_feat_logit).squeeze(1)

        return total_logit, local_patch


class Glancer(nn.Module):
    """
    Global network for glancing
    """

    def __init__(self, args, skip=False):
        super(Glancer, self).__init__()
        self.net = mobilenet_v2(n_class=args.num_classes, pretrained=False)

        # Plug tsm module into mobilenetv2
        for m in self.net.modules():
            if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                m.conv[0] = TemporalShift(m.conv[0], n_segment=args.num_segments_glancer, n_div=args.shift_div)
                print('Adding temporal shift in Glancer mobilenetv2...')

        self.skip = skip

    def forward(self, input):
        return self.net.get_featmap(input)

    def predict(self, input):
        return self.net(input)

    @property
    def feature_dim(self):
        return self.net.feature_dim


class Focuser(nn.Module):
    """
    Local network for focusing
    """

    def __init__(self, size=96, random=False, policy_params: dict = None, focuser_base_model_params: dict = None):
        super(Focuser, self).__init__()
        self.net = TSN(num_segments=focuser_base_model_params['num_segments'],
                       modality=focuser_base_model_params['modality'],
                       base_model=focuser_base_model_params['base_model'],
                       partial_bn=focuser_base_model_params['partial_bn'],
                       pretrain=focuser_base_model_params['pretrain'],
                       is_shift=focuser_base_model_params['is_shift'],
                       shift_div=focuser_base_model_params['shift_div'],
                       shift_place=focuser_base_model_params['shift_place'],
                       fc_lr5=focuser_base_model_params['fc_lr5'],
                       temporal_pool=focuser_base_model_params['temporal_pool'],
                       non_local=focuser_base_model_params['non_local'])

        self.patch_size = size
        self.random = random
        self.patch_sampler = PatchSampler(self.patch_size, self.random)
        self.policy = None
        self.memory = Memory()
        if not self.random:
            assert policy_params is not None
            # FIXME: check!
            # self.patch_sizes = torch.Tensor(patch_sizes).cuda()
            self.patch_sizes = torch.Tensor([self.patch_size, 0]).cuda()
            self.standard_actions_set = {
                16: torch.Tensor([
                    [0, 0], [0, 1 / 3], [0, 2 / 3], [0, 1],
                    [1 / 3, 0], [1 / 3, 1 / 3], [1 / 3, 2 / 3], [1 / 3, 1],
                    [2 / 3, 0], [2 / 3, 1 / 3], [2 / 3, 2 / 3], [2 / 3, 1],
                    [3 / 3, 0], [3 / 3, 1 / 3], [3 / 3, 2 / 3], [3 / 3, 1],
                ]).cuda(),
                25: torch.Tensor([
                    [0, 0], [0, 1 / 4], [0, 2 / 4], [0, 3 / 4], [0, 1],
                    [1 / 4, 0], [1 / 4, 1 / 4], [1 / 4, 2 / 4], [1 / 4, 3 / 4], [1 / 4, 1],
                    [2 / 4, 0], [2 / 4, 1 / 4], [2 / 4, 2 / 4], [2 / 4, 3 / 4], [2 / 4, 1],
                    [3 / 4, 0], [3 / 4, 1 / 4], [3 / 4, 2 / 4], [3 / 4, 3 / 4], [3 / 4, 1],
                    [4 / 4, 0], [4 / 4, 1 / 4], [4 / 4, 2 / 4], [4 / 4, 3 / 4], [4 / 4, 1],
                ]).cuda(),
                36: torch.Tensor([
                    [0, 0], [0, 1 / 5], [0, 2 / 5], [0, 3 / 5], [0, 4 / 5], [0, 5 / 5],
                    [1 / 5, 0], [1 / 5, 1 / 5], [1 / 5, 2 / 5], [1 / 5, 3 / 5], [1 / 5, 4 / 5], [1 / 5, 5 / 5],
                    [2 / 5, 0], [2 / 5, 1 / 5], [2 / 5, 2 / 5], [2 / 5, 3 / 5], [2 / 5, 4 / 5], [2 / 5, 5 / 5],
                    [3 / 5, 0], [3 / 5, 1 / 5], [3 / 5, 2 / 5], [3 / 5, 3 / 5], [3 / 5, 4 / 5], [3 / 5, 5 / 5],
                    [4 / 5, 0], [4 / 5, 1 / 5], [4 / 5, 2 / 5], [4 / 5, 3 / 5], [4 / 5, 4 / 5], [4 / 5, 5 / 5],
                    [5 / 5, 0], [5 / 5, 1 / 5], [5 / 5, 2 / 5], [5 / 5, 3 / 5], [5 / 5, 4 / 5], [5 / 5, 5 / 5],
                ]).cuda(),
                49: torch.Tensor([
                    [0, 0], [0, 1 / 6], [0, 2 / 6], [0, 3 / 6], [0, 4 / 6], [0, 5 / 6], [0, 1],
                    [1 / 6, 0], [1 / 6, 1 / 6], [1 / 6, 2 / 6], [1 / 6, 3 / 6], [1 / 6, 4 / 6], [1 / 6, 5 / 6],
                    [1 / 6, 1],
                    [2 / 6, 0], [2 / 6, 1 / 6], [2 / 6, 2 / 6], [2 / 6, 3 / 6], [2 / 6, 4 / 6], [2 / 6, 5 / 6],
                    [2 / 6, 1],
                    [3 / 6, 0], [3 / 6, 1 / 6], [3 / 6, 2 / 6], [3 / 6, 3 / 6], [3 / 6, 4 / 6], [3 / 6, 5 / 6],
                    [3 / 6, 1],
                    [4 / 6, 0], [4 / 6, 1 / 6], [4 / 6, 2 / 6], [4 / 6, 3 / 6], [4 / 6, 4 / 6], [4 / 6, 5 / 6],
                    [4 / 6, 1],
                    [5 / 6, 0], [5 / 6, 1 / 6], [5 / 6, 2 / 6], [5 / 6, 3 / 6], [5 / 6, 4 / 6], [5 / 6, 5 / 6],
                    [5 / 6, 1],
                    [6 / 6, 0], [6 / 6, 1 / 6], [6 / 6, 2 / 6], [6 / 6, 3 / 6], [6 / 6, 4 / 6], [6 / 6, 5 / 6],
                    [6 / 6, 1],
                ]).cuda(),
                64: torch.Tensor([
                    [0, 0], [0, 1 / 7], [0, 2 / 7], [0, 3 / 7], [0, 4 / 7], [0, 5 / 7], [0, 6 / 7], [0, 7 / 7],
                    [1 / 7, 0], [1 / 7, 1 / 7], [1 / 7, 2 / 7], [1 / 7, 3 / 7], [1 / 7, 4 / 7], [1 / 7, 5 / 7],
                    [1 / 7, 6 / 7], [1 / 7, 7 / 7],
                    [2 / 7, 0], [2 / 7, 1 / 7], [2 / 7, 2 / 7], [2 / 7, 3 / 7], [2 / 7, 4 / 7], [2 / 7, 5 / 7],
                    [2 / 7, 6 / 7], [2 / 7, 7 / 7],
                    [3 / 7, 0], [3 / 7, 1 / 7], [3 / 7, 2 / 7], [3 / 7, 3 / 7], [3 / 7, 4 / 7], [3 / 7, 5 / 7],
                    [3 / 7, 6 / 7], [3 / 7, 7 / 7],
                    [4 / 7, 0], [4 / 7, 1 / 7], [4 / 7, 2 / 7], [4 / 7, 3 / 7], [4 / 7, 4 / 7], [4 / 7, 5 / 7],
                    [4 / 7, 6 / 7], [4 / 7, 7 / 7],
                    [5 / 7, 0], [5 / 7, 1 / 7], [5 / 7, 2 / 7], [5 / 7, 3 / 7], [5 / 7, 4 / 7], [5 / 7, 5 / 7],
                    [5 / 7, 6 / 7], [5 / 7, 7 / 7],
                    [6 / 7, 0], [6 / 7, 1 / 7], [6 / 7, 2 / 7], [6 / 7, 3 / 7], [6 / 7, 4 / 7], [6 / 7, 5 / 7],
                    [6 / 7, 6 / 7], [6 / 7, 7 / 7],
                    [7 / 7, 0], [7 / 7, 1 / 7], [7 / 7, 2 / 7], [7 / 7, 3 / 7], [7 / 7, 4 / 7], [7 / 7, 5 / 7],
                    [7 / 7, 6 / 7], [7 / 7, 7 / 7],
                ]).cuda(),
                81: torch.Tensor([
                    [0, 0], [0, 1 / 8], [0, 2 / 8], [0, 3 / 8], [0, 4 / 8], [0, 5 / 8], [0, 6 / 8], [0, 7 / 8],
                    [0, 8 / 8],
                    [1 / 8, 0], [1 / 8, 1 / 8], [1 / 8, 2 / 8], [1 / 8, 3 / 8], [1 / 8, 4 / 8], [1 / 8, 5 / 8],
                    [1 / 8, 6 / 8], [1 / 8, 7 / 8], [1 / 8, 8 / 8],
                    [2 / 8, 0], [2 / 8, 1 / 8], [2 / 8, 2 / 8], [2 / 8, 3 / 8], [2 / 8, 4 / 8], [2 / 8, 5 / 8],
                    [2 / 8, 6 / 8], [2 / 8, 7 / 8], [2 / 8, 8 / 8],
                    [3 / 8, 0], [3 / 8, 1 / 8], [3 / 8, 2 / 8], [3 / 8, 3 / 8], [3 / 8, 4 / 8], [3 / 8, 5 / 8],
                    [3 / 8, 6 / 8], [3 / 8, 7 / 8], [3 / 8, 8 / 8],
                    [4 / 8, 0], [4 / 8, 1 / 8], [4 / 8, 2 / 8], [4 / 8, 3 / 8], [4 / 8, 4 / 8], [4 / 8, 5 / 8],
                    [4 / 8, 6 / 8], [4 / 8, 7 / 8], [4 / 8, 8 / 8],
                    [5 / 8, 0], [5 / 8, 1 / 8], [5 / 8, 2 / 8], [5 / 8, 3 / 8], [5 / 8, 4 / 8], [5 / 8, 5 / 8],
                    [5 / 8, 6 / 8], [5 / 8, 7 / 8], [5 / 8, 8 / 8],
                    [6 / 8, 0], [6 / 8, 1 / 8], [6 / 8, 2 / 8], [6 / 8, 3 / 8], [6 / 8, 4 / 8], [6 / 8, 5 / 8],
                    [6 / 8, 6 / 8], [6 / 8, 7 / 8], [6 / 8, 8 / 8],
                    [7 / 8, 0], [7 / 8, 1 / 8], [7 / 8, 2 / 8], [7 / 8, 3 / 8], [7 / 8, 4 / 8], [7 / 8, 5 / 8],
                    [7 / 8, 6 / 8], [7 / 8, 7 / 8], [7 / 8, 8 / 8],
                    [8 / 8, 0], [8 / 8, 1 / 8], [8 / 8, 2 / 8], [8 / 8, 3 / 8], [8 / 8, 4 / 8], [8 / 8, 5 / 8],
                    [8 / 8, 6 / 8], [8 / 8, 7 / 8], [8 / 8, 8 / 8],
                ]).cuda(),
                100: torch.Tensor([
                    [0, 0], [0, 1 / 9], [0, 2 / 9], [0, 3 / 9], [0, 4 / 9], [0, 5 / 9], [0, 6 / 9], [0, 7 / 9],
                    [0, 8 / 9], [0, 9 / 9],
                    [1 / 9, 0], [1 / 9, 1 / 9], [1 / 9, 2 / 9], [1 / 9, 3 / 9], [1 / 9, 4 / 9], [1 / 9, 5 / 9],
                    [1 / 9, 6 / 9], [1 / 9, 7 / 9], [1 / 9, 8 / 9], [1 / 9, 9 / 9],
                    [2 / 9, 0], [2 / 9, 1 / 9], [2 / 9, 2 / 9], [2 / 9, 3 / 9], [2 / 9, 4 / 9], [2 / 9, 5 / 9],
                    [2 / 9, 6 / 9], [2 / 9, 7 / 9], [2 / 9, 8 / 9], [2 / 9, 9 / 9],
                    [3 / 9, 0], [3 / 9, 1 / 9], [3 / 9, 2 / 9], [3 / 9, 3 / 9], [3 / 9, 4 / 9], [3 / 9, 5 / 9],
                    [3 / 9, 6 / 9], [3 / 9, 7 / 9], [3 / 9, 8 / 9], [3 / 9, 9 / 9],
                    [4 / 9, 0], [4 / 9, 1 / 9], [4 / 9, 2 / 9], [4 / 9, 3 / 9], [4 / 9, 4 / 9], [4 / 9, 5 / 9],
                    [4 / 9, 6 / 9], [4 / 9, 7 / 9], [4 / 9, 8 / 9], [4 / 9, 9 / 9],
                    [5 / 9, 0], [5 / 9, 1 / 9], [5 / 9, 2 / 9], [5 / 9, 3 / 9], [5 / 9, 4 / 9], [5 / 9, 5 / 9],
                    [5 / 9, 6 / 9], [5 / 9, 7 / 9], [5 / 9, 8 / 9], [5 / 9, 9 / 9],
                    [6 / 9, 0], [6 / 9, 1 / 9], [6 / 9, 2 / 9], [6 / 9, 3 / 9], [6 / 9, 4 / 9], [6 / 9, 5 / 9],
                    [6 / 9, 6 / 9], [6 / 9, 7 / 9], [6 / 9, 8 / 9], [6 / 9, 9 / 9],
                    [7 / 9, 0], [7 / 9, 1 / 9], [7 / 9, 2 / 9], [7 / 9, 3 / 9], [7 / 9, 4 / 9], [7 / 9, 5 / 9],
                    [7 / 9, 6 / 9], [7 / 9, 7 / 9], [7 / 9, 8 / 9], [7 / 9, 9 / 9],
                    [8 / 9, 0], [8 / 9, 1 / 9], [8 / 9, 2 / 9], [8 / 9, 3 / 9], [8 / 9, 4 / 9], [8 / 9, 5 / 9],
                    [8 / 9, 6 / 9], [8 / 9, 7 / 9], [8 / 9, 8 / 9], [8 / 9, 9 / 9],
                    [9 / 9, 0], [9 / 9, 1 / 9], [9 / 9, 2 / 9], [9 / 9, 3 / 9], [9 / 9, 4 / 9], [9 / 9, 5 / 9],
                    [9 / 9, 6 / 9], [9 / 9, 7 / 9], [9 / 9, 8 / 9], [9 / 9, 9 / 9],
                ]).cuda()
            }
            self.policy_feature_dim = policy_params['feature_dim']
            self.policy_state_dim = policy_params['state_dim']
            self.policy_action_dim = policy_params['action_dim']
            self.policy_hidden_state_dim = policy_params['hidden_state_dim']
            self.policy_conv = policy_params['policy_conv']
            self.gpu = policy_params['gpu']  # for ddp
            self.ppo_continuous = policy_params['ppo_continuous']
            if self.ppo_continuous:
                print('Use PPO continuous')
                print('With bn = {}, action std = {}'.format(policy_params['with_bn'], policy_params['action_std']))
                self.policy = PPO_Continuous(self.policy_feature_dim, self.policy_state_dim,
                                             self.policy_hidden_state_dim, self.policy_conv, self.gpu,
                                             gamma=policy_params['gamma'], lr=policy_params['policy_lr'],
                                             action_std=policy_params['action_std'], with_bn=policy_params['with_bn'])
            else:
                print('Use PPO discrete')
                self.policy = PPO(self.policy_feature_dim, self.policy_state_dim, self.policy_action_dim,
                                  self.policy_hidden_state_dim, self.policy_conv, self.gpu,
                                  gamma=policy_params['gamma'], lr=policy_params['policy_lr'])

    def forward(self, *argv, **kwargs):
        if self.random:
            standard_action = None
        else:
            action = self.policy.select_action(kwargs['state'], self.memory, kwargs['restart_batch'],
                                               kwargs['training'])
            if self.ppo_continuous:
                standard_action = action
            else:
                standard_action, patch_size_list = self._get_standard_action(action)

        imgs = kwargs['input']
        _b = imgs.shape[0]
        if self.random:
            # torch.manual_seed(0)
            rand_index = torch.rand(imgs.size(0), 2)
            patch = get_patch(imgs, action_sequence=rand_index, patch_size=self.patch_size)
            return patch
        else:
            patch = get_patch(imgs, action_sequence=standard_action, patch_size=self.patch_size)
            return patch

    def random_patching(self, imgs):
        rand_index = torch.rand(imgs.size(0), 2)
        patch = get_patch(imgs, action_sequence=rand_index, patch_size=self.patch_size)
        return patch

    def predict(self, input):
        return self.net(input)

    def update(self):
        self.policy.update(self.memory)
        self.memory.clear_memory()

    def _get_standard_action(self, action):
        standard_action = self.standard_actions_set[self.policy_action_dim]
        return standard_action[action], None

    @property
    def feature_dim(self):
        return self.net.feature_dim


class PatchSampler(nn.Module):
    """
    Sample patch over the whole image
    """

    def __init__(self, size=96, random=True) -> None:
        super(PatchSampler, self).__init__()
        self.random = random
        self.size = size

    def sample(self, imgs, action=None):
        if self.random:
            # crop at random pos
            batch = []
            print(self.size)
            for img in imgs:
                batch.append(random_crop(img, self.size))
            return torch.stack(batch)
        else:
            # crop at the pos yielded by a policy network
            # FIXME: modify to variable patch size, i.e. action=25*len(patch_sizes)
            assert action != None
            return get_patch(imgs, action, self.size)

    def random_sample(self, imgs):
        # crop at random pos
        batch = []
        for img in imgs:
            batch.append(random_crop(img, self.size))
        return torch.stack(batch)

    def forward(self, *argv, **kwargs):
        raise NotImplementedError("Policy driven patch sampler not implemented.")

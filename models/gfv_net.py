
from PIL.Image import Image
from torch import autograd, nn
from ops.transforms import *
import torch.nn.functional as F
from torch.distributions import Categorical
from .resnet import resnet50
from .mobilenet import mobilenet_v2
from .utils import random_crop, get_patch
from .ppo import PPO, Memory
import torchvision

class GFV(nn.Module):
    """
    top class for adaptive inference on video
    """
    def __init__(self, args):
        super(GFV, self).__init__()
        self.num_segments = args.num_segments
        self.num_class = args.num_classes
        self.rew = args.reward
        if args.dataset == 'fcvid':
            assert args.num_classes == 239
        self.glancer = None
        self.focuser = None
        self.classifier = None
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.with_glancer = args.with_glancer
        self.glancer = Glancer(num_classes=self.num_class)
        state_dim = args.feature_map_channels * math.ceil(args.glance_size / 32) * math.ceil(args.glance_size / 32)
        policy_params = {
            'feature_dim': args.feature_map_channels,
            'state_dim': state_dim,
            'action_dim': args.action_dim,
            'hidden_state_dim': args.hidden_state_dim,
            'policy_conv': args.policy_conv,
            'gpu': args.gpu,
            'continuous': args.continuous,
            'gamma': args.gamma,
            'policy_lr': args.policy_lr
        }
        self.focuser = Focuser(args.patch_size, args.random_patch, policy_params, self.num_class)
        self.dropout = nn.Dropout(p=args.dropout)
        if self.with_glancer:
            feat_dim = self.glancer.feature_dim + self.focuser.feature_dim
        else:
            feat_dim = self.focuser.feature_dim
        if args.consensus == 'gru':
            print('Using GRU-based Classifier!')
            self.classifier = RecurrentClassifier(seq_len=args.num_segments,input_dim = feat_dim, batch_size = self.batch_size,hidden_dim=args.hidden_dim, num_classes=args.num_classes, dropout=args.dropout)
        elif args.consensus == 'fc':
            print('Using Linear Classifier!')
            self.classifier = LinearCLassifier(seq_len=args.num_segments, input_dim = feat_dim, batch_size= self.batch_size, hidden_dim=args.hidden_dim, num_classes=args.num_classes, dropout=args.dropout)
        self.down = torchvision.transforms.Resize((args.patch_size, args.patch_size),interpolation=Image.BILINEAR)
    
    def train(self, mode=True):
        super(GFV, self).train(mode)
        return

    def train_mode(self, args):
        if args.train_stage == 0:
            self.train()
        elif args.train_stage == 1:
            self.train()
            self.glancer.eval()
        elif args.train_stage == 2:
            self.eval()
            self.glancer.eval()
            self.focuser.eval()
            self.focuser.policy.policy.train()
            self.focuser.policy.policy_old.train()
        elif args.train_stage == 3:
            self.train()
            self.glancer.eval()
            self.focuser.eval()
            self.focuser.policy.policy.eval()
            self.focuser.policy.policy_old.eval()
        return

    def forward(self, *argv, **kwargs):
        if kwargs["backbone_pred"]:
            input = kwargs["input"]
            _b, _tc, _h, _w = input.shape  # input (B, T*C, H, W)
            _t, _c = _tc // 3, 3
            input_2d = input.view(_b * _t, _c, _h, _w)
            if kwargs['glancer']:
                pred = self.glancer.predict(input_2d).view(_b, _t, -1)
            else:
                pred = self.focuser.predict(input_2d).view(_b, _t, -1)
            return pred
        elif kwargs["one_step"]:
            gpu = kwargs["gpu"]
            input = kwargs["input"]
            down_sampled = kwargs["scan"]
            _b, _tc, _h, _w = input.shape
            _t, _c = _tc // 3, 3
            input_2d = input.view(_b, _t, _c, _h, _w)

            with torch.no_grad():
                global_feat_map, global_feat = self.glance(down_sampled)
            outputs = []
            preds = []
            features = []
            if not self.focuser.random:
                # for s3 training
                for focus_time_step in range(_t):
                    img = input_2d[:, focus_time_step, :, :, :]
                    cur_global_feat_map = global_feat_map[:, focus_time_step, :, :, :]
                    cur_global_feat = global_feat[:, focus_time_step, :]
                    if self.with_glancer:
                        with torch.no_grad():
                            if focus_time_step == 0:
                                local_feat, patch_size_list = self.focuser(input=img, state=cur_global_feat_map, restart_batch=True, training=kwargs["training"])
                            else:
                                local_feat, patch_size_list = self.focuser(input=img, state=cur_global_feat_map, restart_batch=False, training=kwargs["training"])
                            local_feat = local_feat.view(_b, -1)
                            feature = torch.cat([cur_global_feat, local_feat], dim=1)
                            features.append(feature)
                    else:
                        with torch.no_grad():
                            if focus_time_step == 0:
                                local_feat, patch_size_list = self.focuser(input=img, state=cur_global_feat_map, restart_batch=True, training=kwargs["training"])
                            else:
                                local_feat, patch_size_list = self.focuser(input=img, state=cur_global_feat_map, restart_batch=False, training=kwargs["training"])
                            local_feat = local_feat.view(_b, -1)
                            feature = local_feat
                            features.append(feature)
                features = torch.stack(features, dim=1)
                return self.classifier(features)
        else:
            # for s1 training
            input = kwargs["input"]
            down_sampled = kwargs["scan"]
            _b, _tc, _h, _w = input.shape  # input (B, T*C, H, W)
            _t, _c = _tc // 3, 3
            input_2d = input.view(_b * _t, _c, _h, _w)
            _b, _tc, _h, _w = down_sampled.shape  # input (B, T*C, H, W)
            _t, _c = _tc // 3, 3
            downs_2d = down_sampled.view(_b * _t, _c, _h, _w)
            with torch.no_grad():
                global_feat_map, global_feat = self.glancer(downs_2d)
            local_feat = self.focuser(input=input_2d, state=global_feat_map, restart_batch=True, training=kwargs["training"])[0].view(_b*_t, -1)

            feature = torch.cat([global_feat, local_feat], dim=1)
            feature = feature.view(_b, _t, -1)
            return self.classifier(feature)

    def glance(self, input_prime):
        _b, _tc, _h, _w = input_prime.shape  # input (B, T*C, H, W)
        _t, _c = _tc // 3, 3
        downs_2d = input_prime.view(_b * _t, _c, _h, _w)
        global_feat_map, global_feat = self.glancer(downs_2d)
        _, _featc, _feath, _featw = global_feat_map.shape
        return global_feat_map.view(_b, _t, _featc, _feath, _featw), global_feat.view(_b, _t, -1)

    def one_step_act(self, img, global_feat_map, global_feat, restart_batch=False, training=True):
        _b, _c, _h, _w = img.shape
        local_feat, pack = self.focuser(input=img, state=global_feat_map, restart_batch=restart_batch, training=training)
        if pack is not None:
            patch_size_list, action_list = pack
        else:
            patch_size_list, action_list = None, None
        
        if self.with_glancer:
            feature = torch.cat([global_feat, local_feat.view(_b, -1)], dim=1)
        else:
            feature = local_feat.view(_b, -1)
        feature = torch.unsqueeze(feature, 1) # (B, 1, feat)

        # for reward that contrast to random patching
        if self.rew == 'random':
            baseline_local_feature, pack = self.focuser.random_patching(img)
            if self.with_glancer:
                baseline_feature = torch.cat([global_feat, baseline_local_feature.view(_b, -1)], dim=1)
            else:
                baseline_feature = baseline_local_feature.view(_b, -1)
        elif self.rew == 'padding':
            # for reward that padding 0
            print('reward padding 0!')
            if self.with_glancer:
                baseline_feature = torch.cat([global_feat, torch.zeros(_b, self.focuser.feature_dim).cuda()], dim=1)
            else:
                baseline_feature = torch.zeros(_b, self.focuser.feature_dim).cuda()
        elif self.rew == 'prev':
            # bsl feat not used
            if self.with_glancer:
                baseline_feature = torch.cat([global_feat, torch.zeros(_b, self.focuser.feature_dim).cuda()], dim=1)
            else:
                baseline_feature = torch.zeros(_b, self.focuser.feature_dim).cuda()
        elif self.rew == 'conf':
            # bsl feat not used
            if self.with_glancer:
                baseline_feature = torch.cat([global_feat, torch.zeros(_b, self.focuser.feature_dim).cuda()], dim=1)
            else:
                baseline_feature = torch.zeros(_b, self.focuser.feature_dim).cuda()
        else:
            raise NotImplementedError

        baseline_feature = torch.unsqueeze(baseline_feature, 1)
        with torch.no_grad():
            baseline_logits, _ = self.classifier.test_single_forward(baseline_feature, reset=restart_batch)
            logits, last_out = self.classifier.single_forward(feature, reset=restart_batch)
        if training:
            return logits, last_out, patch_size_list, baseline_logits
        else:
            return logits, last_out, patch_size_list, action_list, baseline_logits

    @property
    def scale_size(self):
        return self.input_size * 256 // 224
    
    @property
    def crop_size(self):
        return self.input_size

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])

    def get_patch_augmentation(self):
        return torchvision.transforms.Compose([GroupScale(self.patch_size), GroupCenterCrop(self.patch_size)])


class Glancer(nn.Module):
    """
    Global network for glancing
    """
    def __init__(self, skip=False, num_classes=200):
        super(Glancer, self).__init__()
        self.net = mobilenet_v2(pretrained=True)
        num_ftrs = self.net.last_channel
        self.net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes),
        )
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
    def __init__(self, size=96, random=True, policy_params: dict = None, num_classes=200):
        super(Focuser, self).__init__()
        self.net = resnet50(pretrained=True)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, num_classes)

        self.patch_size = size
        self.random = random
        self.patch_sampler = PatchSampler(self.patch_size, self.random)
        self.policy = None
        self.memory = Memory()
        if not self.random:
            assert policy_params != None
            self.standard_actions_set = {
                25: torch.Tensor([
                    [0, 0], [0, 1/4], [0, 2/4], [0, 3/4], [0, 1],
                    [1/4, 0], [1/4, 1/4], [1/4, 2/4], [1/4, 3/4], [1/4, 1],
                    [2/4, 0], [2/4, 1/4], [2/4, 2/4], [2/4, 3/4], [2/4, 1],
                    [3/4, 0], [3/4, 1/4], [3/4, 2/4], [3/4, 3/4], [3/4, 1],
                    [4/4, 0], [4/4, 1/4], [4/4, 2/4], [4/4, 3/4], [4/4, 1],
                ]).cuda(),
                36: torch.Tensor([
                    [0, 0], [0, 1/5], [0, 2/5], [0, 3/5], [0, 4/5], [0, 5/5],
                    [1/5, 0], [1/5, 1/5], [1/5, 2/5], [1/5, 3/5], [1/5, 4/5], [1/5, 5/5],
                    [2/5, 0], [2/5, 1/5], [2/5, 2/5], [2/5, 3/5], [2/5, 4/5], [2/5, 5/5],
                    [3/5, 0], [3/5, 1/5], [3/5, 2/5], [3/5, 3/5], [3/5, 4/5], [3/5, 5/5],
                    [4/5, 0], [4/5, 1/5], [4/5, 2/5], [4/5, 3/5], [4/5, 4/5], [4/5, 5/5],
                    [5/5, 0], [5/5, 1/5], [5/5, 2/5], [5/5, 3/5], [5/5, 4/5], [5/5, 5/5],
                ]).cuda(),
                49: torch.Tensor([
                    [0, 0], [0, 1/6], [0, 2/6], [0, 3/6], [0, 4/6], [0, 5/6], [0, 1],
                    [1/6, 0], [1/6, 1/6], [1/6, 2/6], [1/6, 3/6], [1/6, 4/6], [1/6, 5/6], [1/6, 1],
                    [2/6, 0], [2/6, 1/6], [2/6, 2/6], [2/6, 3/6], [2/6, 4/6], [2/6, 5/6], [2/6, 1],
                    [3/6, 0], [3/6, 1/6], [3/6, 2/6], [3/6, 3/6], [3/6, 4/6], [3/6, 5/6], [3/6, 1],
                    [4/6, 0], [4/6, 1/6], [4/6, 2/6], [4/6, 3/6], [4/6, 4/6], [4/6, 5/6], [4/6, 1],
                    [5/6, 0], [5/6, 1/6], [5/6, 2/6], [5/6, 3/6], [5/6, 4/6], [5/6, 5/6], [5/6, 1],
                    [6/6, 0], [6/6, 1/6], [6/6, 2/6], [6/6, 3/6], [6/6, 4/6], [6/6, 5/6], [6/6, 1],
                ]).cuda(),
                64: torch.Tensor([
                    [0, 0], [0, 1/7], [0, 2/7], [0, 3/7], [0, 4/7], [0, 5/7], [0, 6/7], [0, 7/7],
                    [1/7, 0], [1/7, 1/7], [1/7, 2/7], [1/7, 3/7], [1/7, 4/7], [1/7, 5/7], [1/7, 6/7], [1/7, 7/7],
                    [2/7, 0], [2/7, 1/7], [2/7, 2/7], [2/7, 3/7], [2/7, 4/7], [2/7, 5/7], [2/7, 6/7], [2/7, 7/7],
                    [3/7, 0], [3/7, 1/7], [3/7, 2/7], [3/7, 3/7], [3/7, 4/7], [3/7, 5/7], [3/7, 6/7], [3/7, 7/7],
                    [4/7, 0], [4/7, 1/7], [4/7, 2/7], [4/7, 3/7], [4/7, 4/7], [4/7, 5/7], [4/7, 6/7], [4/7, 7/7],
                    [5/7, 0], [5/7, 1/7], [5/7, 2/7], [5/7, 3/7], [5/7, 4/7], [5/7, 5/7], [5/7, 6/7], [5/7, 7/7],
                    [6/7, 0], [6/7, 1/7], [6/7, 2/7], [6/7, 3/7], [6/7, 4/7], [6/7, 5/7], [6/7, 6/7], [6/7, 7/7],
                    [7/7, 0], [7/7, 1/7], [7/7, 2/7], [7/7, 3/7], [7/7, 4/7], [7/7, 5/7], [7/7, 6/7], [7/7, 7/7],
                ]).cuda()
            }
            self.policy_feature_dim = policy_params['feature_dim']
            self.policy_state_dim = policy_params['state_dim']
            self.policy_action_dim = policy_params['action_dim']
            self.policy_hidden_state_dim = policy_params['hidden_state_dim']
            self.policy_conv = policy_params['policy_conv']
            self.gpu = policy_params['gpu'] #for ddp
            self.policy = PPO(self.policy_feature_dim, self.policy_state_dim, self.policy_action_dim, self.policy_hidden_state_dim, self.policy_conv, self.gpu, gamma=policy_params['gamma'], lr=policy_params['policy_lr'])
    
    def forward(self, *argv, **kwargs):
        if self.random:
            standard_action = None
        else:
            action = self.policy.select_action(kwargs['state'], self.memory, kwargs['restart_batch'], kwargs['training'])
            standard_action, patch_size_list = self._get_standard_action(action)

        # print('action:', standard_action)
        imgs = kwargs['input']
        _b = imgs.shape[0]
        if self.random:
            patch = self.patch_sampler.sample(imgs, standard_action)
            return self.net.get_featmap(patch, pooled=True), None
        else:
            patch = self.patch_sampler.sample(imgs, standard_action)
            return self.net.get_featmap(patch, pooled=True), (None, standard_action)
    

    def random_patching(self, imgs):
        patch = self.patch_sampler.random_sample(imgs)
        return self.net.get_featmap(patch, pooled=True), None

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

    def sample(self, imgs, action = None):
        if self.random:
            # crop at random position
            batch = []
            print(self.size)
            for img in imgs:
                batch.append(random_crop(img, self.size))
            return torch.stack(batch)
        else:
            # crop at the position yielded by policy network
            assert action != None
            return get_patch(imgs, action, self.size)

    def random_sample(self, imgs):
        # crop at random position
        batch = []
        for img in imgs:
            batch.append(random_crop(img, self.size))
        return torch.stack(batch)

    def forward(self, *argv, **kwargs):
        raise NotImplementedError



class LinearCLassifier(nn.Module):
    def __init__(self, seq_len, input_dim, batch_size, hidden_dim, num_classes, dropout):
        super(LinearCLassifier, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.fc = nn.Linear(self.input_dim, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, feature):
        _b, _t, _f = feature.shape
        out = self.dropout(feature)
        logits = self.fc(out.reshape(_b*_t, -1))
        softmax = self.softmax(logits).reshape(_b, _t, -1)
        avg = softmax.mean(dim=1, keepdim=False)
        log_softmax = torch.log(avg)
        return log_softmax, avg

class RecurrentClassifier(nn.Module):
    """
    GRU based classifier
    """
    def __init__(self, seq_len, input_dim, batch_size, hidden_dim, num_classes, dropout, bias=True):
        super(RecurrentClassifier, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, bias=bias, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

        self.hx = None
        self.cx = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature):
        _b, _t, _f = feature.shape
        hx = torch.zeros(self.gru.num_layers, _b,  self.hidden_dim).cuda()
        self.gru.flatten_parameters()
        out, hn = self.gru(feature, hx) #out(_b, _t, hidden_size) with batch_first=true
        out = self.dropout(out)
        logits = self.fc(out.reshape(_b*_t, -1))
        last_out = logits.reshape(_b, _t, -1)[:, -1, :].reshape(_b, -1)
        return logits, last_out
    
    def single_forward(self, feature, reset=False, gpu=0):
        _b, _t, _f = feature.shape
        if reset:
            self.hx = torch.zeros(self.gru.num_layers, _b,  self.hidden_dim).cuda(gpu)
        self.gru.flatten_parameters()
        out, self.hx = self.gru(feature, self.hx) #out(_b, _t, hidden_size) with batch_first=true
        out = self.dropout(out)
        logits = self.fc(out.reshape(_b*_t, -1))
        last_out = logits.reshape(_b, _t, -1)[:, -1, :].reshape(_b, -1)
        return logits, last_out
    
    def test_single_forward(self, feature, reset=False, gpu=0):
        _b, _t, _f = feature.shape
        if reset:
            self.hx = torch.zeros(self.gru.num_layers, _b,  self.hidden_dim).cuda(gpu)
        self.gru.flatten_parameters()
        out, _ = self.gru(feature, self.hx) #out(_b, _t, hidden_size) with batch_first=true
        out = self.dropout(out)
        logits = self.fc(out.reshape(_b*_t, -1))
        last_out = logits.reshape(_b, _t, -1)[:, -1, :].reshape(_b, -1)
        return logits, last_out

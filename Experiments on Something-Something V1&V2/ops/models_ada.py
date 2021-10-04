from torch import nn
from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from ops.net_flops_table import feat_dim_dict

from torch.distributions import Categorical


def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell


class TSN_Ada(nn.Module):
    def __init__(self, num_class, num_segments,
                 base_model='resnet101', consensus_type='avg', before_softmax=True, dropout=0.8,
                 crop_num=1, partial_bn=True, pretrain='imagenet', fc_lr5=False, args=None):
        super(TSN_Ada, self).__init__()
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.pretrain = pretrain

        self.fc_lr5 = fc_lr5

        # TODO(yue)
        self.args = args
        self.rescale_to = args.rescale_to
        if self.args.ada_reso_skip:
            base_model = self.args.backbone_list[0] if len(self.args.backbone_list) >= 1 else None
        self.base_model_name = base_model
        self.num_class = num_class
        self.multi_models = False
        self.time_steps = self.num_segments

        if self.args.ada_reso_skip:
            self.reso_dim = self._get_resolution_dimension()
            self.skip_dim = len(self.args.skip_list)
            self.action_dim = self._get_action_dimension()
            self._prepare_policy_net()
            self._extends_to_multi_models()

        self._prepare_base_model(base_model)
        self._prepare_fc(num_class)

        self.consensus = ConsensusModule(consensus_type, args=self.args)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _extends_to_multi_models(self):
        if len(self.args.backbone_list) >= 1:
            self.multi_models = True
            self.base_model_list = nn.ModuleList()
            self.new_fc_list = nn.ModuleList()

    def _prep_a_net(self, model_name, shall_pretrain):
        if "efficientnet" in model_name:
            if shall_pretrain:
                model = EfficientNet.from_pretrained(model_name)
            else:
                model = EfficientNet.from_named(model_name)
            model.last_layer_name = "_fc"
        else:
            model = getattr(torchvision.models, model_name)(shall_pretrain)
            if "resnet" in model_name:
                model.last_layer_name = 'fc'
            elif "mobilenet_v2" in model_name:
                model.last_layer_name = 'classifier'
        return model

    def _get_resolution_dimension(self):
        reso_dim = 0
        for i in range(len(self.args.backbone_list)):
            reso_dim += self.args.ada_crop_list[i]
        if self.args.policy_also_backbone:
            reso_dim += 1
        return reso_dim

    def _get_action_dimension(self):
        action_dim = self.reso_dim + self.skip_dim
        return action_dim

    def _prepare_policy_net(self):
        shall_pretrain = not self.args.policy_from_scratch
        self.lite_backbone = self._prep_a_net(self.args.policy_backbone, shall_pretrain)
        self.policy_feat_dim = feat_dim_dict[self.args.policy_backbone]
        self.rnn = nn.LSTMCell(input_size=self.policy_feat_dim, hidden_size=self.args.hidden_dim, bias=True)

    def _prepare_base_model(self, base_model):
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        if self.args.ada_reso_skip:
            shall_pretrain = len(self.args.model_paths) == 0 or self.args.model_paths[0].lower() != 'none'
            for bbi, backbone_name in enumerate(self.args.backbone_list):
                model = self._prep_a_net(backbone_name, shall_pretrain)
                self.base_model_list.append(model)
        else:
            self.base_model = self._prep_a_net(base_model, self.pretrain == 'imagenet')

    def _prepare_fc(self, num_class):
        def make_a_linear(input_dim, output_dim):
            linear_model = nn.Linear(input_dim, output_dim)
            normal_(linear_model.weight, 0, 0.001)
            constant_(linear_model.bias, 0)
            return linear_model

        i_do_need_a_policy_network = True

        if self.args.ada_reso_skip and i_do_need_a_policy_network:
            setattr(self.lite_backbone, self.lite_backbone.last_layer_name, nn.Dropout(p=self.dropout))
            feed_dim = self.args.hidden_dim if not self.args.frame_independent else self.policy_feat_dim
            self.linear = make_a_linear(feed_dim, self.action_dim)
            self.lite_fc = make_a_linear(feed_dim, num_class)

        if self.multi_models:
            multi_fc_list = [None]
            for bbi, base_model in enumerate(self.base_model_list):
                for fc_i, exit_index in enumerate(multi_fc_list):
                    last_layer_name = base_model.last_layer_name
                    feature_dim = getattr(base_model, last_layer_name).in_features

                    new_fc = make_a_linear(feature_dim, num_class)
                    self.new_fc_list.append(new_fc)
                    setattr(base_model, last_layer_name, nn.Dropout(p=self.dropout))

        elif self.base_model_name is not None:
            if "mobilenet_v2" == self.base_model_name:
                feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[1].in_features
            else:
                feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features

            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = make_a_linear(feature_dim, num_class)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_Ada, self).train(mode)
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            if self.args.ada_reso_skip:
                models = [self.lite_backbone]
                if self.multi_models:
                    models = models + self.base_model_list
            else:
                models = [self.base_model]

            for the_model in models:
                count = 0
                bn_scale = 1
                for m in the_model.modules():
                    if isinstance(m, nn.BatchNorm2d):  # TODO(yue)
                        count += 1
                        if count >= (2 * bn_scale if self._enable_pbn else bn_scale):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.LSTMCell):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_weight.append(ps[1])
                normal_bias.append(ps[2])
                normal_bias.append(ps[3])

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def backbone(self, input_data, the_base_model, new_fc, signal=-1, indices_list=[], boost=False, b_t_c=False,
                 **kwargs):
        _b, _tc, _h, _w = input_data.shape  # TODO(yue) input (B, T*C, H, W)
        _t, _c = _tc // 3, 3

        if b_t_c:
            input_b_t_c = input_data.view(_b, _t, _c, _h, _w)
        else:
            input_2d = input_data.view(_b * _t, _c, _h, _w)

        if b_t_c:
            feat = the_base_model(input_b_t_c, signal=signal, **kwargs)
        else:
            feat = the_base_model(input_2d)

        _base_out = None
        if b_t_c:
            if new_fc is not None:
                _base_out = new_fc(feat.view(_b * _t, -1)).view(_b, _t, -1)
        else:
            if new_fc is not None:
                _base_out = new_fc(feat).view(_b, _t, -1)
            feat = feat.view(_b, _t, -1)
        return feat, _base_out

    def get_lite_j_and_r(self, input_list, online_policy, tau):

        feat_lite, _ = self.backbone(input_list[self.args.policy_input_offset], self.lite_backbone, None)

        r_list = []
        lite_j_list = []
        batch_size = feat_lite.shape[0]
        hx = init_hidden(batch_size, self.args.hidden_dim)
        cx = init_hidden(batch_size, self.args.hidden_dim)

        remain_skip_vector = torch.zeros(batch_size, 1)
        old_hx = None
        old_r_t = None

        if self.args.use_reinforce:
            log_prob_r_list = []
            prob_r_list = []

        for t in range(self.time_steps):
            if self.args.frame_independent:
                feat_t = feat_lite[:, t]
            else:
                hx, cx = self.rnn(feat_lite[:, t], (hx, cx))
                feat_t = hx
            if self.args.use_reinforce:
                p_t = F.softmax(self.linear(feat_t), dim=1).clamp(min=1e-8)
            else:
                p_t = torch.log(F.softmax(self.linear(feat_t), dim=1).clamp(min=1e-8))
            j_t = self.lite_fc(feat_t)
            lite_j_list.append(j_t)  # TODO as pred

            # TODO (yue) need a simple case to illustrate this
            if online_policy:
                if self.args.use_reinforce:
                    m = Categorical(p_t)

                    prob_r_list.append(p_t)

                    r_t_idx = m.sample()
                    r_t = torch.eye(self.action_dim)[r_t_idx].cuda()
                    log_prob_r_t = m.log_prob(r_t_idx)
                    log_prob_r_list.append(log_prob_r_t)
                else:
                    r_t = torch.cat(
                        [F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])])

                # TODO update states and r_t
                if old_hx is not None:
                    take_bool = remain_skip_vector > 0.5
                    take_old = torch.tensor(take_bool, dtype=torch.float).cuda()
                    take_curr = torch.tensor(~take_bool, dtype=torch.float).cuda()
                    hx = old_hx * take_old + hx * take_curr
                    r_t = old_r_t * take_old + r_t * take_curr

                # TODO update skipping_vector
                for batch_i in range(batch_size):
                    for skip_i in range(self.action_dim - self.reso_dim):
                        # TODO(yue) first condition to avoid valuing skip vector forever
                        if remain_skip_vector[batch_i][0] < 0.5 and r_t[batch_i][self.reso_dim + skip_i] > 0.5:
                            remain_skip_vector[batch_i][0] = self.args.skip_list[skip_i]
                old_hx = hx
                old_r_t = r_t
                r_list.append(r_t)  # TODO as decision
                remain_skip_vector = (remain_skip_vector - 1).clamp(0)
        if online_policy:
            if self.args.use_reinforce:
                return lite_j_list, torch.stack(r_list, dim=1), torch.stack(log_prob_r_list, dim=1)
            else:
                return lite_j_list, torch.stack(r_list, dim=1)
        else:
            return lite_j_list, None

    def using_online_policy(self):
        if any([self.args.offline_lstm_all, self.args.offline_lstm_last]):
            return False
        elif any([self.args.random_policy, self.args.all_policy]):
            return False
        elif self.args.real_scsampler:
            return False
        else:
            return True

    def input_fusion(self, input_data, r):
        # TODO data: B * TC * H * W
        # TODO r   : B * T * T
        _b, _tc, _h, _w = input_data.shape
        _c = _tc // self.args.num_segments
        fuse_data_list = []

        for bi in range(_b):
            if self.args.identity_prior:
                prior = torch.eye(self.args.num_segments).to(input_data.device)
            else:
                prior = 0
            if self.args.lower_mask:
                mask = torch.tril(torch.ones(self.args.num_segments, self.args.num_segments)).to(input_data.device)
            else:
                mask = 1
            real_r = (r[bi] + prior) * mask
            if self.args.direct_lower_mask:
                real_r = torch.tril(real_r)
            if self.args.row_normalization:
                real_r = real_r / (real_r.sum(dim=1, keepdim=True).clamp_min(1e-6))
            fused_data = torch.matmul(real_r, input_data[bi].view(self.args.num_segments, _c * _h * _w))
            fuse_data_list.append(fused_data)
        return torch.stack(fuse_data_list, dim=0).view(_b, _tc, _h, _w)

    def get_feat_and_pred(self, input_list, r_all, **kwargs):
        feat_out_list = []
        base_out_list = []
        ind_list = []

        for bb_i, the_backbone in enumerate(self.base_model_list):
            feat_out, base_out = self.backbone(input_list[bb_i], the_backbone, self.new_fc_list[bb_i])
            feat_out_list.append(feat_out)
            base_out_list.append(base_out)
        return feat_out_list, base_out_list, ind_list

    def late_fusion(self, base_out_list, in_matrix, out_matrix):
        return base_out_list

    def forward(self, *argv, **kwargs):
        if not self.args.ada_reso_skip:  # TODO simple TSN
            _, base_out = self.backbone(kwargs["input"][0], self.base_model, self.new_fc,
                                        signal=self.args.default_signal)
            output = self.consensus(base_out)
            return output.squeeze(1)

        input_list = kwargs["input"]
        batch_size = input_list[0].shape[0]  # TODO(yue) input[0] B*(TC)*H*W

        if self.args.use_reinforce:
            lite_j_list, r_all, r_log_prob = self.get_lite_j_and_r(input_list, self.using_online_policy(),
                                                                   kwargs["tau"])
        else:
            lite_j_list, r_all = self.get_lite_j_and_r(input_list, self.using_online_policy(), kwargs["tau"])

        if self.multi_models:
            if "tau" not in kwargs:
                kwargs["tau"] = None

            feat_out_list, base_out_list, ind_list = self.get_feat_and_pred(input_list, r_all, tau=kwargs["tau"])
        else:
            feat_out_list, base_out_list, ind_list = [], [], []

        if self.args.policy_also_backbone:
            base_out_list.append(torch.stack(lite_j_list, dim=1))

        if self.args.offline_lstm_last:  # TODO(yue) no policy - use policy net as backbone - just LSTM(last)
            return lite_j_list[-1].squeeze(1), None, None, None

        elif self.args.offline_lstm_all:  # TODO(yue) no policy - use policy net as backbone - just LSTM(average)
            return torch.stack(lite_j_list).mean(dim=0).squeeze(1), None, None, None

        elif self.args.real_scsampler:
            real_pred = base_out_list[0]
            lite_pred = torch.stack(lite_j_list, dim=1)
            output, ind = self.consensus(real_pred, lite_pred)
            return output.squeeze(1), ind, real_pred, lite_pred

        else:
            if self.args.random_policy:  # TODO(yue) random policy
                r_all = torch.zeros(batch_size, self.time_steps, self.action_dim).cuda()
                for i_bs in range(batch_size):
                    for i_t in range(self.time_steps):
                        r_all[i_bs, i_t, torch.randint(self.action_dim, [1])] = 1.0
            elif self.args.all_policy:  # TODO(yue) all policy: take all
                r_all = torch.ones(batch_size, self.time_steps, self.action_dim).cuda()
            output = self.combine_logits(r_all, base_out_list, ind_list)
            if self.args.save_meta and self.args.save_all_preds:
                return output.squeeze(1), r_all, torch.stack(base_out_list, dim=1)
            else:
                if self.args.use_reinforce:
                    return output.squeeze(1), r_all, r_log_prob, torch.stack(base_out_list, dim=1)
                else:
                    return output.squeeze(1), r_all, None, torch.stack(base_out_list, dim=1)

    def combine_logits(self, r, base_out_list, ind_list):
        # TODO r                N, T, K
        # TODO base_out_list  < K * (N, T, C)
        pred_tensor = torch.stack(base_out_list, dim=2)
        r_tensor = r[:, :, :self.reso_dim].unsqueeze(-1)
        t_tensor = torch.sum(r[:, :, :self.reso_dim], dim=[1, 2]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame
        return (pred_tensor * r_tensor).sum(dim=[1, 2]) / t_tensor

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])

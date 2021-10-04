import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F

from ops.dataset import TSNDataSet
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, ProgressMeter
from models.gfv_net import GFV
from basic_tools.checkpoint import save_checkpoint

import os
import time
import hydra
import shutil
import basic_tools
from collections import OrderedDict


def parse_gpus(gpus):
    if type(gpus) is int:
        return [gpus]
    gpu_list = gpus.split('-')
    return [int(g) for g in gpu_list]


@hydra.main(config_path="conf", config_name="stage2.yaml")
def main(args):
    assert args.train_stage == 2, "This code is only used for stage-2 training (only train ppo)!"
    config_yaml = basic_tools.start(args)
    with open('training.log', 'a+') as f_handler:
        f_handler.writelines(config_yaml)

    best_acc1 = 0
    num_class, args.train_list, args.val_list, args.root_path, prefix = \
        dataset_config.return_dataset(args.dataset, modality='RGB', root_dataset=args.data_dir)
    args.num_classes = num_class

    model = GFV(args).cuda()

    if args.pretrained_glancer:
        pretrained_ckpt = torch.load(os.path.expanduser(args.pretrained_glancer), map_location='cpu')

        new_state_dict = OrderedDict()
        for k, v in pretrained_ckpt['state_dict'].items():
            if k[:18] == 'module.base_model.':
                name = k[18:]  # remove `module.`
                new_state_dict[name] = v
            elif k[:14] == 'module.new_fc.':
                name = 'classifier.' + k[14:]  # replace `module.new_fc` with 'classifier'
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        model.glancer.net.load_state_dict(new_state_dict, strict=True)

        print('Load Pretrained Glancer from {}!'.format(args.pretrained_glancer))
        with open('training.log', 'a+') as f_handler:
            f_handler.writelines('Load Pretrained Glancer from {}!'.format(args.pretrained_glancer))

    if args.pretrained_focuser:
        pretrained_ckpt = torch.load(os.path.expanduser(args.pretrained_focuser), map_location='cpu')

        new_state_dict = OrderedDict()
        new_fc_state_ditc = OrderedDict()
        for k, v in pretrained_ckpt['state_dict'].items():
            print('Load ckpt param: {}'.format(k))
            if k[:7] == 'module.' and 'new_fc' not in k:
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            elif 'module.new_fc.' in k:
                name = k[14:]  # remove `module.`
                new_fc_state_ditc[name] = v
            else:
                new_state_dict[k] = v

        model.classifier.load_state_dict(new_fc_state_ditc, strict=True)
        model.focuser.net.load_state_dict(new_state_dict, strict=False)

        print('Load Pretrained Focuser from {}!'.format(args.pretrained_focuser))
        with open('training.log', 'a+') as f_handler:
            f_handler.writelines('Load Pretrained Focuser from {}!'.format(args.pretrained_focuser))

    model.focuser.net.base_model = torch.nn.Sequential(*list(model.focuser.net.base_model.children())[:-1])
    print(model)
    print(model.focuser.policy.policy)
    with open('training.log', 'a+') as f_handler:
        f_handler.writelines('model: {}'.format(model))
        f_handler.writelines('policy net: {}'.format(model.focuser.policy.policy))

    scale_size = model.scale_size
    crop_size = model.crop_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset
                                                              or 'jester' in args.dataset else True)

    # data loading code
    normalize = GroupNormalize(input_mean, input_std)
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list,
                   num_segments_glancer=args.num_segments_glancer,
                   num_segments_focuser=args.num_segments_focuser,
                   new_length=1,
                   modality='RGB',
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list,
                   num_segments_glancer=args.num_segments_glancer,
                   num_segments_focuser=args.num_segments_focuser,
                   new_length=1,
                   modality='RGB',
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.pretrained:
        pretrained_ckpt = torch.load(os.path.expanduser(args.pretrained))

        start_epoch = pretrained_ckpt['epoch']
        print('Load pretrained ckpt from: {}'.format(os.path.expanduser(args.pretrained)))
        print('Load pretrained ckpt from epoch: {}'.format(start_epoch))

        model.glancer.load_state_dict(pretrained_ckpt['glancer'], strict=True)
        model.focuser.load_state_dict(pretrained_ckpt['focuser'], strict=True)
        model.classifier.load_state_dict(pretrained_ckpt['fc'], strict=True)

        ckpt_acc1 = pretrained_ckpt['best_acc']
        print('best ckpt_acc1 for ckpt: {}'.format(ckpt_acc1))
        with open('training.log', 'a+') as f_handler:
            f_handler.writelines('Load pretrained ckpt from: {}'.format(os.path.expanduser(args.pretrained)))
            f_handler.writelines('Load pretrained ckpt from epoch: {}'.format(start_epoch))
            f_handler.writelines('best ckpt_acc1 for ckpt: {}'.format(ckpt_acc1))

    if args.resume:
        resume_ckpt = torch.load(os.path.expanduser(args.resume))

        start_epoch = resume_ckpt['epoch']
        print('resume from epoch: {}'.format(start_epoch))

        model.glancer.load_state_dict(resume_ckpt['glancer'], strict=True)
        model.focuser.load_state_dict(resume_ckpt['focuser'], strict=True)
        model.classifier.load_state_dict(resume_ckpt['fc'], strict=True)
        model.focuser.policy.policy.load_state_dict(resume_ckpt['policy'])
        model.focuser.policy.policy_old.load_state_dict(resume_ckpt['policy'])

        best_acc1 = resume_ckpt['best_acc']
        print('best acc1 for ckpt: {}'.format(best_acc1))
        with open('training.log', 'a+') as f_handler:
            f_handler.writelines('Resume from: {}'.format(os.path.expanduser(args.resume)))
            f_handler.writelines('Resume from epoch: {}'.format(start_epoch))
            f_handler.writelines('best_acc1 for resume: {}'.format(best_acc1))
    else:
        start_epoch = 0

    if args.evaluate:
        acc1, val_logs = validate(val_loader, model, criterion, args)
        with open('training.log', 'a+') as f_handler:
            f_handler.writelines(val_logs)
        print('Best Acc@1 = {}'.format(acc1))
        return

    for epoch in range(start_epoch, args.epochs + 1):
        acc1 = 0

        train_logs = train(train_loader, model, criterion, epoch, args)
        acc1, val_logs = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'glancer': model.glancer.state_dict(),
            'focuser': model.focuser.state_dict(),
            'fc': model.classifier.state_dict(),
            'policy': model.focuser.policy.policy.state_dict(),
            'acc': acc1,
            'best_acc': best_acc1})
        if is_best:
            shutil.copyfile('checkpoint.pth.tar', 'checkpoint.pth.tar'.replace('checkpoint', 'model_best'))
        with open('training.log', 'a+') as f_handler:
            f_handler.writelines(train_logs)
            if epoch < 40:
                if epoch % args.eval_freq == 0:
                    f_handler.writelines(val_logs)
            else:
                f_handler.writelines(val_logs)


def train(train_loader, model: GFV, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    reward_list = [AverageMeter('Rew', ':6.5f') for _ in range(args.video_div)]

    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5,
                             prefix="Epoch: [{}]".format(epoch))

    logs = []

    model.eval()
    model.focuser.policy.policy.train()
    model.focuser.policy.policy_old.train()

    end = time.time()
    all_targets = []
    for i, (glancer_images, focuser_images, target) in enumerate(train_loader):
        # data preparation
        _b = target.shape[0]
        all_targets.append(target)
        data_time.update(time.time() - end)
        glancer_images = glancer_images.cuda()  # images (B, T * C, H, W)
        focuser_images = focuser_images.cuda()  # images (B, T * C, H, W)
        target = target.cuda()
        glancer_images = torch.nn.functional.interpolate(glancer_images, (args.glance_size, args.glance_size))
        glancer_images = glancer_images.cuda()

        confidence_last = 0
        focuser_images = focuser_images.view(_b, args.num_segments_focuser, 3, model.input_size, model.input_size)

        # Glancer: output global feature
        with torch.no_grad():
            global_feat_map, global_feat_logit = model.glance(
                glancer_images)  # feat_map (B, T, C, H, W) feat_vec (B, T, _)

        local_patch_list = []
        for focus_time_step in range(args.video_div):
            pred, baseline_logit, local_patch = model.action_stage2(
                focuser_images, global_feat_map, global_feat_logit, focus_time_step, args,
                prev_local_patch=None if focus_time_step == 0 else local_patch_list[focus_time_step - 1], training=True)

            local_patch_list.append(local_patch)

            loss = criterion(pred, target)
            confidence = torch.gather(F.softmax(pred.detach(), 1), dim=1, index=target.view(-1, 1)).view(1, -1)

            bsl_confidence = torch.gather(F.softmax(baseline_logit.detach(), 1), dim=1,
                                          index=target.view(-1, 1)).view(1, -1)
            reward = confidence - bsl_confidence

            reward_list[focus_time_step].update(reward.data.mean().item(), glancer_images.size(0))
            model.focuser.memory.rewards.append(reward)

        model.focuser.update()

        # Update evaluation metrics
        acc1, acc5 = accuracy(pred, target, topk=(1, 5))
        losses.update(loss.item(), glancer_images.size(0))
        top1.update(acc1[0], glancer_images.size(0))
        top5.update(acc5[0], glancer_images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        _reward = [reward.avg for reward in reward_list]
        print('reward of each step: {}'.format(_reward))

        logs.append(progress.print(i))
        logs.append(' '.join(map(str, _reward)) + '\n')

    return logs


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    reward_list = [AverageMeter('Rew', ':6.5f') for _ in range(args.video_div)]
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix='Test: ')

    logs = []
    # switch to evaluate mode
    model.eval()
    model.focuser.policy.policy.eval()
    model.focuser.policy.policy_old.eval()

    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (glancer_images, focuser_images, target) in enumerate(val_loader):
            _b = target.shape[0]
            all_targets.append(target)
            glancer_images = glancer_images.cuda()
            focuser_images = focuser_images.cuda()
            target = target.cuda()
            glancer_images = torch.nn.functional.interpolate(glancer_images, (args.glance_size, args.glance_size))
            glancer_images = glancer_images.cuda()

            # compute output
            focuser_images = focuser_images.view(_b, args.num_segments_focuser, 3, model.input_size, model.input_size)
            # MDP Focusing
            with torch.no_grad():
                global_feat_map, global_feat_logit = model.glance(
                    glancer_images)  # feat_map (B, T, C, H, W) feat_vec (B, T, _)

            for focus_time_step in range(args.video_div):
                pred, baseline_logit, local_patch = model.action_stage2(
                    focuser_images, global_feat_map, global_feat_logit, focus_time_step, args,
                    prev_local_patch=None if focus_time_step == 0 else local_patch, training=False)

                loss = criterion(pred, target)
                confidence = torch.gather(F.softmax(pred.detach(), 1), dim=1, index=target.view(-1, 1)).view(1, -1)

                bsl_confidence = torch.gather(F.softmax(baseline_logit.detach(), 1), dim=1,
                                              index=target.view(-1, 1)).view(1, -1)
                reward = confidence - bsl_confidence

                reward_list[focus_time_step].update(reward.data.mean().item(), glancer_images.size(0))

            # Update evaluation metrics
            acc1, acc5 = accuracy(pred, target, topk=(1, 5))
            losses.update(loss.item(), glancer_images.size(0))
            top1.update(acc1[0], glancer_images.size(0))
            top5.update(acc5[0], glancer_images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            _reward = [reward.avg for reward in reward_list]
            print('reward of each step: {}'.format(_reward))

            logs.append(progress.print(i))
            logs.append(' '.join(map(str, _reward)) + '\n')

    return top1.avg, logs


def save_model(prefix, model, i):
    filename = os.path.join(os.getcwd(), f"{prefix}-{i}.pt")
    torch.save(model, filename)
    print(f"[{i}] Saving {prefix} to {filename}")


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="9"  # specify which GPU(s) to be used
    main()

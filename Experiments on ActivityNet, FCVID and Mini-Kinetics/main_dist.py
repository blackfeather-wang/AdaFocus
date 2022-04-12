
import os
from os.path import join as ospj
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torch.cuda.amp import autocast, GradScaler

import hydra
import basic_tools
from basic_tools.utils import *
from basic_tools.checkpoint import save_checkpoint

from ops.dataset import TSNDataSet
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map, Recorder, ProgressMeter
from models.gfv_net import GFV


best_acc1 = 0

# Use hydra to maintain configuration and parse arguments in an elegant way.
@hydra.main(config_path="conf", config_name="default")
def main(args):
    config_yaml = basic_tools.start(args)

    # Log file
    with open('training.log','a+') as f_handler:
        f_handler.writelines(config_yaml)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    print("curent thread rank:",args.rank)

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,args.data_dir)
    args.num_classes = num_class

    model =  GFV(args)
    start_epoch = 0
    if args.train_stage == 0:
    # when pretrain backbone on 224x224 images, we do not load pretrained weights
        pass
    else:
        if args.pretrained_glancer:
            resume_ckpt = torch.load(args.pretrained_glancer, map_location='cpu')
            model.glancer.load_state_dict(resume_ckpt['glancer'], strict=False)
            print('Pretrained Glancer with best acc {} Loaded!'.format(resume_ckpt['best_acc']))
        if args.pretrained_focuser:
            resume_ckpt = torch.load(args.pretrained_focuser, map_location='cpu')
            model.focuser.load_state_dict(resume_ckpt['focuser'], strict=False)
            print('Pretrained Focuser with best acc {} Loaded!'.format(resume_ckpt['best_acc']))
    if args.resume:
        # resume from checkpoint of previous training stage
        resume_ckpt = torch.load(args.resume, map_location='cpu')
        print('best preformance: ',resume_ckpt['best_acc'])
        model.glancer.load_state_dict(resume_ckpt['glancer'])
        model.focuser.load_state_dict(resume_ckpt['focuser'], strict=False)
        model.classifier.load_state_dict(resume_ckpt['fc'])

        if args.train_stage == 3:
            model.focuser.policy.policy.load_state_dict(resume_ckpt['policy'])
            model.focuser.policy.policy_old.load_state_dict(resume_ckpt['policy'])

    if args.train_stage == 2:
        # we do not use dp or ddp when training stage2, please modify this field in YAML file
        assert not args.distributed

    scale_size = model.scale_size
    crop_size = model.crop_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation(flip=True)
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            print("Using DDP with specific GPU!")
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            print("Using DDP with all GPUs!")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.train_stage != 2:
            model = torch.nn.DataParallel(model).cuda()
            print('Using DP with GPUs')
        else:
            model = model.cuda()

    # define loss function (criterion)
    if args.consensus == 'gru':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.consensus == 'fc':
        criterion = nn.NLLLoss().cuda(args.gpu)
    else:
        raise NotImplementedError()

    # specify different optimizer to different training stage
    if args.train_stage == 0:
        optimizer = torch.optim.SGD([
                                    {'params': model.module.glancer.parameters()},
                                    {'params': model.module.focuser.parameters()},
                                    {'params': model.module.classifier.parameters()}
                                    ], lr=0,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.train_stage == 1:
        optimizer = torch.optim.SGD([
                                    {'params': model.module.focuser.parameters()},
                                    {'params': model.module.classifier.parameters()}
                                    ], lr=0,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.train_stage == 2:
        # We use the optimizer inside the PPO instance
        optimizer = None
    elif args.train_stage == 3:
        optimizer = torch.optim.SGD([
                                    {'params': model.module.classifier.parameters()}
                                    ], lr=0,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("Cannot specify optimizer to such a training stage.")

    cudnn.benchmark = True

    # data loading
    normalize = GroupNormalize(input_mean, input_std)
    train_dataset =  TSNDataSet(root_path=args.root_path, list_file=args.train_list, num_segments=args.num_segments,image_tmpl=prefix, 
        transform=torchvision.transforms.Compose([
            train_augmentation, 
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize]),
        dense_sample=args.dense_sample, 
        dataset=args.dataset, 
        partial_fcvid_eval=args.partial_fcvid_eval, 
        partial_ratio=args.partial_ratio, 
        ada_reso_skip=args.ada_reso_skip, 
        reso_list=args.reso_list, 
        random_crop=args.center_crop, 
        center_crop=args.center_crop, 
        ada_crop_list=args.ada_crop_list, 
        rescale_to=args.rescale_to,
        policy_input_offset=args.policy_input_offset, 
        save_meta=args.save_meta)

    val_dataset =  TSNDataSet(root_path=args.root_path, list_file=args.val_list,num_segments=args.num_segments, image_tmpl=prefix,random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize]), 
        dense_sample=args.dense_sample,
        dataset=args.dataset,
        partial_fcvid_eval=args.partial_fcvid_eval,
        partial_ratio=args.partial_ratio,
        ada_reso_skip=args.ada_reso_skip,
        reso_list=args.reso_list,
        random_crop=args.random_crop,
        center_crop=args.center_crop,
        ada_crop_list=args.ada_crop_list,
        rescale_to=args.rescale_to,
        policy_input_offset=args.policy_input_offset,
        save_meta=args.save_meta)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


    # Start Training......
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    for epoch in range(start_epoch, args.epochs + 1):
        acc1 = 0
        # time.sleep(1)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        if optimizer:
            adjust_learning_rate(args, optimizer, epoch)
        else:
            # in stage 2, we do not train backbone and classifier
            assert args.train_stage == 2

        # train for one epoch
        train_logs = train(train_loader, model, criterion, optimizer, epoch, args, scaler)

        # evaluate the model on validation set
        if epoch % args.eval_freq == 0:
            acc1, val_logs = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # logging and checkpoint in the main thread
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if is_best:
                if isinstance(model, GFV):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(), 
                        'glancer': model.glancer.state_dict(),
                        'focuser': model.focuser.state_dict(),
                        'fc': model.classifier.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'policy': model.focuser.policy.policy.state_dict() if not args.random_patch else None,
                        'acc': acc1, 
                        'best_acc': best_acc1,
                        'optimizer': optimizer.state_dict() if optimizer else None})
                else:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': model.module.state_dict(), 
                        'glancer': model.module.glancer.state_dict(),
                        'focuser': model.module.focuser.state_dict(),
                        'fc': model.module.classifier.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'policy': model.module.focuser.policy.policy.state_dict() if not args.random_patch else None,
                        'acc': acc1, 
                        'best_acc': best_acc1,
                        'optimizer': optimizer.state_dict() if optimizer else None})
            with open('training.log','a+') as f_handler:
                f_handler.writelines(train_logs)
                if epoch % args.eval_freq == 0:
                    f_handler.writelines(val_logs)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    meanAP = AverageMeter('mAP', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix='Test: ')

    logs = []
    # switch to evaluate mode
    model.eval()
    if args.evaluate:
        set_all_seeds(args.seed)

    for eval_time in range(1):
        all_results = []
        all_targets = []
        all_local_results = []
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                _b = target.shape[0]
                all_targets.append(target)
                images = images.cuda()
                target = target[:,0].cuda()
                input_prime = torch.nn.functional.interpolate(images, (args.glance_size, args.glance_size))
                input_prime = input_prime.cuda()

                # compute output
                if args.train_stage == 1:
                    output, pred = model(input=images, scan=input_prime, training=False, backbone_pred=False, one_step=False)
                    if args.consensus == 'gru':
                        loss = criterion(output, target.view(_b, -1).expand(_b, args.num_segments).reshape(-1))
                    elif args.consensus == 'fc':
                        loss = criterion(output, target)
                    all_results.append(pred)
                elif args.train_stage == 2:
                    local_results = []
                    confidence_last = 0
                    images = images.view(_b, args.num_segments, 3, model.input_size, model.input_size)
                    # MDP Focusing
                    with torch.no_grad():
                        global_feat_map, global_feat = model.glance(input_prime) # feat_map (B, T, C, H, W) feat_vec (B, T, _)

                    for focus_time_step in range(args.num_segments):
                        img = images[:, focus_time_step, :, :]
                        cur_global_map = global_feat_map[:, focus_time_step, :, :, :]
                        cur_global_feat = global_feat[:, focus_time_step, :]
                        if focus_time_step == 0:
                            # here output equals to pred
                            output, pred, patch_size_list, _, baseline_logits = model.one_step_act(img, cur_global_map, cur_global_feat, restart_batch=True, training=False)
                        else:
                            output, pred, patch_size_list, _, baseline_logits = model.one_step_act(img, cur_global_map, cur_global_feat, restart_batch=False, training=False)
                        local_results.append(pred)
                        loss = criterion(output, target)
                        confidence = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target.view(-1, 1)).view(1, -1)
                        bsl_confidence = torch.gather(F.softmax(baseline_logits.detach(), 1), dim=1, index=target.view(-1, 1)).view(1, -1)
                        reward, confidence_last = get_reward(args, confidence, confidence_last, bsl_confidence)
                    all_results.append(pred)
                    all_local_results.append(torch.stack(local_results))
                elif args.train_stage == 3:
                    outputs, pred = model(input=images, scan=input_prime, training=False, backbone_pred=False, one_step=True, gpu=args.gpu)
                    loss = criterion(outputs, target.view(_b, -1).expand(_b, args.num_segments).reshape(-1))
                    all_results.append(pred)
                    all_local_results.append(outputs.reshape(_b, args.num_segments, -1))
                elif args.train_stage == 0:
                    pred = model(input=input_prime, scan=None, glancer=args.pretrain_glancer, backbone_pred=True, one_step=False)
                    pred = pred.mean(1, keepdim=True).squeeze(1)
                    loss = criterion(pred, target)
                    all_results.append(pred)
                else:
                    raise NotImplementedError

                # measure accuracy and record loss
                acc1, acc5 = accuracy(pred, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                logs.append(progress.print(i))

            if args.dataset == 'fcvid':
                mAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())
            else:
                mAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0)[:, 0:1].cpu())
            meanAP.update(mAP, 1)
            print('mAP: ', mAP)
        logs.append('mAP: {mAP:.5f}\n'.format(mAP=meanAP.avg))
        print(' * Acc@1 {top1.avg:.5f} Acc@5 {top5.avg:.5f} mAP {meanAP.avg:.5f}'.format(top1=top1, top5=top5, meanAP=meanAP))
        
        if args.train_stage == 3:
            for i in range(args.num_segments):
                if args.dataset == 'fcvid':
                    mAP, _ = cal_map(torch.cat(all_local_results, 0)[:, i, :].cpu(), torch.cat(all_targets, 0).cpu())  # TODO(yue) single-label mAP
                else:
                    mAP, _ = cal_map(torch.cat(all_local_results, 0)[:, i, :].cpu(), torch.cat(all_targets, 0)[:, 0:1].cpu())  # TODO(yue) single-label mAP
                logs.append('mAP @ time step {step}: {mAP:.5f}\n'.format(mAP=mAP, step=i))
                print('mAP @ time step {step}: {mAP:.5f}\n'.format(mAP=mAP, step=i))

        if args.train_stage == 2:
            for i in range(args.num_segments):
                if args.dataset == 'fcvid':
                    mAP, _ = cal_map(torch.cat(all_local_results, 1)[i, :, :].cpu(), torch.cat(all_targets, 0).cpu())
                else:
                    mAP, _ = cal_map(torch.cat(all_local_results, 1)[i, :, :].cpu(), torch.cat(all_targets, 0)[:, 0:1].cpu())
                logs.append('mAP @ time step {step}: {mAP:.5f}\n'.format(mAP=mAP, step=i))
                print('mAP @ time step {step}: {mAP:.5f}\n'.format(mAP=mAP, step=i))

    if args.dataset == 'minik':
        return top1.avg, logs
    else:
        return mAP, logs

def train(train_loader, model, criterion, optimizer, epoch, args, scaler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    if isinstance(model, GFV):
        model.train_mode(args)
    else:
        model.module.train_mode(args)

    if args.train_stage == 0:
        backbone_lr = 'BackBone LR: '+ str(optimizer.param_groups[0]['lr'])
        fc_lr = 'GRU LR: '+ str(optimizer.param_groups[2]['lr'])
    elif args.train_stage == 1:
        backbone_lr = 'BackBone LR: '+ str(optimizer.param_groups[0]['lr'])
        fc_lr = 'GRU LR: '+ str(optimizer.param_groups[1]['lr'])
    elif args.train_stage == 2:
        backbone_lr = 'BackBone LR: '+ str(0)
        fc_lr = 'GRU LR: '+ str(0)
    elif args.train_stage == 3:
        backbone_lr = 'BackBone LR: '+ str(0)
        fc_lr = 'GRU LR: '+ str(optimizer.param_groups[0]['lr'])

    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5, backbone_lr, fc_lr, prefix="Epoch: [{}]".format(epoch))

    logs = []
    end = time.time()

    # for training stage 2
    all_local_results = []
    all_targets = []

    if args.amp:
        assert scaler is not None

    for i, (images, target) in enumerate(train_loader):
        _b = target.shape[0]
        all_targets.append(target)
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target[:,0].cuda()
        input_prime = torch.nn.functional.interpolate(images, (args.glance_size, args.glance_size))
        input_prime = input_prime.cuda()
        
        if optimizer:
            optimizer.zero_grad()
        if args.train_stage == 1:
            if args.amp:
                with autocast():
                    output, pred = model(input=images, scan=input_prime, training=True, backbone_pred=False, one_step=False)
                    if args.consensus == 'gru':
                        loss = criterion(output, target.view(_b, -1).expand(_b, args.num_segments).reshape(-1))
                    elif args.consensus == 'fc':
                        loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output, pred = model(input=images, scan=input_prime, backbone_pred=False, one_step=False)
                if args.consensus == 'gru':
                    loss = criterion(output, target.view(_b, -1).expand(_b, args.num_segments).reshape(-1))
                elif args.consensus == 'fc':
                    loss = criterion(output, target)
                loss = criterion(output, target.view(_b, -1).expand(_b, args.num_segments).reshape(-1))
                loss.backward()
                optimizer.step()
        elif args.train_stage == 2:
            confidence_last = 0
            images = images.view(_b, args.num_segments, 3, model.input_size, model.input_size)
            # MDP focusing
            with torch.no_grad():
                global_feat_map, global_feat = model.glance(input_prime) # feat_map (B, T, C, H, W) feat_vec (B, T, _)
            local_results = []
            for focus_time_step in range(args.num_segments):
                img = images[:, focus_time_step, :, :]
                cur_global_map = global_feat_map[:, focus_time_step, :, :, :]
                cur_global_feat = global_feat[:, focus_time_step, :]
                if focus_time_step == 0:
                    # here output equals to pred
                    output, pred, patch_size_list, baseline_logits = model.one_step_act(img, cur_global_map, cur_global_feat, restart_batch=True, training=True)
                else:
                    output, pred, patch_size_list, baseline_logits = model.one_step_act(img, cur_global_map, cur_global_feat, restart_batch=False, training=True)
                local_results.append(pred)
                loss = criterion(output, target)
                confidence = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target.view(-1, 1)).view(1, -1)
                bsl_confidence = torch.gather(F.softmax(baseline_logits.detach(), 1), dim=1, index=target.view(-1, 1)).view(1, -1)
                # getting reward...
                reward, confidence_last = get_reward(args, confidence, confidence_last, bsl_confidence)
                model.focuser.memory.rewards.append(reward)
            all_local_results.append(torch.stack(local_results))
            model.focuser.update()
        elif args.train_stage == 3:
            assert args.random_patch == False
            if args.amp:
                with autocast():
                    outputs, pred = model(input=images, scan=input_prime, training=False, backbone_pred=False, one_step=True, gpu=args.gpu)
                    loss = criterion(outputs, target.view(_b, -1).expand(_b, args.num_segments).reshape(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, pred = model(input=images, scan=input_prime, training=False, backbone_pred=False, one_step=True, gpu=args.gpu)
                loss = criterion(outputs, target.view(_b, -1).expand(_b, args.num_segments).reshape(-1))
                loss.backward()
                optimizer.step()
            model.module.focuser.memory.clear_memory()
        elif args.train_stage == 0:
            if args.amp:
                with autocast():
                    pred = model(input=images, scan=None, glancer=args.pretrain_glancer, backbone_pred=True, one_step=False)
                    pred = pred.mean(1, keepdim=True).squeeze(1)
                    loss = criterion(pred, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(input=images, scan=None, glancer=args.pretrain_glancer, backbone_pred=True, one_step=False)
                pred = pred.mean(1, keepdim=True).squeeze(1)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
        else:
            raise NotImplementedError("Undefined training stage.")

        # Update evaluation metrics
        acc1, acc5 = accuracy(pred, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        logs.append(progress.print(i))

    if args.train_stage == 2:
        for i in range(args.num_segments):
            if args.dataset == 'fcvid':
                mAP, _ = cal_map(torch.cat(all_local_results, 1)[i, :, :].cpu(), torch.cat(all_targets, 0).cpu())
            else:
                mAP, _ = cal_map(torch.cat(all_local_results, 1)[i, :, :].cpu(), torch.cat(all_targets, 0)[:, 0:1].cpu())
            logs.append('mAP @ time step {step}: {mAP:.5f}\n'.format(mAP=mAP, step=i))
            print('mAP @ time step {step}: {mAP:.5f}\n'.format(mAP=mAP, step=i))

    return logs


def get_reward(args, confidence, confidence_last, baseline):
    if args.reward == 'prev':
        reward = confidence - confidence_last
    elif args.reward == 'conf':
        reward = confidence
    elif args.reward == 'random':
        reward = confidence - baseline
    return reward, confidence

if __name__ == '__main__':
    main()
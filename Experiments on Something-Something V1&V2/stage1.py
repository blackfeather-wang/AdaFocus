import torch

torch.multiprocessing.set_sharing_strategy('file_system')

import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

from ops.dataset import TSNDataSet
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, ProgressMeter
from models.gfv_net import GFV
from basic_tools.utils import *
from basic_tools.checkpoint import save_checkpoint

import os
import time
import hydra
import shutil
import warnings
import basic_tools
from collections import OrderedDict

best_acc1 = 0


@hydra.main(config_path="conf", config_name="stage1.yaml")
def main(args):
    assert args.train_stage == 1, "This code is only used for stage-1 training!"
    config_yaml = basic_tools.start(args)
    with open('training.log', 'a+') as f_handler:
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
    print("curent thread rank:", args.rank)

    num_class, args.train_list, args.val_list, args.root_path, prefix = \
        dataset_config.return_dataset(args.dataset, modality='RGB', root_dataset=args.data_dir)
    args.num_classes = num_class

    # create model
    model = GFV(args)
    if args.pretrained_glancer:
        resume_ckpt = torch.load(args.pretrained_glancer, map_location='cpu')

        new_state_dict = OrderedDict()
        for k, v in resume_ckpt['state_dict'].items():
            if k[:18] == 'module.base_model.':
                name = k[18:]  # remove `module.`
                new_state_dict[name] = v
            elif k[:14] == 'module.new_fc.':
                name = 'classifier.' + k[14:]  # replace `module.new_fc` with 'classifier'
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        model.glancer.net.load_state_dict(new_state_dict, strict=True)

        print('Load Pretrained Glancer from {}! '.format(args.pretrained_glancer))
        with open('training.log', 'a+') as f_handler:
            f_handler.writelines('Load Pretrained Glancer from {}! '.format(args.pretrained_glancer))

    if args.pretrained_focuser:
        resume_ckpt = torch.load(args.pretrained_focuser, map_location='cpu')

        new_state_dict = OrderedDict()
        for k, v in resume_ckpt['state_dict'].items():
            if k[:7] == 'module.':
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        model.focuser.net.load_state_dict(new_state_dict, strict=False)

        if args.evaluate or args.load_pretrained_focuser_fc:
            print('Load Pretrained Focuser FC Weight! ')
            with open('training.log', 'a+') as f_handler:
                f_handler.writelines('Load Pretrained Focuser FC Weight! ')
            new_state_dict = OrderedDict()
            new_fc_state_ditc = OrderedDict()
            for k, v in resume_ckpt['state_dict'].items():
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

        print('Load Pretrained Focuser from {}! '.format(args.pretrained_focuser))
        with open('training.log', 'a+') as f_handler:
            f_handler.writelines('Load Pretrained Focuser from {}! '.format(args.pretrained_focuser))

    model.focuser.net.base_model = torch.nn.Sequential(*list(model.focuser.net.base_model.children())[:-1])

    with open('training.log', 'a+') as f_handler:
        f_handler.writelines('model: {}'.format(model))
    scale_size = model.scale_size
    crop_size = model.crop_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset
                                                              or 'jester' in args.dataset else True)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # print(args.gpu)
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
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    policies = model.module.focuser.net.get_optim_policies()
    optimizer = torch.optim.SGD(policies + [{'params': model.module.classifier.parameters()}],
                                lr=0,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # data loading code
    normalize = GroupNormalize(input_mean, input_std)
    train_dataset = TSNDataSet(args.root_path, args.train_list,
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
                               ]), dense_sample=args.dense_sample)

    val_dataset = TSNDataSet(args.root_path, args.val_list,
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
                             ]), dense_sample=args.dense_sample)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.workers,
                                               pin_memory=False, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)

    if args.resume:
        resume_ckpt = torch.load(os.path.expanduser(args.resume))

        start_epoch = resume_ckpt['epoch']
        print('resume from epoch: {}'.format(start_epoch))

        model.module.glancer.load_state_dict(resume_ckpt['glancer'], strict=True)
        model.module.focuser.load_state_dict(resume_ckpt['focuser'], strict=True)
        model.module.classifier.load_state_dict(resume_ckpt['fc'], strict=True)
        optimizer.load_state_dict(resume_ckpt['optimizer'])

        best_acc1 = resume_ckpt['best_acc'].to(args.gpu)
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
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train_logs = train(train_loader, model, criterion, optimizer, epoch, args, scaler)

        if epoch % args.eval_freq == 0 or epoch > args.start_eval:
            # evaluate on validation set
            acc1, val_logs = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # TODO: save scaler
        # logging and checkpoint in the main thread
        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'glancer': model.module.glancer.state_dict(),
                'focuser': model.module.focuser.state_dict(),
                'fc': model.module.classifier.state_dict(),
                'acc': acc1,
                'best_acc': best_acc1,
                'optimizer': optimizer.state_dict()})
            if is_best:
                shutil.copyfile('checkpoint.pth.tar', 'checkpoint.pth.tar'.replace('checkpoint', 'model_best'))
            with open('training.log', 'a+') as f_handler:
                f_handler.writelines(train_logs)
                if epoch % args.eval_freq == 0:
                    f_handler.writelines(val_logs)

    output = ('Best Testing Results: Prec@1 {top1:.3f}'.format(top1=best_acc1))
    print(output)
    with open('training.log', 'a+') as f_handler:
        f_handler.writelines(output)


def train(train_loader, model, criterion, optimizer, epoch, args, scaler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    backbone_lr = 'Focuser BackBone LR: ' + str(optimizer.param_groups[0]['lr'])
    fc_lr = 'FC LR: ' + str(optimizer.param_groups[-1]['lr'])
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5, backbone_lr,
                             fc_lr, prefix="Epoch: [{}]".format(epoch))

    logs = []
    model.train()
    model.module.glancer.eval()
    end = time.time()

    if args.amp:
        assert scaler is not None

    for i, (glancer_images, focuser_images, target) in enumerate(train_loader):
        _b = target.shape[0]
        data_time.update(time.time() - end)
        glancer_images = glancer_images.cuda()
        focuser_images = focuser_images.cuda()
        target = target.cuda()
        glancer_images = torch.nn.functional.interpolate(glancer_images, (args.glance_size, args.glance_size))
        glancer_images = glancer_images.cuda()

        optimizer.zero_grad()
        if args.amp:
            with autocast():
                pred = model(input=focuser_images, scan=glancer_images, training=True, backbone_pred=False,
                             one_step=False)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(input=focuser_images, scan=glancer_images, backbone_pred=False, one_step=False)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

        # Update evaluation metrics
        acc1, acc5 = accuracy(pred, target, topk=(1, 5))
        losses.update(loss.item(), glancer_images.size(0))
        top1.update(acc1[0], glancer_images.size(0))
        top5.update(acc5[0], glancer_images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            logs.append(progress.print(i))

    return logs


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix='Test: ')

    logs = []
    # switch to evaluate mode
    model.eval()

    all_results = []
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
            pred = model(input=focuser_images, scan=glancer_images, training=False, backbone_pred=False, one_step=False)
            loss = criterion(pred, target)
            all_results.append(pred)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(pred, target, topk=(1, 5))
            losses.update(loss.item(), glancer_images.size(0))
            top1.update(acc1[0], glancer_images.size(0))
            top5.update(acc5[0], glancer_images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader) - 1:
                logs.append(progress.print(i))

        output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(top1=top1, top5=top5, loss=losses))
        logs.append(output)
        print(output)

    return top1.avg, logs


def save_model(prefix, model, i):
    filename = os.path.join(os.getcwd(), f"{prefix}-{i}.pt")
    torch.save(model, filename)
    print(f"[{i}] Saving {prefix} to {filename}")


if __name__ == '__main__':
    main()

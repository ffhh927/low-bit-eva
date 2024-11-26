# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer
from models.resnet20 import ResNet as resnet20
from models.autoencoder import Autoencoder
from bitsandbytes.optim import Adam8bit, SGD8bit, Adam32bit, SGD32bit # 导入AdamW8bit
from kfac import EVA, EVA_Fused, EVA8bit, EVA8bit_Fused, EVA8bit_CPU
from shampoo import ShampooSGD4bit_svd, ShampooSGD4bit
import torch_optimizer as optim
from datetime import datetime
# from fused import FusedEVA
# parsers

import random

def set_seed(seed):
    random.seed(seed)  # Python的随机性
    np.random.seed(seed)  # NumPy的随机性
    torch.manual_seed(seed)  # PyTorch的CPU随机性
    torch.cuda.manual_seed(seed)  # PyTorch的GPU随机性
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    torch.backends.cudnn.deterministic = True  # 确保卷积操作的可重复性
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN自动优化（可能会稍慢）

# 设置随机种子
seed = 42
set_seed(seed)
    
def worker_init_fn(worker_id):
    np.random.seed(worker_id + seed)  # 每个worker使用不同的种子

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default="cifar10")
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

args = parser.parse_args()

# take in args
# usewandb = ~args.nowandb
usewandb = False
if usewandb:
    import wandb
    watermark = "{}_{}_lr{}_{}_bs{}_epochs{}".format(args.opt, args.net, args.lr, args.dataset, args.bs, args.n_epochs)
    wandb_file_path = "/workspace/vision-transformers-cifar10/wandb/{}".format(watermark)
    os.environ["WANDB_DIR"] = wandb_file_path   
    wandb.init(project="cifar10-challange-11-21",
            name=watermark,
            mode='offline')
    wandb.config.update(args)
    log_file = "wandb_offline.log"
    with open(log_file, "a") as f:  # 使用 'a' 模式进行追加
        f.write(f"wandb sync {wandb_file_path}\n")

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize


if args.dataset == 'mnist':
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
else:    
    # Prepare dataset
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2,
                                          worker_init_fn=worker_init_fn,  # 确保每个worker的种子固定
                                            generator=torch.Generator().manual_seed(seed)  # 固定数据洗牌顺序
                                        )

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# # Add RandAugment with N, M(hyperparameter)
# if aug:  
#     N = 2; M = 14;
#     transform_train.transforms.insert(0, RandAugment(N, M))

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res20':
    net = resnet20()
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=='autoencoder':
    net = Autoencoder()
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = 10
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=10,
                downscaling_factors=(2,2,2,1))

# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = Adam32bit(net.parameters(), lr=args.lr, percentile_clipping=90) 
elif args.opt == 'adam8bit':
    optimizer = Adam8bit(net.parameters(), lr=args.lr, percentile_clipping=90) 
elif args.opt == "sgd":
    optimizer = SGD32bit(net.parameters(), lr=args.lr, weight_decay = 5e-4,momentum = 0.9)  
elif args.opt == "sgd8bit":
    optimizer = SGD8bit(net.parameters(), lr=args.lr, weight_decay = 5e-4,momentum = 0.9)  
elif args.opt == 'eva':
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #optimizer = AdamW8bit(net.parameters(), lr=lr)
    optimizer = EVA(net, lr=args.lr)
elif args.opt == 'eva8bit':
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #optimizer = AdamW8bit(net.parameters(), lr=lr)
    optimizer = EVA8bit(net, lr=args.lr)
elif args.opt == 'eva8bit_cpu':
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #optimizer = AdamW8bit(net.parameters(), lr=lr)
    optimizer = EVA8bit_CPU(net, lr=args.lr, sgd_momentum=0.9)
elif args.opt == 'eva_fused':
    optimizer = EVA_Fused(net, lr=args.lr)
elif args.opt == 'eva8bit_fused':
    optimizer = EVA8bit_Fused(net, lr=args.lr)
elif args.opt == 'shampoo32bit':
    optimizer = optim.Shampoo(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
elif args.opt == 'shampoo4bit_svd':
    optimizer = ShampooSGD4bit_svd(net.parameters(),
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=0.0005,
                    nesterov=False,
                    start_prec_step=1,
                    stat_compute_steps=100,
                    prec_compute_steps=500,
                    stat_decay=0.95,
                    matrix_eps=1e-6,
                    prec_maxorder=1200,
                    prec_bits=4,
                    min_lowbit_size=4096,
                    quan_blocksize=64)   
elif args.opt == 'shampoo4bit':
    optimizer = ShampooSGD4bit(net.parameters(),
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=0.0005,
                    nesterov=False,
                    start_prec_step=1,
                    stat_compute_steps=100,
                    prec_compute_steps=500,
                    stat_decay=0.95,
                    matrix_eps=1e-6,
                    prec_maxorder=1200,
                    prec_bits=4,
                    min_lowbit_size=4096,
                    quan_blocksize=64)  
else:
    raise ValueError("Unsupported optimizer name. Choose 'eva','eva8bit','eva_fused','eva8bit_fused','adam8it', \
                        adam32bit', 'sgd32bit', 'sgd8bit', 'shampoo4bit', 'shampoo4bit_svd', 'shampoo32bit'.") 
        
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
#scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train(epoch, ismem = False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    count = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
       # with torch.cuda.amp.autocast(enabled=use_amp):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        #scaler.scale(loss).backward()
        loss.backward()
        # if args.opt == 'eva' or args.opt == 'eva8bit':
        #     preconditioner.step()
            # print(preconditioner.state)
        optimizer.step()
        # print(optimizer.state)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if ismem and count > 1:
            break
        count += 1
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

import pandas as pd
import os

def trace_handler(prof: torch.profiler.profile):
    # 计算最大内存分配和保留的内存，转换为 MB 并保留三位小数
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    max_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    # 创建 DataFrame 并存储信息
    df = pd.DataFrame({
        "Batch Size": [args.bs],
        "Optim": [args.opt],
        "Net": [args.net],
        "Max Memory Allocated (MB)": [round(max_memory_allocated, 3)],
        "Max Memory Reserved (MB)": [round(max_memory_reserved, 3)]
    })

    # 文件路径
    file_path = "./export_memory/memory_log_shampoo.xlsx"

    # 检查文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，追加数据并跳过表头
        with pd.ExcelWriter(file_path, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        # 如果文件不存在，创建文件并写入表头
        df.to_excel(file_path, index=False)

    # 打印 Batch Size 和内存使用信息
    print(f"Batch Size: {args.bs}")
    print(f"Optim: {args.opt}")
    print(f"Net: {args.net}")
    print(f"Max memory allocated: {max_memory_allocated:.3f} MB")
    print(f"Max memory reserved: {max_memory_reserved:.3f} MB")

    
def check_gpu():
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2)} MB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**2} MB")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2} MB")
    print("\n")

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)
    
net.cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

ismem = False
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    if ismem:
        # trainloss = train(epoch, ismem)
        # trainloss = train(epoch, ismem)
        # trainloss = train(epoch, ismem)
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=trace_handler,
            ) as prof:
            trainloss = train(epoch, ismem)
    else:
        trainloss = train(epoch, True)
        # trainloss = train(epoch)
        # val_loss, acc = test(epoch)
        
        # scheduler.step(epoch-1) # step cosine scheduling
        
        # list_loss.append(val_loss)
        # list_acc.append(acc)
        
        # # Log training..
        # if usewandb:
        #     wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        #     "epoch_time": time.time()-start})

        # # Write out csv..
        # with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        #     writer = csv.writer(f, lineterminator='\n')
        #     writer.writerow(list_loss) 
        #     writer.writerow(list_acc) 
        # print(list_loss)

# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))
    

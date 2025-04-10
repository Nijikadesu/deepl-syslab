#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import vgg  # 请确保vgg.py在同一目录下或能够正确导入
import thop  # 用于计算模型参数量与 FLOPs
import numpy as np

def test_accuracy(model, testloader, device):
    """
    使用测试数据集计算模型精度
    """
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return test_loss/len(testloader), acc

def profile_model(model, device):
    """
    使用 thop 库计算模型的参数量与 FLOPs
    输入为 (1, 3, 32, 32) 的张量（适用于 CIFAR 系列数据集）
    """
    model.eval()
    input_tensor = torch.randn(1, 3, 32, 32).to(device)
    flops, params = thop.profile(model, inputs=(input_tensor, ))
    return params, flops

def main():
    parser = argparse.ArgumentParser(description='Test pruned VGG model performance and compute FLOPs/params')
    parser.add_argument('--dir_data', type=str, required=True, help='数据集存放路径')
    parser.add_argument('--baseline', type=str, required=True, help='剪枝前模型权重文件路径')
    parser.add_argument('--pruned', type=str, required=True, help='剪枝后（微调前）模型权重文件路径')
    parser.add_argument('--finetune', type=str, required=True, help='微调后模型权重文件路径')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='数据集名称')
    parser.add_argument('--depth', type=int, default=11, help='VGG 网络深度')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # 数据预处理：对于 CIFAR10/100 均采用标准归一化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # 加载测试数据集
    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root=args.dir_data, train=False, download=True, transform=transform_test)
    else:
        testset = torchvision.datasets.CIFAR100(root=args.dir_data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    print("========== Testing baseline model (pre-pruning) ==========")
    # 加载剪枝前模型（Baseline）
    model_baseline = vgg.vgg(dataset=args.dataset, depth=args.depth)
    print("=> Loading baseline checkpoint from '{}'".format(args.baseline))
    checkpoint = torch.load(args.baseline, map_location='cpu')
    model_baseline.load_state_dict(checkpoint['state_dict'], strict=False)
    model_baseline.to(device)
    _, acc_baseline = test_accuracy(model_baseline, testloader, device)
    params_baseline, flops_baseline = profile_model(model_baseline, device)
    print("Baseline model accuracy: {:.2f}%".format(acc_baseline))
    print("Baseline model params: {} | FLOPs: {}".format(params_baseline, flops_baseline))
    print("============================================================\n")

    print("==== Testing pruned model (before fine-tuning) ====")
    # 剪枝后模型的 cfg 配置，根据任务要求：
    pruned_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256]
    model_pruned = vgg.vgg(dataset=args.dataset, depth=args.depth, cfg=pruned_cfg)
    print("=> Loading pruned checkpoint from '{}'".format(args.pruned))
    checkpoint = torch.load(args.pruned, map_location='cpu')
    model_pruned.load_state_dict(checkpoint['state_dict'], strict=False)
    model_pruned.to(device)
    _, acc_pruned = test_accuracy(model_pruned, testloader, device)
    params_pruned, flops_pruned = profile_model(model_pruned, device)
    print("Pruned model (before finetune) accuracy: {:.2f}%".format(acc_pruned))
    print("Pruned model (before finetune) params: {} | FLOPs: {}".format(params_pruned, flops_pruned))
    print("====================================================\n")

    print("====== Testing fine-tuned model ======")
    # 同样使用 pruned 的 cfg 构建模型
    model_finetune = vgg.vgg(dataset=args.dataset, depth=args.depth, cfg=pruned_cfg)
    print("=> Loading fine-tuned checkpoint from '{}'".format(args.finetune))
    checkpoint = torch.load(args.finetune, map_location='cpu')
    model_finetune.load_state_dict(checkpoint['state_dict'], strict=False)
    model_finetune.to(device)
    _, acc_finetune = test_accuracy(model_finetune, testloader, device)
    params_finetune, flops_finetune = profile_model(model_finetune, device)
    print("Fine-tuned model accuracy: {:.2f}%".format(acc_finetune))
    print("Fine-tuned model params: {} | FLOPs: {}".format(params_finetune, flops_finetune))
    print("====================================================\n")

if __name__ == '__main__':
    main()
    # python test.py --dir_data ./data --baseline ./run/pretrain_vgg_model.pth --pruned ./run/pruned_vgg_model.pth --finetune ./run/finetune_vgg_model.pth --dataset cifar10 --depth 11 --cuda


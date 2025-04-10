#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import numpy as np
import vgg  # 请确保vgg.py与本文件在同一目录下或已正确安装

def prune_model(model, cfg):
    """
    根据新的 cfg（例如：[64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256]）
    和卷积核的 L1 范数计算生成通道掩膜（cfg_mask）。
    掩膜中保留的通道对应值为 1，剪枝通道对应值为 0。
    """
    cfg_mask = []
    layer_id = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            # 若当前卷积层的输出通道数已经等于新cfg中设定的输出通道数，则全部保留
            if out_channels == cfg[layer_id]:
                cfg_mask.append(torch.ones(out_channels))
                layer_id += 1
                continue
            # 否则，计算每个卷积核（滤波器）的 L1 范数
            weight_copy = m.weight.data.abs().clone()
            weight_copy = weight_copy.cpu().numpy()
            # 对每个滤波器（沿in_channels, kernel_h, kernel_w维度）求和
            L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            # 对 L1 范数进行排序（默认为升序）
            arg_max = np.argsort(L1_norm)
            # 根据新的 cfg 需要保留的通道数 n，
            # 从大到小取前 n 个输出通道的索引
            n = cfg[layer_id]
            arg_max_rev = arg_max[::-1][:n]
            # 生成掩膜：初始化全0，再将被保留的通道位置置为1
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            layer_id += 1
        elif isinstance(m, nn.MaxPool2d):
            # 对于池化层，cfg 中对应位置为 'M'，直接令 layer_id 加1
            layer_id += 1
    return cfg_mask

def transfer_weights(model, newmodel, cfg_mask):
    """
    根据生成的 cfg_mask 将原模型中需要保留的权重移植到新模型中。
    处理过程：
      1. 对 BatchNorm 层：根据当前层的输出通道掩膜（end_mask）选出需要保留的通道，移植权重、偏置及均值、方差。
      2. 对卷积层：分别根据前一层（start_mask）和当前层（end_mask）的掩膜选出输入和输出通道，移植卷积核权重。
      3. 对全连接层：对于第一层全连接，根据最后一层卷积的通道掩膜剪枝输入节点，其余层直接拷贝权重和偏置。
    """
    # 第一个卷积层的输入通道数为 3 (RGB)
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    end_mask = cfg_mask[layer_id_in_cfg]
    
    # 遍历原模型和新模型各层（两者结构顺序一致）
    for m0, m1 in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            # 得到当前BN层需要保留的通道索引
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            # 移植BN层权重、偏置及运行时参数
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            # 更新下一层的输入通道掩膜为当前层的输出通道掩膜
            layer_id_in_cfg += 1
            start_mask = end_mask
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            # 从 start_mask 中得到当前卷积层需要保留的输入通道索引
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            # 从 end_mask 中得到当前卷积层需要保留的输出通道索引
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            print('In shape: {:d}, Out shape: {:d}.'.format(idx0.size, idx1.size))
            # 移植卷积层权重：先对输入通道进行剪枝，再对输出通道剪枝
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            # 对于第一个全连接层，根据上一层卷积剪枝结果剪枝输入节点
            if layer_id_in_cfg == len(cfg_mask):
                idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0.tolist()].clone()
                m1.bias.data = m0.bias.data.clone()
                layer_id_in_cfg += 1
                continue
            # 其他全连接层直接全部保留
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

def main():
    parser = argparse.ArgumentParser(description='VGG 模型剪枝')
    parser.add_argument('--dataset', type=str, default='cifar10', help='数据集名称')
    parser.add_argument('--depth', type=int, default=11, help='VGG 网络深度')
    parser.add_argument('--model', type=str, required=True, help='预训练模型权重文件的路径')
    parser.add_argument('--save', type=str, required=True, help='剪枝后模型权重保存路径')
    parser.add_argument('--cuda', action='store_true', help='是否使用CUDA')
    args = parser.parse_args()

    # 加载预训练模型（剪枝前的Baseline）
    print("=> Loading pre-trained model from '{}'".format(args.model))
    model = vgg.vgg(dataset=args.dataset, depth=args.depth)
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if args.cuda:
        model.cuda()
    model.eval()

    # 新的剪枝配置 cfg：根据实验要求，这里将部分层的通道数由原来的512剪枝为256
    new_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256]
    
    # 任务3：根据新的cfg和卷积核的L1范数生成剪枝掩膜
    cfg_mask = prune_model(model, new_cfg)
    print("生成的每层剪枝掩膜：")
    for i, mask in enumerate(cfg_mask):
        print("Layer {}: kept {} / {}".format(i, int(torch.sum(mask).item()), mask.size(0)))
    
    # 根据新的 cfg 构建剪枝后的模型
    newmodel = vgg.vgg(dataset=args.dataset, depth=args.depth, cfg=new_cfg)
    if args.cuda:
        newmodel.cuda()
    
    # 任务4：根据通道掩膜，将原模型中保留的权重移植到新模型上
    transfer_weights(model, newmodel, cfg_mask)
    
    # 保存剪枝后的模型权重
    torch.save({'cfg': new_cfg, 'state_dict': newmodel.state_dict()}, args.save)
    print("剪枝后的模型已保存到 '{}'".format(args.save))

if __name__ == '__main__':
    main()

    # python vggprune.py --dataset cifar10 --depth 11 --model ./run/pretrain_vgg_model.pth --save ./run/pruned_vgg_model.pth --cuda


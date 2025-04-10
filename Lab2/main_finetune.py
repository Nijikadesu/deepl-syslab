#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from torch.utils.data import DataLoader
import vgg  # 请确保vgg.py在同一目录下或能正确导入

class FineTuner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        
        print("==> Preparing data...")
        # 数据预处理
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
        # 加载 CIFAR10 或 CIFAR100 数据集
        if self.args.dataset == 'cifar10':
            self.trainset = torchvision.datasets.CIFAR10(root=self.args.dir_data, train=True, download=True, transform=transform_train)
            self.testset  = torchvision.datasets.CIFAR10(root=self.args.dir_data, train=False, download=True, transform=transform_test)
        else:
            self.trainset = torchvision.datasets.CIFAR100(root=self.args.dir_data, train=True, download=True, transform=transform_train)
            self.testset  = torchvision.datasets.CIFAR100(root=self.args.dir_data, train=False, download=True, transform=transform_test)
        
        self.trainloader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        self.testloader  = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False, num_workers=2)
        
        print("==> Loading pruned model from '{}'".format(self.args.refine))
        # 加载剪枝后模型权重（假设剪枝时已保存cfg配置信息）
        checkpoint = torch.load(self.args.refine, map_location='cpu')
        cfg = checkpoint.get('cfg', None)
        self.net = vgg.vgg(dataset=self.args.dataset, depth=self.args.depth, cfg=cfg)
        self.net.load_state_dict(checkpoint['state_dict'], strict=False)
        self.net.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        self.best_acc = 0.0

    def train_epoch(self, epoch):
        self.net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 100 == 0:
                print('Epoch: {} | Batch: {} | Loss: {:.3f} | Acc: {:.3f}%'.format(
                    epoch, batch_idx, train_loss/(batch_idx+1), 100.*correct/total))
        return train_loss/len(self.trainloader), 100.*correct/total

    def test_epoch(self, epoch):
        self.net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        print('Test Epoch: {} | Loss: {:.3f} | Acc: {:.3f}%'.format(epoch, test_loss/len(self.testloader), acc))
        return test_loss/len(self.testloader), acc

    def save_checkpoint(self, epoch, acc):
        print("Saving best model with accuracy: {:.2f}%".format(acc))
        state = {
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'acc': acc,
            'cfg': self.args.cfg if hasattr(self.args, 'cfg') else None
        }
        torch.save(state, self.args.save)

    def run(self):
        for epoch in range(1, self.args.epochs+1):
            self.train_epoch(epoch)
            _, test_acc = self.test_epoch(epoch)
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint(epoch, self.best_acc)
        print("Fine-tuning finished. Best accuracy: {:.2f}%".format(self.best_acc))


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning pruned VGG model (Class Based)')
    parser.add_argument('--dir_data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--refine', type=str, required=True, help='Path to pruned model weights file')
    parser.add_argument('--save', type=str, required=True, help='Path to save fine-tuned model weights')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset name')
    parser.add_argument('--depth', type=int, default=11, help='VGG network depth')
    parser.add_argument('--epochs', type=int, default=20, help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    finetuner = FineTuner(args)
    finetuner.run()

if __name__ == '__main__':
    main()
    # python main_finetune.py --dir_data ./data --refine ./run/pruned_vgg_model.pth --save ./run/finetune_vgg_model.pth --cuda


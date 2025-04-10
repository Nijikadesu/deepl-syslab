import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from vgg import vgg  # 确保vgg.py在同一目录下


class VGG11PreTrainer:
    def __init__(self, args):
        self.args = args
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_data()  # 加载数据集
        self.load_model()  # 加载模型

    def load_data(self):
        # 数据预处理
        print('==> 准备数据集..')
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # 加载CIFAR10数据集
        trainset = torchvision.datasets.CIFAR10(
            root=self.args.dir_data, train=True, download=True, transform=train_transform)
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root=self.args.dir_data, train=False, download=True, transform=test_transform)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

    def load_model(self):
        # 初始化模型
        print('==> 构建VGG11模型..')
        model = vgg(dataset='cifar10', depth=11, init_weights=True).to(self.device)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=self.args.lr, 
            momentum=self.args.momentum, 
            weight_decay=self.args.weight_decay
        )
        self.scheduler = MultiStepLR(self.optimizer, milestones=[100, 150], gamma=0.1)  # 学习率衰减

    # 训练函数
    def train(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 99:  # 每100个batch打印一次
                print(f'Epoch: {epoch} | Batch: {batch_idx+1} | Loss: {loss.item():.3f} | '
                    f'Acc: {100.*correct/total:.2f}%')

    # 测试函数
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        print(f'Test Acc: {acc:.2f}%')
        return acc

    def start(self):
        # 主训练循环
        best_acc = 0
        for epoch in range(1, self.args.epochs + 1):
            print(f'\nEpoch: {epoch}')
            self.train(epoch)
            current_acc = self.test()
            self.scheduler.step()
            
            # 保存最佳模型
            if current_acc > best_acc:
                print('==> 保存最佳模型..')
                state = {
                    'state_dict': self.model.state_dict(),
                    'acc': current_acc,
                    'epoch': epoch,
                }
                torch.save(state, self.args.save)
                best_acc = current_acc

        print(f'训练完成，最佳验证准确率：{best_acc:.2f}%')


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='VGG11 Pre-training on CIFAR10')
    parser.add_argument('--dir_data', type=str, required=True, help='CIFAR10数据集路径')
    parser.add_argument('--save', type=str, required=True, help='预训练模型保存路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练epoch数 (默认: 200)')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小 (默认: 128)')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率 (默认: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量 (默认: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减 (默认: 5e-4)')
    args = parser.parse_args()

    pretrain = VGG11PreTrainer(args=args)
    pretrain.start()
    # python main.py --dir_data ./data --save ./run/pretrain_vgg_model.pth --epochs 200 --batch_size 128 --lr 0.1 --momentum 0.9 --weight_decay 5e-4

    
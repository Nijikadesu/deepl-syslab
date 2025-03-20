import os
import time
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


def load_cifar(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    test_set = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def init_model():
    mynet = MyNet()
    return mynet


def train(model, train_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print('Time:{} | Training process started.'.format(time.strftime("%H:%M:%S")))

    for epoch in range(args.n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % args.n_batches == 0:
                print(
                'Time:{} | Epoch:[{:2d}/{:2d}]({:4d}/{:4d}) | Avg loss:{:.3f}:'.format(
                    time.strftime("%H:%M:%S"),
                    epoch + 1,
                    args.n_epochs,
                    i + 1,
                    len(train_loader),
                    running_loss/args.n_batches))
                running_loss = 0.0

    os.makedirs(args.save_path, exist_ok=True)
    args.save_path = os.path.join(args.save_path, 'cifar_net.pt')
    print('Time:{} | Fished training, saving to {}.'.format(
        time.strftime("%H:%M:%S"),
        args.save_path
    ))
    torch.save(model.state_dict(), args.save_path)


def evaluate(model, test_loader, args):
    model.eval()
    correct = 0
    total = 0
    print('Time:{} | Evaluating process started.'.format(time.strftime("%H:%M:%S")))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(args.device), data[1].to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Time:{} | Finished evaluation, test acc: {:.1f}%'.format(time.strftime("%H:%M:%S"), 100 * correct / total))

    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(args.device), data[1].to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('=' * 50)
    for i in range(10):
        print("Time:{} | Acc on class {}: {:.1f}%".format(
            time.strftime("%H:%M:%S"),
            args.classes[i],
            100 * class_correct[i]/class_total[i]))
    print('=' * 50)


def main():
    parser = argparse.ArgumentParser(description="Lab 0: Cifar10 training config")
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--n_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=int, default=0.01)
    parser.add_argument("--momentum", type=int, default=0.9)
    parser.add_argument("--data_path", type=str, default='./dataset')
    parser.add_argument("--save_path", type=str, default='./out')
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    mynet = init_model().to(args.device)
    train_loader, test_loader, args.classes = load_cifar(args)

    train(mynet, train_loader, args)
    evaluate(mynet, test_loader, args)


if __name__ == '__main__':
    main()
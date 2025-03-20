import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


def load_cifar(data_path: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True, num_workers=2
    )

    test_set = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=4, shuffle=False, num_workers=2
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
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 输入尺寸修正为400（原题可能笔误）
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # 输出尺寸：6x14x14
        x = self.pool(F.relu(self.conv2(x)))     # 输出尺寸：16x5x5
        x = x.view(-1, 16 * 5 * 5)              # 展平为400维
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, train_loader, n_epochs, n_batches):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % n_batches == 0:
                print(f'everage running loss: {running_loss/n_batches:4f}')
                running_loss = 0.0

    print("训练完成")
    torch.save(model.state_dict(), "cifar_net.pth")


def evaluate(model, test_loader, classes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"测试集准确率: {100 * correct / total:.1f}%")

    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f"{classes[i]}的准确率: {100 * class_correct[i]/class_total[i]:.1f}%")


def main():
    train_loader, test_loader, classes = load_cifar(data_path = './dataset')
    mynet = MyNet().to('cuda')
    train(mynet, train_loader, 2, 100)
    evaluate(mynet, test_loader, classes)


if __name__ == '__main__':
    main()

import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5* 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))

        #展平
        x = x.view(-1, 16 * 5 * 5)

        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def load_cifar10():
    # 数据规范化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5) ,(0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=2
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes

def train_model(model, train_set, epochs, optimizer, device, criterion):
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, data in enumerate(train_set, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if idx % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch + 1} / {epochs}] | "
                      f"Batch [{idx:5d}] | "
                      f"loss: {avg_loss:.3f}")
                running_loss = 0.0

        print(f"Epoch {epoch + 1} completed")

    print("训练结束")
    torch.save(model.state_dict(), "CIFAR10_ConvNet.pth")

def eval_model(model, test_set, device, classes):
    model.load_state_dict(torch.load("CIFAR10_ConvNet.pth"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_set:
            imgs, labels = data[0].to(device), data[1].to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print(f"测试集准确率: {100 * correct / total:.1f}%")

    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data in test_set:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f"{classes[i]}的准确率: {100 * class_correct[i] / class_total[i]:.1f}%")


if __name__ == "__main__":
    train_loader, test_loader, classes= load_cifar10()
    mynet = ConvNet()
    optimizer = optim.SGD(mynet.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mynet.to(device)
    train_model(model=mynet, train_set=train_loader, epochs=20, optimizer=optimizer, device=device, criterion=criterion)
    eval_model(model=mynet, test_set=test_loader, classes=classes, device=device)



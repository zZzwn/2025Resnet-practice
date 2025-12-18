import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 超参数（可调：epochs小点加速）
num_epochs = 5  # 5 epochs，GPU上~10-20min
batch_size = 64
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10  # CIFAR-10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (automobile/truck 与自动驾驶相关)

# 数据准备：变换匹配ResNet输入
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet期望224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet均值
])

# 加载数据集（首次运行自动下载~170MB）
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 修改网络：预训练ResNet-18，换fc层
model = models.resnet18(weights='IMAGENET1K_V1')  # 加载ImageNet预训练
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 改成10类
model = model.to(device)

# 损失与优化器（迁移学习：全部微调，小lr防过拟合）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = train_loss / len(train_loader)
    acc = 100. * correct / total
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Acc: {acc:.2f}%')
    return avg_loss, acc

# 测试函数（评估）
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = test_loss / len(test_loader)
    acc = 100. * correct / total
    print(f'Epoch {epoch}: Test Loss: {avg_loss:.4f}, Acc: {acc:.2f}%')
    return avg_loss, acc

# 训练循环 + 保存最佳模型
train_losses, train_accs = [], []
test_losses, test_accs = [], []
best_acc = 0
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best.pth')
        print(f'Saved best model with acc {best_acc:.2f}%')

# 训练曲线可视化（matplotlib）
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label='Train Acc')
plt.plot(epochs, test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig('training_curves.png')  # 保存为png
plt.show()  # 本地显示
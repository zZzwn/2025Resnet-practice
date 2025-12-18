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

# 超参数（增加epochs）
start_epoch = 5  # 之前训了5个，从6开始（改成你的实际）
num_epochs = 10  # 总目标epochs（会训10-5=5个新）
batch_size = 64
learning_rate = 0.0001  # 继续训用小lr，防过拟合
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10

# 数据准备（同前）
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)  # 假设已下载
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 网络（同前）
model = models.resnet18(weights=None)  # 不加载预训，因为继续旧的
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 损失与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 加载检查点（继续训）
def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f'Loaded model from {checkpoint_path}')
    else:
        print(f'No checkpoint found at {checkpoint_path}. Starting from scratch.')

load_checkpoint('best.pth')  # 加载你的best.pth

# 训练/测试函数（同前）
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

# 训练循环（从start_epoch+1继续）
train_losses, train_accs = [], []  # 新曲线，从继续点开始
test_losses, test_accs = [], []
best_acc = 0  # 重置或从旧加载（简化，这里重置）
for epoch in range(start_epoch + 1, num_epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'new_best.pth')  # 保存新best
        print(f'Saved new best model with acc {best_acc:.2f}%')

# 新训练曲线可视化
epochs = range(start_epoch + 1, num_epochs + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('New Loss Curve')
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label='Train Acc')
plt.plot(epochs, test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('New Accuracy Curve')
plt.savefig('new_training_curves.png')
plt.show()
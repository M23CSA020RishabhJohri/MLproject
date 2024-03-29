import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

# Define MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

# Define CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7*7*64, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
class CNN_Modified_Kernel(nn.Module):
    def __init__(self):
        super(CNN_Modified_Kernel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Changed kernel size to 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=3)  # Changed kernel size to 7x7
        self.fc1 = nn.Linear(7*7*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
class CNN_Additional_Layers(nn.Module):
    def __init__(self):
        super(CNN_Additional_Layers, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # Additional convolutional layer
        self.fc1 = nn.Linear(3*3*128, 512)  # Adjusted for additional convolutional layer
        self.fc2 = nn.Linear(512, 128)  # Additional fully connected layer
        self.fc3 = nn.Linear(128, 10)  # Adjusted output from the new fc2 layer

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))  # Passing through the third convolutional layer
        x = x.view(-1, 3*3*128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Passing through the additional fully connected layer
        return F.log_softmax(self.fc3(x), dim=1)
class CNN_Modified_Filters(nn.Module):
    def __init__(self):
        super(CNN_Modified_Filters, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)  # Increased filters to 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # Increased filters to 128
        self.fc1 = nn.Linear(7*7*128, 256)  # Adjusted for the increased number of filters
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7*7*128)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# Load Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training Function
def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + batch_idx)

# Test Function
def test(model, device, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_pred.extend(output.tolist())
            y_true.extend(target.tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true, np.argmax(y_pred, axis=1), average='macro')
    conf_matrix = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
    writer.add_scalar('Accuracy', accuracy, epoch)
    writer.add_scalar('Precision', precision, epoch)
    writer.add_scalar('Recall', recall, epoch)
    writer.add_scalar('F1 Score', f_score, epoch)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)'
          f'\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1: {f_score:.4f}\n')
# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10

models = {'MLP': MLP().to(device), 'CNN': CNN().to(device)}
optimizers = {name: Adam(model.parameters()) for name, model in models.items()}
writers = {name: SummaryWriter(f'runs/{name}_experiment') for name in models}

for name, model in models.items():
    print(f'Training {name}...')
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizers[name], epoch, writers[name])
        test(model, device, test_loader, writers[name], epoch)
    writers[name].close()

print("Training complete. Please run `tensorboard --logdir=runs` to view the results.")

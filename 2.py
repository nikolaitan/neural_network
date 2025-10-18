#3.1
import torch
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader, Subset
import numpy as np

# Параметры данных
img_size = 28
num_classes_cnn = 26  # A-Z
num_classes_mlp = 3   # A,B,C

# Трансформации для CNN
transform_cnn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Трансформации для MLP
transform_mlp = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))  # Вытягиваем в вектор
])


#3.2
# Загрузка EMNIST для букв
train_dataset = EMNIST(root='./data', split='letters', train=True, 
                      download=True, transform=transform_cnn)
test_dataset = EMNIST(root='./data', split='letters', train=False, 
                     transform=transform_cnn)

# Фильтрация для MLP (только A,B,C)
def filter_abc(dataset, classes=[1,2,3]):  # A=1, B=2, C=3 в EMNIST
    indices = [i for i, (_, label) in enumerate(dataset) 
              if label in classes]
    return Subset(dataset, indices)

train_mlp = filter_abc(train_dataset)
test_mlp = filter_abc(test_dataset)


#4.1
import torch.nn as nn
import torch.nn.functional as F

class SymbolCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(SymbolCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

#4.2
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, num_classes=3):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


#5.1
def train_cnn_model():
    # DataLoader для CNN
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model = SymbolCNN(num_classes=26)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение
    for epoch in range(15):
        model.train()
        for images, labels in train_loader:
            # EMNIST labels: 1-26, преобразуем в 0-25
            labels = labels - 1
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Валидация
        accuracy = validate_model(model, test_loader)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy:.2f}%')
    
    return model


#5.2
def train_mlp_model():
    # DataLoader для MLP с преобразованием
    train_loader_mlp = DataLoader(train_mlp, batch_size=64, shuffle=True)
    test_loader_mlp = DataLoader(test_mlp, batch_size=64, shuffle=False)
    
    model = SimpleMLP(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        model.train()
        for images, labels in train_loader_mlp:
            # Преобразуем изображение в вектор и метки в 0-2
            images = images.view(images.size(0), -1)
            labels = labels - 1  # A=0, B=1, C=2
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        accuracy = validate_mlp(model, test_loader_mlp)
        print(f'MLP Epoch {epoch+1}, Accuracy: {accuracy:.2f}%')
    
    return model


#5.3
def validate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels - 1
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def validate_mlp(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            labels = labels - 1
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


#6.2
def analyze_errors(model, test_loader, is_mlp=False):
    model.eval()
    confusion_matrix = torch.zeros(3, 3) if is_mlp else torch.zeros(26, 26)
    
    with torch.no_grad():
        for images, labels in test_loader:
            if is_mlp:
                images = images.view(images.size(0), -1)
                labels = labels - 1
            else:
                labels = labels - 1
                
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix


#8
# Основной скрипт запуска
if __name__ == "__main__":
    print("Запуск эксперимента по распознаванию символов...")
    
    # Создаем DataLoader для тестовых данных
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    test_loader_mlp = DataLoader(test_mlp, batch_size=64, shuffle=False)
    
    # Обучение CNN
    print("Обучение CNN модели...")
    cnn_model = train_cnn_model()
    
    # Обучение MLP
    print("Обучение MLP модели...")
    mlp_model = train_mlp_model()
    
    # Сравнительный анализ
    cnn_accuracy = validate_model(cnn_model, test_loader)
    mlp_accuracy = validate_mlp(mlp_model, test_loader_mlp)
    
    print(f"Результаты:")
    print(f"CNN точность: {cnn_accuracy:.2f}%")
    print(f"MLP точность: {mlp_accuracy:.2f}%")
    
    # Сохранение моделей
    torch.save(cnn_model.state_dict(), 'cnn_model.pth')
    torch.save(mlp_model.state_dict(), 'mlp_model.pth')

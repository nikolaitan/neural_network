import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
 
# Генерация данных 
def f(x):
    return x**3 + 0.5*x**2 - 4*x 
 
X = np.linspace(-5, 5, 1000)
y = f(X) + np.random.normal(0, 0.5, X.shape)  # Добавление шума 
 
# Преобразование в тензоры PyTorch 
X = torch.tensor(X, dtype=torch.float32).view(-1, 1)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
 
# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Архитектуры моделей
class Model_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.layer(x)
 
class Model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)
 
class Model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)
 
class Model_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)
 
# Обучение моделей
def train_model(model, epochs=1500, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
    
    with torch.no_grad():
        test_loss = criterion(model(X_test), y_test)
        print(f'Test MSE: {test_loss.item():.4f}')
    
    return losses, test_loss.item()
 
# Запуск экспериментов 
models = [Model_0(), Model_1(), Model_2(), Model_3()]
results = {}
 
for i, model in enumerate(models):
    print(f"\nTraining Model_{i}...")
    train_loss, test_loss = train_model(model)
    results[f"Model_{i}"] = {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "predictions": model(X).detach().numpy()
    }
 
# Визуализация результатов 
plt.figure(figsize=(15, 10))
 
# Графики потерь
plt.subplot(2, 1, 1)
for label, res in results.items():
    plt.plot(res["train_loss"], label=label)
plt.title('Training Loss (MSE)')
plt.legend()
 
# Графики функций
plt.subplot(2, 1, 2)
plt.scatter(X, y, s=2, label='Исходные данные', alpha=0.5)
for label, res in results.items():
    plt.plot(X, res["predictions"], linewidth=2, label=label)
plt.title('Аппроксимация функции')
plt.legend()
plt.savefig('results.png')

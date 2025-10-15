#Первое издание
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 生成数据
x = np.linspace(-5, 5, 1000)
y = x**3 - 2*x**2 + 3*x - 1

# 转换为PyTorch张量
x_tensor = torch.FloatTensor(x).reshape(-1, 1)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# 手动分割数据集（替代sklearn的train_test_split）
def manual_train_test_split(x, y, test_size=0.2):
    # 随机打乱索引
    indices = torch.randperm(len(x))
    
    # 计算测试集大小
    test_size = int(len(x) * test_size)
    
    # 分割索引
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # 根据索引分割数据
    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return x_train, x_test, y_train, y_test

# 分割数据集
x_train, x_test, y_train, y_test = manual_train_test_split(x_tensor, y_tensor, test_size=0.2)

# 定义神经网络模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# 初始化模型、损失函数和优化器
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
train_losses = []
test_losses = []
epochs = 500

for epoch in range(epochs):
    # 训练阶段
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # 测试阶段
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        test_loss = criterion(y_test_pred, y_test)
        test_losses.append(test_loss.item())
    
    # 每50个epoch打印一次进度
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# 可视化结果
plt.figure(figsize=(15, 5))

# 1. 损失函数曲线
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# 2. 函数拟合结果
plt.subplot(1, 3, 2)
model.eval()
with torch.no_grad():
    y_pred_full = model(x_tensor)
    
plt.plot(x, y, label='True Function')
plt.plot(x, y_pred_full.numpy(), label='MLP Approximation')
plt.title('Function Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 3. 训练和测试数据点
plt.subplot(1, 3, 3)
plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, label='Train Data')
plt.scatter(x_test.numpy(), y_test.numpy(), alpha=0.5, label='Test Data')
plt.plot(x, y, 'k-', label='True Function', linewidth=2)
plt.title('Train/Test Data Split')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印最终结果
print(f'Final Train MSE: {train_losses[-1]:.6f}')
print(f'Final Test MSE: {test_losses[-1]:.6f}')

# 评估模型性能
if test_losses[-1] > train_losses[-1] * 1.5:
    print("警告：可能存在过拟合")
else:
    print("模型表现良好，过拟合风险较低")
'''

#Второе издание
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据生成
def generate_data(n_samples=1000):
    x = np.linspace(-5, 5, n_samples)
    y = x**3 - 3*x**2 + 2*x - 1  # 新的三次函数
    return x, y

x, y = generate_data()
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# 转换为PyTorch张量
x_tensor = torch.FloatTensor(x)
y_tensor = torch.FloatTensor(y)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_tensor, y_tensor, test_size=0.2, random_state=42
)

# 2. 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# 3. 模型初始化
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练过程
train_losses = []
test_losses = []
epochs = 500

for epoch in range(epochs):
    # 训练模式
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test)
        test_loss = criterion(test_pred, y_test)
    
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# 5. 最终评估
model.eval()
with torch.no_grad():
    final_train_pred = model(x_train)
    final_test_pred = model(x_test)
    final_train_loss = criterion(final_train_pred, y_train)
    final_test_loss = criterion(final_test_pred, y_test)

print(f'Final Train MSE: {final_train_loss.item():.6f}')
print(f'Final Test MSE: {final_test_loss.item():.6f}')

# 6. 可视化结果
plt.figure(figsize=(15, 5))

# 损失曲线
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training and Test Loss')

# 函数近似效果
plt.subplot(1, 3, 2)
x_plot = np.linspace(-5, 5, 1000).reshape(-1, 1)
x_plot_tensor = torch.FloatTensor(x_plot)
model.eval()
with torch.no_grad():
    y_plot_pred = model(x_plot_tensor).numpy()

y_plot_true = x_plot**3 - 3*x_plot**2 + 2*x_plot - 1

plt.plot(x_plot, y_plot_true, label='True Function', linewidth=2)
plt.plot(x_plot, y_plot_pred, label='MLP Approximation', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Function Approximation')

# 预测vs真实值散点图
plt.subplot(1, 3, 3)
plt.scatter(y_test.numpy(), final_test_pred.numpy(), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs True Values')

plt.tight_layout()
plt.show()

# 7. 保存模型
torch.save(model.state_dict(), 'mlp_function_approximation.pth')

'''

#Третье издание
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Установка случайного сида
torch.manual_seed(42)
np.random.seed(42)

# 1. Генерация данных
def generate_data():
    x = np.linspace(-5, 5, 1000)
    # Новая кубическая функция: y = 2x³ - 3x² + x + 2
    y = 2*x**3 - 3*x**2 + x + 2
    return x, y

# 2. Предварительная обработка данных
x, y = generate_data()
x_tensor = torch.FloatTensor(x).reshape(-1, 1)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Разделение набора данных
x_train, x_test, y_train, y_test = train_test_split(
    x_tensor, y_tensor, test_size=0.2, random_state=42
)

# 3. Определение модели нейронной сети
class MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# 4. Инициализация модели, функции потерь и оптимизатора
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Обучение модели
train_losses = []
test_losses = []
epochs = 500

for epoch in range(epochs):
    # Режим обучения
    model.train()
    optimizer.zero_grad()
    
    # Прямое распространение
    y_pred = model(x_train)
    train_loss = criterion(y_pred, y_train)
    
    # Обратное распространение
    train_loss.backward()
    optimizer.step()
    
    # Режим оценки
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        test_loss = criterion(y_test_pred, y_test)
    
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    
    if epoch % 100 == 0:
        print(f'Эпоха {epoch}, Потери на обучении: {train_loss.item():.6f}, Потери на тесте: {test_loss.item():.6f}')

# 6. Визуализация результатов
plt.figure(figsize=(15, 5))

# Кривая функции потерь
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Потери на обучении')
plt.plot(test_losses, label='Потери на тесте')
plt.xlabel('Эпоха')
plt.ylabel('MSE Потери')
plt.legend()
plt.title('Функция потерь')

# Результат аппроксимации функции
plt.subplot(1, 3, 2)
x_plot = np.linspace(-5, 5, 1000)
y_true = 2*x_plot**3 - 3*x_plot**2 + x_plot + 2

model.eval()
with torch.no_grad():
    x_tensor_plot = torch.FloatTensor(x_plot).reshape(-1, 1)
    y_pred_plot = model(x_tensor_plot).numpy()

plt.plot(x_plot, y_true, label='Истинная функция', linewidth=2)
plt.plot(x_plot, y_pred_plot, label='Аппроксимация НС', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Аппроксимация функции')

# Распределение ошибки
plt.subplot(1, 3, 3)
error = y_true - y_pred_plot.flatten()
plt.plot(x_plot, error)
plt.xlabel('x')
plt.ylabel('Ошибка')
plt.title('Распределение ошибки')
plt.axhline(y=0, color='r', linestyle='-')

plt.tight_layout()
plt.show()

# Финальная оценка
final_train_loss = train_losses[-1]
final_test_loss = test_losses[-1]
print(f'\nФинальные потери на обучении (MSE): {final_train_loss:.6f}')

print(f'Финальные потери на тесте (MSE): {final_test_loss:.6f}')

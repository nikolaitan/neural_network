#Первое издание
'''
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# Проверить доступность устройства CUDA и выбрать устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(torch.nn.Module):  # Наследование
    def __init__(self):  # Метод инициализации
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)  # Первый полносвязный слой: преобразует входные данные размерности 28 * 28 в 64 признака
        self.fc2 = torch.nn.Linear(64, 64)  # Второй полносвязный слой: преобразует 64 признака в 64 признака
        self.fc3 = torch.nn.Linear(64, 64)  # Третий полносвязный слой: преобразует 64 признака в 64 признака
        self.fc4 = torch.nn.Linear(64, 10)  # Четвертый полносвязный слой: преобразует 64 признака в 10 выходных классов

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))  # Применение ReLU-активации к выходу первого слоя
        x = torch.nn.functional.relu(self.fc2(x))  # Применение ReLU-активации к выходу второго слоя
        x = torch.nn.functional.relu(self.fc3(x))  # Применение ReLU-активации к выходу третьего слоя
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)  # Применение логарифмического softmax для получения вероятностей классов
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST(root='mnist_data/', train=is_train, download=True, transform=to_tensor)
    return DataLoader(data_set, batch_size=15, shuffle=True, pin_memory=True)

def evaluate(test_data, net):
    net.eval()  # Установить модель в режим оценки
    n_correct = 0  # Счетчик правильных предсказаний
    n_total = 0  # Общее количество тестовых примеров
    
    with torch.no_grad():  # Отключение вычисления градиентов
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)  # Перемещение данных на устройство
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    net.to(device)  # Перемещение модели на устройство

    print("Начальная точность:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(2):
        net.train()  # Установить модель в режим обучения
        for (x, y) in train_data:
            x, y = x.to(device), y.to(device)  # Перемещение данных на устройство
            net.zero_grad()  # Обнуление градиентов
            outputs = net.forward(x.view(-1, 28 * 28))  # Прямой проход
            loss = torch.nn.functional.nll_loss(outputs, y)  # Вычисление потерь
            loss.backward()  # Обратное распространение ошибки
            optimizer.step()  # Обновление параметров
        print("Эпоха:", epoch, "Точность:", evaluate(test_data, net))

    plt.figure(figsize=(10, 5))
    for n, (x, _) in enumerate(test_data):
        if n > 3:
            break
        x = x.to(device)
        prediction = torch.argmax(net(x.view(-1, 28 * 28)))
        plt.subplot(2, 2, n + 1)
        plt.imshow(x[0].view(28, 28).cpu().numpy(), cmap='gray')  # Убедиться, что данные на CPU для визуализации
        plt.title("Предсказание: " + str(prediction.item()))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
    '''


#Второе издание
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

# Проверка доступности устройства CUDA и выбор устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST(root='mnist_data/', train=is_train, download=True, transform=to_tensor)
    return DataLoader(data_set, batch_size=15, shuffle=True, pin_memory=True)

def evaluate(test_data, net):
    net.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

def analyze_specific_image(net, image_index, test_data):
    """Анализ точности распознавания конкретного изображения"""
    net.eval()
    with torch.no_grad():
        # Получение конкретного изображения
        data_iter = iter(test_data)
        images, labels = next(data_iter)
        for _ in range(image_index):
            images, labels = next(data_iter)
        specific_image = images[0].to(device)
        specific_label = labels[0].to(device)
        
        # Прогноз
        output = net.forward(specific_image.view(-1, 28 * 28))
        probabilities = torch.exp(output)  # Преобразование вывода log_softmax в вероятности
        predicted_class = torch.argmax(probabilities).item()
        true_class = specific_label.item()
        
        # Получение вероятностей для всех классов
        prob_list = probabilities.cpu().numpy()[0]
        
        return specific_image, true_class, predicted_class, prob_list

def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    net.to(device)
    
    print("Начальная точность:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Запись процесса обучения
    train_losses = []
    accuracies = []
    
    for epoch in range(2):
        net.train()
        epoch_loss = 0
        for (x, y) in train_data:
            x, y = x.to(device), y.to(device)
            net.zero_grad()
            outputs = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_data)
        train_losses.append(avg_loss)
        accuracy = evaluate(test_data, net)
        accuracies.append(accuracy)
        print(f"Эпоха {epoch}: Потери = {avg_loss:.4f}, Точность = {accuracy:.4f}")
    
    # Построение кривых обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Потери обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('Кривая потерь обучения')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Тестовая точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.title('Кривая точности')
    plt.legend()
    plt.show()
    
    # Анализ конкретного изображения (например, 5-е изображение в тестовом наборе)
    image_index = 5  # Можно изменить на любой индекс
    specific_image, true_class, predicted_class, prob_list = analyze_specific_image(net, image_index, test_data)
    
    print(f"\nАнализ конкретного изображения (индекс {image_index}):")
    print(f"Истинный класс: {true_class}")
    print(f"Предсказанный класс: {predicted_class}")
    print(f"Распределение вероятностей предсказания:")
    for i, prob in enumerate(prob_list):
        print(f"  Класс {i}: {prob:.4f}")
    
    # Отображение конкретного изображения и прогноза
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(specific_image.cpu().view(28, 28).numpy(), cmap='gray')
    plt.title(f"Истинный: {true_class}, Предсказанный: {predicted_class}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), prob_list)
    plt.xlabel('Класс')
    plt.ylabel('Вероятность')
    plt.title('Распределение вероятностей по классам')
    plt.xticks(range(10))
    plt.show()

if __name__ == '__main__':
    main()
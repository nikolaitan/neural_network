# -*- coding: utf-8 -*-
"""
Эксперимент по проектированию линейного фильтра с использованием нейронных сетей
Автор: Тань Лифэн
Дата: 3.11.2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ
# =============================================================================

def generate_signal(N=10000, M=20, train_ratio=0.8, noise_std=0.3):
    """
    Генерация синтетического сигнала с шумом
    
    Параметры:
    ----------
    N : int, количество отсчётов
    M : int, порядок фильтра
    train_ratio : float, доля обучающей выборки
    noise_std : float, стандартное отклонение шума
    
    Возвращает:
    -----------
    X_train, X_test, y_train, y_test : массивы данных
    x, s, v : исходные сигналы
    """
    
    # Генерация чистого сигнала (комбинация двух синусоид)
    n = np.arange(N)
    freq1, freq2 = 0.01, 0.03
    s = np.sin(2 * np.pi * freq1 * n) + 0.5 * np.sin(2 * np.pi * freq2 * n)
    
    # Добавление гауссова белого шума
    v = noise_std * np.random.randn(N)
    x = s + v  # Зашумлённый сигнал
    
    # Формирование обучающих пар (X, y)
    X_data = []
    y_data = []
    
    for i in range(M, N):
        # Входной вектор: последние M отсчётов
        X_data.append(x[i:i-M:-1])  # Обратный порядок для соответствия свёртке
        y_data.append(s[i])  # Целевой выход - чистый сигнал
    
    X_data = np.array(X_data)  # Форма: [N-M, M]
    y_data = np.array(y_data)  # Форма: [N-M]
    
    # Разделение на обучающую и тестовую выборки
    split_idx = int(len(X_data) * train_ratio)
    X_train, X_test = X_data[:split_idx], X_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
    return X_train, X_test, y_train, y_test, x, s, v

# =============================================================================
# 2. ОПРЕДЕЛЕНИЕ КЛАССОВ ДЛЯ РАБОТЫ С ДАННЫМИ
# =============================================================================

class SignalDataset(Dataset):
    """Пользовательский класс Dataset для сигналов"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =============================================================================
# 3. ОПРЕДЕЛЕНИЕ НЕЙРОСЕТЕВОЙ МОДЕЛИ
# =============================================================================

class LinearFilterNet(nn.Module):
    """Однослойная линейная нейронная сеть (эквивалент КИХ-фильтра)"""
    
    def __init__(self, input_dim):
        super(LinearFilterNet, self).__init__()
        # Линейный слой без смещения (bias=False) для соответствия КИХ-фильтру
        self.linear = nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x):
        return self.linear(x)

# =============================================================================
# 4. ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ И ОЦЕНКИ
# =============================================================================

def train_model(model, train_loader, test_loader, num_epochs=30, learning_rate=0.01):
    """
    Обучение нейросетевой модели
    
    Параметры:
    ----------
    model : нейросетевая модель
    train_loader, test_loader : DataLoader для обучения и тестирования
    num_epochs : количество эпох обучения
    learning_rate : скорость обучения
    
    Возвращает:
    -----------
    train_losses, test_losses : списки потерь по эпохам
    """
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # Режим обучения
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(inputs)
        
        # Средняя ошибка на эпоху
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Режим оценки на тестовой выборке
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                test_loss += loss.item() * len(inputs)
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        # Вывод прогресса каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            print(f"Эпоха [{epoch+1}/{num_epochs}], "
                  f"Ошибка обучения: {epoch_loss:.6f}, "
                  f"Ошибка тестирования: {test_loss:.6f}")
    
    return train_losses, test_losses

def apply_filter(weights, x, M):
    """
    Применение обученного фильтра к сигналу
    
    Параметры:
    ----------
    weights : коэффициенты фильтра
    x : входной сигнал
    M : порядок фильтра
    
    Возвращает:
    -----------
    y_hat : отфильтрованный сигнал
    """
    y_hat = np.zeros_like(x)
    for i in range(M, len(x)):
        y_hat[i] = np.dot(weights, x[i:i-M:-1])
    return y_hat

def calculate_metrics(y_hat, s, x, M):
    """
    Вычисление метрик качества фильтрации
    
    Параметры:
    ----------
    y_hat : отфильтрованный сигнал
    s : чистый сигнал
    x : зашумлённый сигнал
    M : порядок фильтра
    
    Возвращает:
    -----------
    mse : среднеквадратичная ошибка
    snr_improvement : улучшение отношения сигнал-шум (дБ)
    """
    # Используем только часть сигнала после установления фильтра
    start_idx = M
    mse = np.mean((y_hat[start_idx:] - s[start_idx:])**2)
    
    # Улучшение SNR
    original_noise_power = np.var(x[start_idx:] - s[start_idx:])
    filtered_noise_power = np.var(y_hat[start_idx:] - s[start_idx:])
    snr_improvement = 10 * np.log10(original_noise_power / filtered_noise_power)
    
    return mse, snr_improvement

def frequency_response(weights, num_points=512):
    """
    Вычисление частотной характеристики фильтра
    
    Параметры:
    ----------
    weights : коэффициенты фильтра
    num_points : количество точек для расчёта
    
    Возвращает:
    -----------
    freq_axis : массив частот
    H : комплексная частотная характеристика
    """
    freq_axis = np.linspace(0, np.pi, num_points)
    H = np.zeros(len(freq_axis), dtype=complex)
    
    for i, omega in enumerate(freq_axis):
        for k in range(len(weights)):
            H[i] += weights[k] * np.exp(-1j * omega * k)
    
    return freq_axis, H

def design_classical_fir(M, cutoff=0.05):
    """
    Проектирование классического КИХ-фильтра
    
    Параметры:
    ----------
    M : порядок фильтра
    cutoff : частота среза
    
    Возвращает:
    -----------
    fir_coeffs : коэффициенты классического фильтра
    """
    # Проектирование ФНЧ с частотой среза cutoff
    fir_coeffs = signal.firwin(M, cutoff, window='hamming')
    return fir_coeffs

# =============================================================================
# 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================

def plot_results(s, x, y_hat, M, train_losses, test_losses, weights, 
                 classical_weights=None, show_plots=True):
    """
    Построение графиков результатов
    
    Параметры:
    ----------
    s, x, y_hat : сигналы (чистый, зашумлённый, отфильтрованный)
    M : порядок фильтра
    train_losses, test_losses : ошибки обучения
    weights : коэффициенты нейросетевого фильтра
    classical_weights : коэффициенты классического фильтра
    show_plots : показывать графики
    """
    
    # 1. График сигналов
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    n_plot = min(500, len(s))  # Ограничиваем для наглядности
    n_range = np.arange(n_plot)
    plt.plot(n_range, s[:n_plot], 'g-', alpha=0.7, label='Чистый сигнал')
    plt.plot(n_range, x[:n_plot], 'r-', alpha=0.5, label='Зашумлённый сигнал')
    plt.plot(n_range, y_hat[:n_plot], 'b-', label='Отфильтрованный сигнал')
    plt.xlabel('Отсчёты')
    plt.ylabel('Амплитуда')
    plt.title('Сравнение сигналов')
    plt.legend()
    plt.grid(True)
    
    # 2. График обучения
    plt.subplot(2, 2, 2)
    plt.plot(train_losses, 'b-', label='Ошибка обучения')
    plt.plot(test_losses, 'r-', label='Ошибка тестирования')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.title('Кривая обучения')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 3. Частотная характеристика
    plt.subplot(2, 2, 3)
    freq_axis, H = frequency_response(weights)
    plt.plot(freq_axis, np.abs(H), 'b-', linewidth=2, label='Нейросетевой фильтр')
    
    if classical_weights is not None:
        _, H_classical = frequency_response(classical_weights)
        plt.plot(freq_axis, np.abs(H_classical), 'r--', linewidth=2, label='Классический фильтр')
    
    # Отметим частоты сигнала
    plt.axvline(x=2*np.pi*0.01, color='g', linestyle=':', alpha=0.7, label='Частота сигнала 1')
    plt.axvline(x=2*np.pi*0.03, color='g', linestyle='--', alpha=0.7, label='Частота сигнала 2')
    
    plt.xlabel('Нормированная частота (рад/отсчёт)')
    plt.ylabel('|H(ω)|')
    plt.title('Амплитудно-частотная характеристика')
    plt.legend()
    plt.grid(True)
    
    # 4. Импульсная характеристика
    plt.subplot(2, 2, 4)
    n_coeff = np.arange(M)
    plt.stem(n_coeff, weights, 'b-', markerfmt='bo', basefmt=' ', label='Нейросетевой фильтр')
    
    if classical_weights is not None:
        plt.stem(n_coeff, classical_weights, 'r--', markerfmt='rx', basefmt=' ', label='Классический фильтр')
    
    plt.xlabel('Коэффициент')
    plt.ylabel('Значение')
    plt.title('Импульсная характеристика')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if show_plots:
        plt.show()
    
    return plt.gcf()

# =============================================================================
# 6. ОСНОВНАЯ ПРОГРАММА
# =============================================================================

def main():
    """Основная функция эксперимента"""
    
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ ПО ПРОЕКТИРОВАНИЮ ЛИНЕЙНОГО ФИЛЬТРА")
    print("С ИСПОЛЬЗОВАНИЕМ НЕЙРОННЫХ СЕТЕЙ")
    print("=" * 60)
    
    # Параметры эксперимента
    N = 10000      # Общее количество отсчётов
    M = 20         # Порядок фильтра
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.01
    
    print(f"Параметры эксперимента:")
    print(f"- Количество отсчётов: {N}")
    print(f"- Порядок фильтра: {M}")
    print(f"- Размер пакета: {BATCH_SIZE}")
    print(f"- Количество эпох: {NUM_EPOCHS}")
    print(f"- Скорость обучения: {LEARNING_RATE}")
    print()
    
    # 1. Генерация данных
    print("1. Генерация синтетического сигнала...")
    X_train, X_test, y_train, y_test, x, s, v = generate_signal(N, M)
    
    # Преобразование в тензоры PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Создание DataLoader
    train_dataset = SignalDataset(X_train_t, y_train_t)
    test_dataset = SignalDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"   Размер обучающей выборки: {len(X_train)}")
    print(f"   Размер тестовой выборки: {len(X_test)}")
    
    # 2. Создание и обучение модели
    print("\n2. Создание и обучение нейросетевой модели...")
    model = LinearFilterNet(M)
    print(f"   Архитектура модели: {model}")
    
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE
    )
    
    # 3. Извлечение обученных коэффициентов
    weights = model.linear.weight.detach().numpy().flatten()
    print(f"\n3. Обучение завершено.")
    print(f"   Коэффициенты фильтра извлечены (форма: {weights.shape})")
    
    # 4. Применение фильтра и оценка качества
    print("\n4. Применение фильтра и оценка качества...")
    y_hat = apply_filter(weights, x, M)
    mse, snr_improvement = calculate_metrics(y_hat, s, x, M)
    
    print(f"   Среднеквадратичная ошибка (MSE): {mse:.6f}")
    print(f"   Улучшение отношения сигнал-шум: {snr_improvement:.2f} дБ")
    
    # 5. Сравнение с классическим методом
    print("\n5. Сравнение с классическим КИХ-фильтром...")
    classical_weights = design_classical_fir(M)
    y_hat_classical = apply_filter(classical_weights, x, M)
    mse_classical, snr_improvement_classical = calculate_metrics(y_hat_classical, s, x, M)
    
    print(f"   MSE классического фильтра: {mse_classical:.6f}")
    print(f"   Улучшение SNR классического фильтра: {snr_improvement_classical:.2f} дБ")
    print(f"   Относительное улучшение нейросетевого метода: "
          f"{(mse_classical - mse) / mse_classical * 100:.1f}%")
    
    # 6. Визуализация результатов
    print("\n6. Визуализация результатов...")
    fig = plot_results(s, x, y_hat, M, train_losses, test_losses, weights, classical_weights)
    
    # Сохранение графиков
    fig.savefig('filter_design_results.png', dpi=300, bbox_inches='tight')
    print("   Графики сохранены в файл 'filter_design_results.png'")
    
    # 7. Дополнительный анализ
    print("\n7. Дополнительный анализ...")
    
    # Анализ чувствительности к порядку фильтра
    orders = [10, 20, 30, 50]
    mse_results = []
    
    print("   Анализ чувствительности к порядку фильтра:")
    for order in orders:
        # Быстрое обучение для анализа
        temp_model = LinearFilterNet(order)
        
        # Генерация данных для текущего порядка
        X_train_temp, X_test_temp, y_train_temp, y_test_temp, _, _, _ = generate_signal(N, order)
        
        # Преобразование и создание DataLoader
        X_train_t_temp = torch.tensor(X_train_temp, dtype=torch.float32)
        y_train_t_temp = torch.tensor(y_train_temp, dtype=torch.float32)
        train_dataset_temp = SignalDataset(X_train_t_temp, y_train_t_temp)
        train_loader_temp = DataLoader(train_dataset_temp, batch_size=BATCH_SIZE, shuffle=True)
        
        # Краткое обучение (5 эпох)
        temp_optimizer = optim.Adam(temp_model.parameters(), lr=LEARNING_RATE)
        temp_criterion = nn.MSELoss()
        
        for _ in range(5):
            for inputs, targets in train_loader_temp:
                temp_optimizer.zero_grad()
                outputs = temp_model(inputs)
                loss = temp_criterion(outputs.squeeze(), targets)
                loss.backward()
                temp_optimizer.step()
        
        # Оценка качества
        temp_weights = temp_model.linear.weight.detach().numpy().flatten()
        y_hat_temp = apply_filter(temp_weights, x, order)
        mse_temp, _ = calculate_metrics(y_hat_temp, s, x, order)
        mse_results.append(mse_temp)
        
        print(f"     M={order}: MSE = {mse_temp:.6f}")
    
    # Поиск оптимального порядка
    optimal_order = orders[np.argmin(mse_results)]
    print(f"   Оптимальный порядок фильтра: M={optimal_order}")
    
    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ УСПЕШНО ЗАВЕРШЕН")
    print("=" * 60)
    
    return {
        'model': model,
        'weights': weights,
        'classical_weights': classical_weights,
        'mse': mse,
        'snr_improvement': snr_improvement,
        'mse_classical': mse_classical,
        'optimal_order': optimal_order,
        'sensitivity_analysis': dict(zip(orders, mse_results))
    }

# =============================================================================
# 7. ЗАПУСК ЭКСПЕРИМЕНТА
# =============================================================================

if __name__ == "__main__":
    # Запуск основного эксперимента
    results = main()
    
    # Дополнительно: демонстрация онлайн-фильтрации
    print("\nДЕМОНСТРАЦИЯ ОНЛАЙН-ФИЛЬТРАЦИИ")
    print("-" * 40)
    
    # Используем обученные коэффициенты для имитации онлайн-обработки
    weights = results['weights']
    M = len(weights)
    
    # Генерация нового короткого сигнала для демонстрации
    n_online = 200
    t_online = np.arange(n_online)
    s_online = np.sin(2 * np.pi * 0.02 * t_online)  # Чистый сигнал
    x_online = s_online + 0.3 * np.random.randn(n_online)  # Зашумлённый сигнал
    
    # Буфер для онлайн-обработки
    buffer = np.zeros(M)
    y_online = np.zeros(n_online)
    
    print("Обработка сигнала в реальном времени:")
    for i in range(n_online):
        # Обновление буфера (скользящее окно)
        if i < M:
            # Начальное заполнение буфера
            buffer[M-i-1] = x_online[i]
        else:
            # Сдвиг буфера и добавление нового отсчёта
            buffer = np.roll(buffer, -1)
            buffer[-1] = x_online[i]
        
        # Применение фильтра, когда буфер заполнен
        if i >= M-1:
            y_online[i] = np.dot(weights, buffer)
            
            # Вывод прогресса для первых нескольких отсчётов
            if i < M + 5:
                print(f"  Отсчёт {i}: вход={x_online[i]:.3f}, выход={y_online[i]:.3f}")
    
    print("...")
    print("Онлайн-фильтрация завершена.")
    
    # Визуализация онлайн-обработки
    plt.figure(figsize=(10, 6))
    plt.plot(t_online, s_online, 'g-', label='Чистый сигнал', linewidth=2)
    plt.plot(t_online, x_online, 'r-', alpha=0.5, label='Зашумлённый сигнал')
    plt.plot(t_online, y_online, 'b-', label='Онлайн-фильтрация')
    plt.axvline(x=M, color='k', linestyle='--', label='Начало фильтрации')
    plt.xlabel('Отсчёты')
    plt.ylabel('Амплитуда')
    plt.title('Демонстрация онлайн-фильтрации')
    plt.legend()
    plt.grid(True)
    plt.savefig('online_filtering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nГрафик онлайн-фильтрации сохранен в 'online_filtering.png'")
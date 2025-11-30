import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 参数配置
np.random.seed(42)
WINDOW_SIZE = 5       # 输入窗口大小
HIDDEN_UNITS = [64, 32] # 隐层神经元数量
LEARNING_RATE = 0.001 # 学习率
EPOCHS = 100          # 训练轮次
BATCH_SIZE = 64       # 批次大小
TEST_SPLIT = 0.2      # 测试集比例

# 信号生成函数
def generate_signal(length=10000):
    """生成复合正弦信号"""
    t = np.linspace(0, 10, length)
    signal = (
        np.sin(0.01 * t) +       # 主频分量
        0.5 * np.sin(0.03 * t) + # 次频分量
        0.2 * np.sin(0.05 * t)   # 高频分量
    )
    return signal

# 噪声生成函数
def generate_noise(signal, snr_db):
    """按SNR生成高斯白噪声"""
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    return noise

# 数据预处理
def create_dataset(signal, window_size):
    """创建滑动窗口数据集"""
    X, y = [], []
    for i in range(len(signal) - window_size):
        X.append(signal[i:i+window_size])
        y.append(signal[i+window_size])
    return np.array(X), np.array(y)

# 模型构建函数
def build_model(input_shape):
    """构建自适应神经网络模型"""
    model = Sequential([
        Dense(HIDDEN_UNITS[0], activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(HIDDEN_UNITS[1], activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    return model

# 动态噪声模拟
def simulate_dynamic_noise(signal, snr_list, split_point):
    """模拟动态变化的噪声环境"""
    noise1 = generate_noise(signal[:split_point], snr_list[0])
    noise2 = generate_noise(signal[split_point:], snr_list[1])
    noisy_signal = np.concatenate([signal[:split_point]+noise1, 
                                   signal[split_point:]+noise2])
    return noisy_signal

# 主程序
if __name__ == "__main__":
    # 生成原始信号
    original_signal = generate_signal()
    
    # 创建动态噪声环境（前50% SNR=20dB，后50% SNR=5dB）
    noisy_signal = simulate_dynamic_noise(
        original_signal, 
        snr_list=[20, 5],
        split_point=int(len(original_signal)*0.5)
    )
    
    # 创建数据集
    X, y = create_dataset(noisy_signal, WINDOW_SIZE)
    
    # 数据分割
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 创建模型
    model = build_model((WINDOW_SIZE,))
    
    # 训练监控
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        ]
    )
    
    # 模型评估
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print("\n训练集指标:")
    print(f"MSE: {mean_squared_error(y_train, train_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_train, train_pred):.4f}")
    
    print("\n测试集指标:")
    print(f"MSE: {mean_squared_error(y_test, test_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, test_pred):.4f}")
    
    # 结果可视化
    plt.figure(figsize=(15, 8))
    
    # 原始信号 vs 含噪信号
    plt.subplot(2, 2, 1)
    plt.plot(original_signal, label='原始信号')
    plt.plot(noisy_signal, label='含噪信号', alpha=0.7)
    plt.title('信号对比')
    plt.legend()
    
    # 训练过程可视化
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('训练过程')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    # 滤波效果对比
    plt.subplot(2, 2, 3)
    plt.plot(y_test, label='真实值', color='green')
    plt.plot(test_pred, label='预测值', color='blue', linestyle='--')
    plt.title('滤波效果对比')
    plt.legend()
    
    # 动态适应演示（后半段噪声增强）
    plt.subplot(2, 2, 4)
    plt.plot(noisy_signal[-2000:], label='输入噪声', alpha=0.7)
    plt.plot(test_pred[-2000:], label='输出信号', color='red')
    plt.title('动态噪声适应演示')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import onnxruntime as ort

# ---------------------------
# 1. Модуль генерации данных
# ---------------------------
def generate_motor_data(fs=5000, duration=20, seed=42):
    """Генерация данных с учетом нелинейностей"""
    np.random.seed(seed)
    
    # Параметры системы
    K = 12   # Коэффициент усиления (Нм/А)
    T = 0.02 # Время сходимости (с)
    B = 3.5e-4 # Коэффициент трения (Нм/(рад/с))
    J = 8e-4 # Момент инерции (кг·м²)
    
    # Нелинейная модель трения
    def friction_model(omega):
        return np.sign(omega) * 0.1  # Сухое трение
    
    # Дифференциальное уравнение системы
    def model(x, t, u):
        theta, omega = x
        dtheta_dt = omega
        domega_dt = (K*u - B*omega - friction_model(omega)) / J
        return [dtheta_dt, domega_dt]
    
    # Генерация входного сигнала (PRBS + синусоидальное возмущение)
    t = np.arange(0, duration, 1/fs)
    prbs = np.random.choice([-1, 1], size=len(t))  # PRBS-сигнал
    sine = 0.2 * np.sin(2 * np.pi * 0.5 * t)      # Синусоидальное возмущение
    u = prbs + sine
    
    # Начальные условия
    x0 = [0.1, 0]
    
    # Решение ДУ
    sol = odeint(model, x0, t, args=(u,))
    
    # Добавление шума измерения
    noise_theta = np.random.normal(0, 0.002, len(sol))
    noise_omega = np.random.normal(0, 0.002, len(sol))
    
    # Создание DataFrame
    df = pd.DataFrame({
        'time': t,
        'theta_ref': sol[:,0] + noise_theta,
        'omega_ref': sol[:,1] + noise_omega,
        'u': u,
        'theta_meas': sol[:,0],
        'omega_meas': sol[:,1]
    })
    
    return df

# Генерация и сохранение данных
df = generate_motor_data()
df.to_csv("motor_dataset.csv", index=False)

# ---------------------------
# 2. Модуль предобработки данных
# ---------------------------
def preprocess_data(df, window_size=10):
    """Стандартизация и создание последовательностей"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['omega_ref', 'u']])
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size][1])  # Целевая переменная
    
    return np.array(X), np.array(y), scaler

X, y, scaler = preprocess_data(df)

# ---------------------------
# 3. Модуль построения модели
# ---------------------------
def build_lstm_model(input_shape):
    """Построение би-направленной LSTM сети"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss='mse',
                  metrics=['mae'])
    return model

model = build_lstm_model((10, 2))
model.summary()

# ---------------------------
# 4. Модуль обучения модели
# ---------------------------
def train_model(model, X, y, epochs=200, batch_size=64):
    """Обучение с ранней остановкой"""
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop]
    )
    
    return model, history

model, history = train_model(model, X, y)

# Визуализация результатов обучения
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Обучение')
plt.plot(history.history['val_loss'], label='Валидация')
plt.title('Loss по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Обучение')
plt.plot(history.history['val_mae'], label='Валидация')
plt.title('MAE по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# 5. Модуль оценки модели
# ---------------------------
def evaluate_model(model, X, y):
    """Оценка модели"""
    y_pred = model.predict(X)
    rmse = np.sqrt(np.mean((y_pred - y)**2))
    mae = np.mean(np.abs(y_pred - y))
    
    print(f"Оценка модели:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

evaluate_model(model, X, y)

# ---------------------------
# 6. Модуль экспорта модели
# ---------------------------
def save_tflite_model(model, filename='observer.tflite'):
    """Экспорт в TFLite с квантованием"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    
    with open(filename, 'wb') as f:
        f.write(tflite_quant_model)
        
save_tflite_model(model)

# ---------------------------
# 7. Модуль развертывания
# ---------------------------
def deploy_on_jetson(model_path):
    """Развертывание на платформе Jetson"""
    # Инициализация ONNX Runtime
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # Тестирование задержки
    test_input = X[0:1].astype(np.float32)
    start_time = time.time()
    
    for _ in range(100):
        output = sess.run([output_name], {input_name: test_input})[0]
    
    elapsed_time = (time.time() - start_time) * 1e6 / 100  # Микросекунды
    print(f"Задержка вывода: {elapsed_time:.2f} мкс")

deploy_on_jetson('observer.tflite')
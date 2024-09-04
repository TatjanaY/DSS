import numpy as np


# Определим линейную функцию
def linear_function(x1, x2):
    return 2 * x1 + 3 * x2 + 1  # Например, y = 2*x1 + 3*x2 + 1


# Инициализация весов и смещения
weights = np.random.rand(2)  # Веса для x1 и x2
bias = np.random.rand(1)  # Смещение


# Функция прямого распространения
def forward_pass(X):
    return np.dot(X, weights) + bias


# Функция потерь (среднеквадратичная ошибка)
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Обратное распространение и обновление весов
def update_weights(X, y_true, y_pred, learning_rate):
    # объявление глобальных переменных
    global weights, bias
    # вычисление ошибки
    error = y_pred - y_true
    # вычисление локального градиента
    weights_gradient = np.dot(X.T, error) / len(X)
    bias_gradient = np.mean(error)

    # корректировка весов и биаса
    weights -= learning_rate * weights_gradient
    bias -= learning_rate * bias_gradient


# Генерация тренировочных данных
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_train = np.array([linear_function(x1, x2) for x1, x2 in X_train])

# Параметры обучения
epochs = 1000 # число эпок
learning_rate = 0.01 # шаг сходимости

# Обучение нейронной сети для каждой эпохи
for epoch in range(epochs):
    # Прямое распространение
    y_pred = forward_pass(X_train)

    # Вычисление потерь
    loss = compute_loss(y_train, y_pred)

    # Обратное распространение
    update_weights(X_train, y_train, y_pred, learning_rate)

    # вывод результатов для каждой 100-й эпохи
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.8f}, weights {weights}, bias{bias}')

# Проверка результата для произвольных данных
X_test = np.array([[6, 7], [7, 8]])
# получение результата с помощью нейронной сети
predictions = forward_pass(X_test)
# получение результата методом прямого вычисления значения функции
y_true=np.array([linear_function(x1, x2) for x1, x2 in X_test])
# вывод результатов вычислений
print('Predictions:', predictions,'y_true:', y_true)

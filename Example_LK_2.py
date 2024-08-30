# вспомогательный модуль для работы с векторами и матрицами
import numpy as np


# вспомогательная функция – функция активации, будет возвращать 0 если x меньше 0,5 и 1 во всех остальных случаях
def act(x):
    return 0 if x < 0.5 else 1


# функция, которая пропускает через нейронную сеть входной сигнал (house, rock, attr) и отображает на выходе результат формирования или отсутствия симпатии
def go(house, rock, attr):
    # формирования вектора входного сигнала (они могут принимать значения 1 или 0)
    x = np.array([house, rock, attr])
    # веса для 1-го нейрона скрытого слоя
    w11 = [0.3, 0.3, 0]
    # веса для 2-го нейрона скрытого слоя
    w12 = [0.4, -0.5, 1]
    # объединение весов в матрицу (2 нейрона скрытого слоя, у каждого нейрона скрытого слоя по 3 входных связи)
    weight1 = np.array([w11, w12])  # матрица 2x3
    # формирование вектора связи для выходного нейрона
    weight2 = np.array([-1, 1])  # вектор 1х2

    sum_hidden = np.dot(weight1,
                        x)  # вычисляем сумму на входах нейронов скры-того слоя – вектор, состоящий из двух компонент и определяющий сумму на каждом нейроне скрытого слоя
    print("Значения сумм на нейронах скрытого слоя: " + str(sum_hidden))

    # вектор сумм пропускаем через функцию активации и на выходе получаем соответ-ственно вектор – выходные значения из каждого нейрона скрытого слоя
    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: " + str(out_hidden))

    # вычисляем сумму на выходном нейроне последнего слоя: веса умножаем на выходные значения нейронов скрытого слоя
    sum_end = np.dot(weight2, out_hidden)
    # пропускаем сумму через функцию активации и получаем результат работы НС
    y = act(sum_end)
    print("Выходное значение НС: " + str(y))

    return y


# определяем входные параметры
house = 1  # у парня есть квартира
rock = 0  # парень не любит тяжелый рок
attr = 1  # парень привлекателен

# пропускаем входные сигналы через НС
res = go(house, rock, attr)
# анализируем результат работы НС
if res == 1:
    print("Ты мне нравишься")
else:
    print("Созвонимся")
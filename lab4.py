# 22.Формируется матрица F следующим образом: скопировать в нее А и  если в Е количество чисел,
# больших К в четных столбцах больше, чем произведение чисел в нечетных строках , то поменять местами С и Е симметрично,
# иначе С и В поменять местами несимметрично. При этом матрица А не меняется.
# После чего если определитель матрицы А больше суммы диагональных элементов матрицы F,
# то вычисляется выражение: A*AT – K * F-1, иначе вычисляется выражение (A-1 +G-FТ)*K,
# где G-нижняя треугольная матрица, полученная из А. Выводятся по мере формирования А,
# F и все матричные операции последовательно.

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


k = int(input("Введите число K являющееся коэффициентом при умножении: "))
n = int(input("Введите число n больше 3 которое являеться рзмером матрицы: "))
while n <= 3:
    n = int(input("\nВведите число больше 3 "))
A = np.random.randint(-10.0, 11.0, (n, n))
# A = np.ones((n,n))      # или задание единичной матрицы для тестирования
print("Матрица A:\n", A)


half_n = n // 2
maxfix_n = half_n
minfix_n = half_n
if n % 2 != 0:
    maxfix_n += 1
    minfix_n = maxfix_n - 1


F = A.copy()
E = np.array(A[:minfix_n, :minfix_n])
print('Подматрица E матрицы A:\n', E)
B = np.array(A[:minfix_n, maxfix_n:])
print('Подматрица B матрицы A:\n', B)
C = np.array(A[maxfix_n:, maxfix_n:])
print('Подматрица C матрицы A:\n', C)
D = np.array(A[maxfix_n:, :minfix_n])
print('Подматрица D матрицы A:\n', D)


morek_E = (E[:, 1::2] > k).sum()
sum_E = np.prod(E[0::2, :])

if morek_E < sum_E:
    print("Меняем симметрично B и C")
    F[:minfix_n, :minfix_n] = C[-1::-1, :minfix_n]
    F[maxfix_n:, maxfix_n:] = E[-1::-1, :minfix_n]
else:
    print("Меняем несимметрично B и Е")
    F[:minfix_n, maxfix_n:] = C
    F[maxfix_n:, maxfix_n:] = B
print(F)
trans_A = np.transpose(A)
trans_F = np.transpose(F)
det_A = np.linalg.det(A)
diag_F = np.trace(F)

if det_A > diag_F:
    print('Вычисляем выражение : A * AT – K * F-1 ')
    print('Транспонированая матрица A:\n', trans_A)

    mod_A = np.dot(A, trans_A)
    print('Умножение A * AT\n', mod_A)

    power_F = np.linalg.matrix_power(F, -1)
    print('Возведение матрицы F в -1 степень:\n', power_F)

    mod_power_F = np.dot(k, power_F)
    print('Умножение K *FT\n', mod_power_F)

    result = np.subtract(mod_A, mod_power_F)
    print('Разница матриц\n', result)
else:
    print('Вычисляем выражение:(A-1 + G-FТ)*K ')

    power_A = np.linalg.matrix_power(A, -1)
    print('Возведение матрицы A в -1 степень:\n', power_A)

    print('Транспонированая матрица F:\n', trans_F)

    G = np.tril(A)
    print('Нижняя треугольная матрица G из матрицы A:\n', G)

    pAG = np.add(power_A, G)
    print('Сумма A-1 + G:\n', pAG)

    pAGFT = np.subtract(pAG, trans_F)
    print('Разница A-1 + G - FT:\n', pAGFT)

    result = np.dot(pAGFT, k)
    print('Умножение на K\n', result)

print('Результат вычислений\n', result)


explode = [0] * (n - 1)
explode.append(0.1)
plt.title("Круговая диаграмма")
try:
    sizes = [round(np.mean(abs(F[i, ::])) * 100, 1) for i in range(n)]
except IndexError:
    sizes = [round(np.mean(abs(F[i, ::])) * 100, 1) for i in range(n)]
plt.pie(sizes, labels=list(range(1, n + 1)), explode=explode, autopct='%1.1f%%', shadow=True)
plt.show()

plt.plot(A)
plt.title("График")
plt.ylabel("y axis")
plt.xlabel("x axis")
plt.show()

sns.heatmap(A, cmap="Spectral", annot=True)
plt.title("Тепловая карта")
plt.ylabel("Номер строки")
plt.xlabel("Номер столбца")
plt.show()

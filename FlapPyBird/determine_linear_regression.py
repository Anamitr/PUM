import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import linear_model

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

FILE_NAME = '3'
IN_FILE_PATH = './data/' + FILE_NAME + '.csv'
OUT_FILE_PATH = './data/' + FILE_NAME + 'out.csv'


def linear_regression(X, Y):
    LinReg = linear_model.LinearRegression()
    LinReg.fit(X.reshape(-1, 1), Y)
    Y_Pred = LinReg.predict(X.reshape(-1, 1))
    #
    plt.figure()
    plt.scatter(X, Y)
    plt.plot(X, Y_Pred, color='red')
    plt.title('Linear regression')
    plt.show()


def save_to_csv(X, Y):
    if len(X) != len(Y):
        raise Exception('len(X) != len(Y) !')
    with open(OUT_FILE_PATH, 'w', newline='') as csv_file:
        employee_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(X)):
            employee_writer.writerow([X.item(i), Y.item(i)])


def polynomial_regression(Xin, Yin):
    colors = ['teal', 'yellowgreen', 'gold']
    first_degree = 12  # 9
    last_degree = 13  # 13
    degree_step = 1
    colors = iter(cm.rainbow(np.linspace(0, 1,
                                         30)))

    plt.scatter(Xin, Yin)

    for degree in range(first_degree, last_degree, degree_step):
        lastIntX = int(Xin[-1])
        Xout = np.arange(lastIntX)
        # degree = 12
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(Xin.reshape(-1, 1), Yin)
        Yout = model.predict(Xout.reshape(-1, 1))
        # plt.plot(Xout, Yout, 'ro')
        plt.plot(Xout, Yout,
                 # color=('red'),
                 # linewidth=2,
                 label="degree %d" % degree)

    plt.legend(loc='lower left')
    plt.show()
    save_to_csv(Xout, Yout)

Etykiety = []
Dane = []

# wczytanie danych z pliku
with open(IN_FILE_PATH, mode='r') as plik_csv:
    csv_Rdr = csv.reader(plik_csv, delimiter=',')
    # wczytanie pierwszego wiersza
    Etykiety = next(csv_Rdr)
    for wiersz in csv_Rdr:
        Dane.append(wiersz)

print(Dane)
X = np.empty([1])
Y = np.empty([1])
for d in Dane:
    X = np.append(X, float(d[0]))
    Y = np.append(Y, float(d[1]))

print('X = ', X)
print('Y = ', Y)

# linear_regression(X, Y)
polynomial_regression(X, Y)

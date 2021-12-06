import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from math import e

def cut_tail(x):
    # consecutive days with new infection <= 2
    for i in range(len(x)-1, -1, -1):
        if x[i] - x[i-1] <= 2:
            continue
        return x[:i+1]

def read_explode_data(path, plot=False):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    data = []
    for key, val in res.items():
        for i in range(len(val)):
            # data.append((key, val[i][0], np.array(val[i][1]+[val[i][1][-1]]*20)))
            data.append((key, val[i][0], val[i][1]))
            # data.append((key, val[i][0], cut_tail(val[i][1])))
    if plot:
        for name, t, arr in data:
            plt.plot(pd.to_datetime(t), arr, c='b')
            plt.title(name)
            plt.show()
            print(f'{name}: start date {t[0]}, end date {t[-1]}')
    return data

def calculate_a(l1, l2, arr):
    t = np.arange(len(arr))
    et1 = np.exp(l1*t)
    et2 = np.exp(l2*t)
    x1 = np.sum((et1 - 1)**2)
    y1 = np.sum((et1 - 1)*(et2 - 1))
    z1 = np.sum((et1 - 1)*arr)
    x2 = np.sum((et1 - 1)*(et2 - 1))
    y2 = np.sum((et2 - 1)**2)
    z2 = np.sum((et2 - 1)*arr)
    a1 = (z1*y2 - z2*y1) / (x1*y2 - x2*y1)
    a2 = (z1*x2 - z2*x1) / (y1*x2 - y2*x1)
    return a1, a2

def calculate_loss(l1, l2, arr):
    a1, a2 = calculate_a(l1, l2, arr)
    # if np.isnan(a1): import pdb; pdb.set_trace()
    t = np.arange(len(arr))
    et1 = np.exp(l1*t)
    et2 = np.exp(l2*t)
    err = a1*et1 + a2*et2 - a1 - a2 - arr
    return np.mean(err**2)

def fit(arr, method, plot=False):
    if method == 'sa':
        # simulated annealing
        beta = 0.999
        t = 1
        min_t = 0.001
        l1, l2 = 0.1, -0.1
        loss = calculate_loss(l1, l2, arr)
        min_loss = loss
        min_l1, min_l2 = l1, l2
        restart = 3
        for i in range(restart):
            cnt = 0
            l1, l2 = min_l1, min_l2
            loss = calculate_loss(l1, l2, arr)
            t = 1
            while t >= min_t:
                new_l1 = l1 + np.random.normal()
                new_l2 = l2 + np.random.normal()
                new_loss = calculate_loss(new_l1, new_l2, arr)
                diff = new_loss - loss
                if diff < 0:
                    l1, l2 = new_l1, new_l2
                    loss = new_loss
                else:
                    if np.random.random() < e**(-diff/t):
                        l1, l2 = new_l1, new_l2
                        loss = new_loss
                t *= beta
                if loss < min_loss:
                    min_loss = loss
                    min_l1, min_l2 = l1, l2
                cnt += 1
                if cnt % 100 == 0:
                    print(f'Round {i+1}, Iteration {cnt:6d}, current best lambdas: {min_l1}, {min_l2}, loss {min_loss}')
        a1, a2 = calculate_a(min_l1, min_l2, arr)
        print(f'Iteration {cnt:6d} best lambdas: {min_l1}, {min_l2},'
              f' a1 = {a1}, a2 = {a2}, loss {min_loss}')
        if plot:
            t = np.arange(len(arr))
            plt.plot(t, arr, label='Target')
            et1 = np.exp(min_l1*t)
            et2 = np.exp(min_l2*t)
            plt.plot(t, a1*et1 + a2*et2 - a1 - a2, label='Fit')
            plt.legend()
            plt.show()
        return a1, a2, min_l1, min_l2
                    

def search_lambda(arr, lwr=-0.5, upr=0.5, d=0.01, plot=False):
    ls = np.arange(lwr, upr, d)
    t = np.arange(len(arr))
    min_loss = 1e10
    min_l, min_a = None, None
    for l in ls:
        et = np.exp(l*t) - 1
        a = (np.sum(et*arr)) / (np.sum(et*et))
        err = a * et - arr
        loss = np.mean(err**2)
        if loss < min_loss:
            min_loss = loss
            min_l = l
            min_a = a
    r_sq = 1 - min_loss / np.mean((arr - np.mean(arr))**2)
    print(f'Found lambda = {min_l}, a = {min_a}, loss {min_loss}, R sq {r_sq}')
    if plot:
        plt.plot(t, arr, label='Target')
        plt.plot(t, min_a * (np.exp(min_l*t) - 1), label='Fit')
        plt.legend()
        plt.show()
    return min_l, min_a

def analyze(arr, method='sa', tie_l=False):
    base = arr[0]
    arr = arr[1:] - base
    i0 = arr[0]
    if tie_l:
        l, a = search_lambda(arr, plot=False)
        e0 = l*a
        l1 = l2 = l
    else:
        a1, a2, l1, l2 = fit(arr, method=method, plot=True)
        e0 = l1*a1 + l2*a2
    return (l1, l2, e0, i0)

def fit_proportion(x, y):
    # y = kx
    k = np.sum(x*y) / np.sum(x*x)
    err = y - k*x
    r_sq = 1 - (np.sum(err**2) / np.sum((y-np.mean(y))**2))
    print('Best k = {}, R sq = {}'.format(k, r_sq))
    return k

def fit_time(i0, t):
    x = np.log(i0)
    xm = np.mean(x)
    tm = np.mean(t)
    a = (np.sum((x - xm)*(t - tm))) / (np.sum((x - xm)**2))
    b = tm - a*xm
    err = t - a*x - b
    r_sq = 1 - np.sum(err**2) / np.sum((t - tm)**2)
    print(f'Best param: a = {a}, b = {b}, R sq = {r_sq}')
    return a, b

def calculate_days(arr):
    # remove tail (consecutive days with new infection <= 3)
    n = len(arr)
    return n
    # inc = np.zeros(n)
    # arr = np.array(arr)
    # inc[1:] = arr[1:] - arr[:-1]
    # for i in range(n):
    #     if inc[n-1-i] <= 2:
    #         continue
    #     plt.plot(np.arange(n), arr)
    #     plt.plot(np.arange(n-i), arr[:n-i])
    #     plt.show()
    #     return n - i
    

if __name__ == '__main__':
    data = read_explode_data('data/processed-explode.txt', plot=False)
    for i in range(len(data)):
        # (name, dates, arr) -> (names, dates, arr, lambda1, lambda2, e0, i0)
        print(f'Processing {data[i][0]}')
        data[i] += analyze(data[i][2], tie_l=True)
    names, _, arr, l1, l2, e0, i0 = zip(*data)
    plt.scatter(i0, e0)
    k = fit_proportion(np.array(i0), np.array(e0))
    t = np.linspace(min(i0), max(i0), 3)
    plt.plot(t, k*t, '--', label='Fit')
    plt.xlabel('$i_0$')
    plt.ylabel('$\kappa e_0$')
    plt.legend()
    plt.show()
    t = np.array(list(map(calculate_days, arr)))
    a, b = fit_time(i0, t)
    plt.scatter(np.log(i0), t)
    plt.plot(np.log(i0), a*np.log(i0)+b, '--', label='Fit')
    plt.xlabel('$\ln i_0$')
    plt.ylabel('$days$')
    plt.legend()
    plt.show()

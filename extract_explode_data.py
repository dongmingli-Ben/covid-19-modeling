from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def check_explode(arr):
    if len(arr) < 5 or max(arr) < 10:
        return False
    # consecutive 5 days with new confirmed > 10
    cnt = 0
    for i in range(len(arr)):
        if arr[i] > 10:
            cnt += 1
        else:
            cnt = 0
        if cnt >= 5: return True
    return False

def extract_explode(t_arr: np.ndarray, c_arr: np.ndarray) -> List[Tuple[List[str], List[int]]]:
    """```t_arr```: array of dates, ```c_arr```: array of cumulative confirmed"""
    inc = np.zeros(len(t_arr))
    inc[1:] = c_arr[1:] - c_arr[:-1]
    periods = []
    start, end = None, None
    # tolerance of zeros in the series: 14
    cnt = 0
    for i in range(len(t_arr)):
        if inc[i] > 0 and start is None:
            start = i
        if inc[i] == 0 and start is not None:
            end = i-1
            cnt += 1
            if cnt > 14 and check_explode(inc[start:end+1]):
                periods.append((t_arr[start-1:end+1].tolist(), c_arr[start-1:end+1].tolist()))
            if cnt > 14:
                start = None
                cnt = 0
        if inc[i] > 10 and start is not None and cnt > 0:
            cnt = 0
    return periods

def extract(path, save_path):
    df = pd.read_csv(path)
    df = df[df['Country/Region'] == 'China'].reset_index(drop=True)
    dates = df.columns[4:].to_numpy()
    res = {}
    for i in range(df.shape[0]):
        item = df.iloc[i]
        name = item['Province/State']
        arr = item[dates.tolist()].to_numpy()
        periods = extract_explode(dates, arr)
        # plot and choose
        res[name] = []
        for (t, c_arr) in periods:
            plt.plot(pd.to_datetime(dates), arr, c='b')
            plt.plot(pd.to_datetime(t), c_arr, c='r')
            plt.title(name)
            plt.show()
            choice = input('Accept? [y]/n: ')
            if choice != 'n':
                res[name].append((t, c_arr))
    with open(save_path, 'wb') as f:
        pickle.dump(res, f)
    return res

def remove_head_and_tail(t, arr):
    c = np.array(arr)
    inc = np.zeros(len(arr))
    inc[1:] = c[1:] - c[:-1]
    # threshold for first spike 2x
    for i in range(1, len(inc)):
        if inc[i] > 2*inc[i-1] and inc[i] >=5 and inc[i+1] >= inc[i]*0.9:
            start = i-1
            break
    # do not remove tail currently
    return t[start:], arr[start:]

def post_process(path, save_path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    for key, val in res.items():
        for i in range(len(val)):
            t, arr = val[i]
            plt.plot(pd.to_datetime(t), arr, c='b')
            t, arr = remove_head_and_tail(t, arr)
            plt.plot(pd.to_datetime(t), arr, c='r')
            plt.title(key)
            plt.show()
            val[i] = (t, arr)
    with open(save_path, 'wb') as f:
        pickle.dump(res, f)

# extract('data/time_series_covid19_confirmed_global.csv', 'data/filtered-explode.txt')
post_process('data/filtered-explode.txt', 'data/processed-explode.txt')

from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simulator import DTMCSEIR, CTMCSEIR, ApproxSEIR, ExactSEIR, Model
from copy import deepcopy
from multiprocessing import Pool
import pickle
import os
from datetime import datetime
from math import e

def read_data(path, country_name):
    df = pd.read_csv(path)
    df = df[df['Country/Region'] == country_name]
    df = df[df.columns[4:]]
    return pd.to_datetime(df.columns).to_numpy(), df.to_numpy().squeeze()

def read_calibrate_data(country_name, start_time=None, end_time=None):
    t, confirmed = read_data('data/current_confirmed.csv', country_name)
    t, recovered = read_data('data/cumulative-recovered.csv', country_name)
    t, deaths = read_data('data/cumulative-deaths.csv', country_name)
    t, cum_conf = read_data('data/cumulative-confirmed.csv', country_name)
    # import pdb; pdb.set_trace()
    if start_time is not None:
        index = t.astype(str) >= f'{start_time}T00:00:00.000000000'
        t = t[index]
        confirmed = confirmed[index]
        recovered = recovered[index]
        deaths = deaths[index]
        cum_conf = cum_conf[index]
    if end_time is not None:
        index = t.astype(str) <= f'{end_time}T00:00:00.000000000'
        t = t[index]
        confirmed = confirmed[index]
        recovered = recovered[index]
        deaths = deaths[index]
        cum_conf = cum_conf[index]
    return t, confirmed, recovered, deaths, cum_conf

def get_all_settings(param_ranges: Dict[str, List[float]], init=None) -> List[dict]:
    if param_ranges == {}: return init
    if init is None: init = []
    param_ranges = deepcopy(param_ranges)
    key, ranges = param_ranges.popitem()
    new = []
    if init == []:
        for value in ranges:
            new.append({key: value})
    else:
        for value in ranges:
            for setting in init:
                s = deepcopy(setting)
                s[key] = value
                new.append(s)
    return get_all_settings(param_ranges, new)

class LossFunc:
    
    def __init__(self, model: Model, state, conf, rec, dea, n_days,
                 wi=0.2, wr=0.3, wd=0.5):
        self.model = model
        self.model.reset(state=state, keep_init=False)
        self.state = state
        self.n = n_days - 1
        self.wi = wi
        self.wr = wr
        self.wd = wd
        self.conf = conf
        self.rec = rec
        self.deaths = dea
        
    def __call__(self, param, steps=100):
        self.model.reset(parameters=param, state=self.state, keep_init=False)
        for i in range(self.n):
            if isinstance(self.model, CTMCSEIR) or isinstance(self.model, DTMCSEIR):
                self.model.step()
            else:
                self.model.step(steps=steps)
        if isinstance(self.model, ApproxSEIR):
            e, i, r, p, cr = zip(*self.model.history)
        else:
            s, e, i, r, p, cr = zip(*self.model.history)
        # import pdb; pdb.set_trace()
        mse_i = np.mean((i-self.conf)**2)
        mse_r = np.mean((cr-self.rec)**2)
        mse_d = np.mean((p-self.deaths)**2)
        loss = mse_i*self.wi + mse_r*self.wr + mse_d*self.wd
        return loss

def get_scores(model: Model, settings, data, parellel):
    t, confirmed, recovered, deaths, cum_conf = data
    init_state = {'e': confirmed[0],
                  'i': confirmed[0],
                  'r': recovered[0],
                  'p': deaths[0],
                  'cr': recovered[0]}
    days = len(t)
    loss = LossFunc(model, init_state, confirmed, recovered, deaths, days)
    if parellel:
        pool = Pool(4)
        scores = list(pool.map(loss, settings))
    else:
        scores = list(map(loss, settings))
    scores = np.array(scores)
    return scores

def plot_model_and_data(model: Model, data, param={}):
    fig, ax = plt.subplots()
    t, confirmed, recovered, deaths, cum_conf = data
    ax.plot(t, confirmed, label='Confirmed')
    ax.plot(t, recovered, label='Cum Recovered')
    ax.plot(t, deaths, label='Deaths')
    # setup model
    n = len(t) - 1
    init_state = {'i': confirmed[0],
                  'r': recovered[0],
                  'p': deaths[0],
                  'cr': recovered[0]}
    model.reset(state=init_state, parameters=param, keep_init=False)
    for i in range(n):
        model.step()
    if isinstance(model, ApproxSEIR):
        e, i, r, p, cr = zip(*model.history)
    else:
        s, e, i, r, p, cr = zip(*model.history)
    ax.plot(t, e, linestyle='--', label='Exposed')
    ax.plot(t, i, linestyle='--', label='Infected')
    ax.plot(t, r, linestyle='--', label='Recovered')
    ax.plot(t, p, linestyle='--', label='Deaths')
    ax.plot(t, cr, linestyle='--', label='Cum Recovered')
    ax.legend()
    plt.show()
    
def save_data(settings, scores, path):
    directory = '/'.join(path.split('/')[:-1]) if '/' in path else None
    if directory:
        if not os.path.exists(directory): os.makedirs(directory)
    with open(path, 'wb') as f:
        pickle.dump(settings, f)
        pickle.dump(scores, f)
        
def load_result(path):
    with open(path, 'rb') as f:
        t = pickle.load(f)
        scores = pickle.load(f)
    return t, scores

def calibrate(model: Model, parameters_range, data, parellel=True, plot=True):
    settings = get_all_settings(parameters_range)
    print(f'Get total {len(settings)} settings!')
    # import pdb; pdb.set_trace()
    print('Start calibrating ...')
    scores = get_scores(model, settings, data, parellel)
    save_data(settings, scores, f'results/{str(datetime.now())}.txt'.replace(':', '.'))
    # settings, scores = load_result('results/2021-11-14 10.24.45.886660.txt')
    index = ~np.isnan(scores)
    settings = np.array(settings)[index]
    scores = scores[index]
    index = np.argmin(scores)
    best_param = settings[index]
    print(f'Find param {best_param} with loss {scores[index]}')
    if plot:
        model.reset(parameters=best_param)
        plot_model_and_data(model, data)
    return best_param

def simulated_annealing(model: Model, init_param: dict, data, beta=0.999, init_t=1, stop_t=0.001,
                        early_stopping=3e3, verbose=True):
    t, confirmed, recovered, deaths, cum_conf = data
    init_state = {'i': confirmed[0],
                  'r': recovered[0],
                  'p': deaths[0],
                  'cr': recovered[0]}
    loss_func = LossFunc(model, init_state, confirmed, recovered, deaths, len(t))
    min_loss = loss_func(init_param)
    min_param = init_param
    # restart = 3
    total_nan = 0
    patient_cnt = 0
    i = 0
    stop = False
    while not stop:
        cnt = 0
        param = min_param.copy()
        loss = loss_func(param)
        t = init_t
        while t >= stop_t:
            new_param = {}
            for key, val in param.items():
                if key == 's':
                    new_param[key] = max(val + np.random.normal()*1e6, 0)
                elif (key == 'ae' or key == 'ai') and not isinstance(model, ApproxSEIR):
                    new_param[key] = max(val + np.random.normal()*1e-8, 0)
                elif key == 'max_e' or key == 'max_i' or key == 'e':
                    new_param[key] = max(val + np.random.normal()*1e3, 0)
                else:
                    new_param[key] = max(val + np.random.normal()*1e-3, 0)
            steps = 100
            new_loss = loss_func(new_param, steps)
            while np.isnan(new_loss) and steps <= 1e4:
                # import pdb; pdb.set_trace()
                steps *= 2
                new_loss = loss_func(new_param, steps)
            # new_loss = loss_func(new_param)
            diff = new_loss - loss
            if np.isnan(diff): total_nan += 1
            if diff < 0 or (np.isnan(loss) and not np.isnan(new_loss)):
                param = new_param
                loss = new_loss
            else:
                if np.random.random() < e**(-diff/t):
                    param = new_param
                    loss = new_loss
            t *= beta
            patient_cnt += 1
            cnt += 1
            if loss < min_loss:
                min_loss = loss
                min_param = param
                patient_cnt = 0
            if patient_cnt > early_stopping:
                stop = True
                break
            if cnt % 100 == 0 and verbose:
                print(f'Round {i+1}, Iteration {cnt:6d}, total nan {total_nan}, current best lambdas: {min_param}, loss {min_loss}')
        i += 1
    return min_param, min_loss

def calibrate_sa(model: Model, init_parameters, data, plot=True):
    print('Start calibrating ...')
    param, loss = simulated_annealing(model, init_parameters, data)
    print(f'Find param {param} with loss {loss}')
    if plot:
        plot_model_and_data(model, data, param)
    return param

if __name__ == '__main__':
    data = read_calibrate_data('India', end_time='2021-02-28', start_time='2020-07-01')
    model = DTMCSEIR(
        {'ae': 8.712675265945978e-07, 
         'ai': 1.1960480100482177e-06, 
         'gamma': 0, 
         'kappa': 0, 
         'beta': 0, 
         'rho': 0.07931972326570251, 
         'mu': 0.05042566344153261, 
         's': 993990.4996814968
         }, {})
    # plot_model_and_data(model, data)
    # parameters_range = {
    #     'ae': np.linspace(5, 100, 5),
    #     'ai': np.linspace(5, 100, 5),
    #     'gamma': np.linspace(0, 0.03, 5),
    #     'kappa': np.linspace(0.05, 0.2, 5),
    #     'beta': np.linspace(0.01, 0.3, 5),
    #     'rho': np.linspace(0.005, 0.1, 5),
    #     'mu': np.linspace(0.005, 0.05, 5),
    # }
    # calibrate(model, parameters_range, data, parellel=False)
    init_param = {
        'ae': 1e-8,
        'ai': 1e-8,
        'gamma': 0.01,
        'kappa': 0.1,
        'beta': 0.1,
        'rho': 0.01,
        'mu': 0.01,
        's': 5e7,
        'e': 1e3,
        'max_e': 1e3,
        'max_i': 1e3
    }
    calibrate_sa(model, init_param, data)
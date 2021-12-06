import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simulator import DTMCSEIR, CTMCSEIR, ApproxSEIR, ExactSEIR, Model
from calibrate_dtmc import simulated_annealing

def calculate_loss_normalize(history, conf, rec, deaths, plot=True):
    s, e, i, r, p, cr = zip(*history)
    mse_i = np.mean((i-conf)**2)
    mse_r = np.mean((cr-rec)**2)
    mse_d = np.mean((p-deaths)**2)
    r_i = mse_i**0.5 / np.max(conf)
    r_r = mse_r**0.5 / np.max(rec)
    r_d = mse_d**0.5 / np.max(deaths)
    loss = mse_i*.2 + mse_r*.3 + mse_d*.5
    loss_normal = r_i*.2 + r_r*.3 + r_d*.5
    print(f'Normalized loss: infected {r_i}, recovered {r_r}, deaths {r_d}')
    if plot:
        plt.plot(i, '--', label='i')
        plt.plot(cr, '--', label='cr')
        plt.plot(p, '--', label='p')
        plt.plot(conf, label='active')
        plt.plot(rec, label='recovered')
        plt.plot(deaths, label='deaths')
        plt.legend()
        plt.show()
    return loss, mse_i, mse_r, mse_d, loss_normal, r_i, r_r, r_d

def fit_predict_evaluate(name, model: Model, calibrate_data, eval_data, init_param={}):
    param, loss = simulated_annealing(model, init_param, calibrate_data, verbose=False)
    t, confirmed, recovered, deaths, cum_conf = calibrate_data
    init_state = {'i': confirmed[0],
                  'r': recovered[0],
                  'p': deaths[0],
                  'cr': recovered[0]}
    model.reset(param, init_state, clear_history=True, keep_init=False)
    for _ in range(len(t)-1):
        model.step()
    l, lc, lr, lp, norm, nc, nr, np = calculate_loss_normalize(model.history, confirmed, recovered, deaths)
    t, confirmed, recovered, deaths, cum_conf = eval_data
    init_state = {'s': model.history[-1][0],
                  's': model.history[-1][1],
                  'i': confirmed[0],
                  'r': recovered[0],
                  'p': deaths[0],
                  'cr': recovered[0]}
    best_param = param.copy()
    param.pop('s')
    param.pop('e')
    model.reset(param, init_state, clear_history=True, keep_init=False)
    for _ in range(len(t)-1):
        model.step()
    eval_l, eval_lc, eval_lr, eval_lp, eval_norm, eval_nc, eval_nr, eval_np = \
        calculate_loss_normalize(model.history, confirmed, recovered, deaths)
    print(f'{name} best param {best_param}. Fitting: Loss {l}, Normalized Loss {norm}.'
          f' Eval: Loss {eval_l}, Normalized Loss {eval_norm}')
    return [l, lc, lr, lp, norm, nc, nr, np,
            eval_l, eval_lc, eval_lr, eval_lp, eval_norm, eval_nc, eval_nr, eval_np]

def read_data(path, index):
    df = pd.read_csv(path)
    df = df[df.columns[4:]]
    data = df.iloc[index]
    return pd.to_datetime(df.columns).to_numpy(), data.to_numpy().squeeze()

def read_calibrate_data(start_time, end_time, split_date, index):
    t, confirmed = read_data('data/current_confirmed.csv', index)
    t, recovered = read_data('data/cumulative-recovered.csv', index)
    t, deaths = read_data('data/cumulative-deaths.csv', index)
    t, cum_conf = read_data('data/cumulative-confirmed.csv', index)
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
    if split_date is not None:
        index = t.astype(str) <= f'{split_date}T00:00:00.000000000'
        calibrate_t = t[index]
        calibrate_confirmed = confirmed[index]
        calibrate_recovered = recovered[index]
        calibrate_deaths = deaths[index]
        calibrate_cum_conf = cum_conf[index]
        index = t.astype(str) >= f'{split_date}T00:00:00.000000000'
        eval_t = t[index]
        eval_confirmed = confirmed[index]
        eval_recovered = recovered[index]
        eval_deaths = deaths[index]
        eval_cum_conf = cum_conf[index]
    calibrate_data = (
        calibrate_t,
        calibrate_confirmed,
        calibrate_recovered,
        calibrate_deaths,
        calibrate_cum_conf
    )
    eval_data = (
        eval_t,
        eval_confirmed,
        eval_recovered,
        eval_deaths,
        eval_cum_conf
    )
    return calibrate_data, eval_data

def evaluate(start_date, end_date, split_date, init_param):
    df = pd.read_csv('data/current_confirmed.csv')
    info = df[df.columns[:4]]
    del df
    model = DTMCSEIR({}, {})
    res = []
    for i in range(info.shape[0]):
        calibrate_data, eval_data = \
            read_calibrate_data(start_date, end_date, split_date, i)
        res.append(fit_predict_evaluate(
            f"{info['Country/Region'].iloc[i]}-{info['Province/State'].iloc[i]}",
            model,
            calibrate_data,
            eval_data,
            init_param
            ))
    print(res)
    df = pd.DataFrame(res)
    df.columns = ['fl', 'flc', 'flr', 'flp', 'fn', 'fnc', 'fnr', 'fnp', 
                  'el', 'elc', 'elr', 'elp', 'en', 'enc', 'enr', 'enp']
    df.to_csv('./results/fit-evaluate.csv', index=False)
    res = pd.concat([info, df], axis=1)
    res.to_csv('./results/fit-evaluate.csv', index=False)
    
def evaluate_plot(start_date, end_date, split_date, init_param, name):
    df = pd.read_csv('data/current_confirmed.csv')
    info = df['Country/Region'].to_numpy().tolist()
    del df
    index = info.index(name)
    model = DTMCSEIR({}, {})
    calibrate_data, eval_data = \
        read_calibrate_data(start_date, end_date, split_date, index)
    res = fit_predict_evaluate(
            f"{name}",
            model,
            calibrate_data,
            eval_data,
            init_param
            )
    print(res)
    
evaluate('2021-03-01', '2021-05-31', '2021-05-01',
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
    })
evaluate_plot('2021-03-01', '2021-05-31', '2021-05-01',
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
    }, name='Germany')
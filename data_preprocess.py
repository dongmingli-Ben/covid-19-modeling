import pandas as pd
import numpy as np

def calculate_current_confirmed_cases(cum_conf_arr, cum_rec_arr, cum_dea_arr):
    # current_conf = np.zeros_like(cum_conf_arr)
    # current_conf[:, 0] = cum_conf_arr[:, 0]
    # for i in range(1, cum_conf_arr.shape[1]):
    #     conf_diff = cum_conf_arr[:, i] - cum_conf_arr[:, i-1]
    #     rec_diff = cum_rec_arr[:, i] - cum_rec_arr[:, i-1]
    #     dea_diff = cum_dea_arr[:, i] - cum_dea_arr[:, i-1]
    #     current_conf[:, i] = current_conf[:, i-1] + conf_diff - rec_diff - dea_diff
    current_conf = cum_conf_arr - cum_rec_arr - cum_dea_arr
    return current_conf

def get_current_confirmed_cases(cum_conf_path, cum_rec_path, cum_dea_path):
    cum_conf = pd.read_csv(cum_conf_path)
    cum_rec = pd.read_csv(cum_rec_path)
    cum_dea = pd.read_csv(cum_dea_path)
    cum_conf['code'] = cum_conf.apply(lambda s: f"{s['Country/Region']}-{s['Province/State']}", axis=1)
    cum_rec['code'] = cum_rec.apply(lambda s: f"{s['Country/Region']}-{s['Province/State']}", axis=1)
    cum_dea['code'] = cum_dea.apply(lambda s: f"{s['Country/Region']}-{s['Province/State']}", axis=1)
    valid = set(cum_conf['code'].to_numpy().tolist()).intersection(
        cum_rec['code'].to_numpy().tolist()).intersection(cum_dea['code'].to_numpy().tolist())
    cum_conf = cum_conf[cum_conf['code'].isin(valid)]
    cum_rec = cum_rec[cum_rec['code'].isin(valid)]
    cum_dea = cum_dea[cum_dea['code'].isin(valid)]
    cum_conf = cum_conf.sort_values('code').reset_index(drop=True)
    cum_rec = cum_rec.sort_values('code').reset_index(drop=True)
    cum_dea = cum_dea.sort_values('code').reset_index(drop=True)
    # import pdb; pdb.set_trace()
    cum_conf = cum_conf.drop('code', axis=1)
    cum_rec = cum_rec.drop('code', axis=1)
    cum_dea = cum_dea.drop('code', axis=1)
    cum_conf = cum_conf[cum_conf.columns[:565]]
    cum_rec = cum_rec[cum_rec.columns[:565]]
    cum_dea = cum_dea[cum_dea.columns[:565]]
    assert cum_conf.shape == cum_rec.shape == cum_dea.shape, \
        f'{cum_conf.shape}, {cum_rec.shape}, {cum_dea.shape}'
    cur_conf_arr = calculate_current_confirmed_cases(
        cum_conf[cum_conf.columns[4:]].to_numpy(),
        cum_rec[cum_rec.columns[4:]].to_numpy(),
        cum_dea[cum_dea.columns[4:]].to_numpy(),
    )
    df = cum_conf[cum_conf.columns[:4]]
    cases = pd.DataFrame(cur_conf_arr, columns=cum_conf.columns[4:])
    df = pd.concat([df, cases], axis=1)
    # filter incorrect entries (negative active cases)
    negative = cases.apply(lambda s: s.min() < 0, axis=1)
    df = df[~negative]
    cum_conf = cum_conf[~negative]
    cum_rec = cum_rec[~negative]
    cum_dea = cum_dea[~negative]
    cum_conf.to_csv('data/cumulative-confirmed.csv', index=False)
    cum_rec.to_csv('data/cumulative-recovered.csv', index=False)
    cum_dea.to_csv('data/cumulative-deaths.csv', index=False)
    return df

current_confirmed = get_current_confirmed_cases(
    'data/time_series_covid19_confirmed_global.csv',
    'data/time_series_covid19_recovered_global.csv',
    'data/time_series_covid19_deaths_global.csv',
)
print(current_confirmed)
current_confirmed.to_csv('data/current_confirmed.csv', index=False)
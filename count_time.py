from simulator import DTMCSEIR, CTMCSEIR, ApproxSEIR, ExactSEIR, Model
from time import time

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

init_state = {
    'i': 1e5,
    'r': 2e6,
    'cr': 2e6,
    'p': 1e5,
}

days = 100
models = [
    (ExactSEIR({}, {}), 'ExactSEIR'),
    (DTMCSEIR({}, {}), 'DTMCSEIR'),
    (CTMCSEIR({}, {}), 'CTMCSEIR'),
]

for model, name in models:
    model.reset(init_param, init_state, keep_init=False)
    t0 = time()
    for i in range(days):
        if isinstance(model, ExactSEIR):
            model.step(steps=100)
        else:
            model.step()
    t = time() - t0
    print(f'{name} {t}s')
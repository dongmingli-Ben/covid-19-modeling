import numpy as np
import matplotlib.pyplot as plt
from simulator import DTMCSEIR, ExactSEIR, CTMCSEIR

def compare_and_plot(init_state, param, n_days, n_samples):
    ode = ExactSEIR(param, init_state)
    ctmc = CTMCSEIR(param, init_state)
    # ode
    for i in range(n_days):
        ode.step()
    fig, ax = plt.subplots()
    ode.plot(ax, linestyle='--')
    colors = ['aqua', 'coral', 'g', 'orange', 'purple', 'hotpink']
    # ctmc
    s, e, i, r, p, cr = [], [], [], [], [], []
    for k in range(n_samples):
        ctmc.reset(state=init_state, keep_init=False)
        for j in range(n_days):
            ctmc.step()
        s_, e_, i_, r_, p_, cr_ = zip(*ctmc.history)
        s.append(s_)
        e.append(e_)
        i.append(i_)
        r.append(r_)
        p.append(p_)
        cr.append(cr_)
    s = np.array(s)
    e = np.array(e)
    i = np.array(i)
    r = np.array(r)
    p = np.array(p)
    cr = np.array(cr)
    sm = np.mean(s, axis=0)
    em = np.mean(e, axis=0)
    im = np.mean(i, axis=0)
    rm = np.mean(r, axis=0)
    pm = np.mean(p, axis=0)
    crm = np.mean(cr, axis=0)
    sstd = np.std(s, axis=0, ddof=1)
    estd = np.std(e, axis=0, ddof=1)
    istd = np.std(i, axis=0, ddof=1)
    rstd = np.std(r, axis=0, ddof=1)
    pstd = np.std(p, axis=0, ddof=1)
    crstd = np.std(cr, axis=0, ddof=1)
    t = np.arange(n_days+1)
    ax.plot(t, sm, label='Susceptible (CTMC)', c=colors[0])
    ax.plot(t, em, label='Exposed (CTMC)', c=colors[1])
    ax.plot(t, im, label='Infected (CTMC)', c=colors[2])
    ax.plot(t, rm, label='Recovered (CTMC)', c=colors[3])
    ax.plot(t, pm, label='Deaths (CTMC)', c=colors[4])
    ax.plot(t, crm, label='Cum Recovered (CTMC)', c=colors[5])
    ax.fill_between(t, sm - 1.96*sstd/n_samples**0.5, sm + 1.96*sstd/n_samples**0.5, 
                    alpha=.3, color=colors[0])
    ax.fill_between(t, em - 1.96*estd/n_samples**0.5, em + 1.96*estd/n_samples**0.5, 
                    alpha=.3, color=colors[1])
    ax.fill_between(t, im - 1.96*istd/n_samples**0.5, im + 1.96*istd/n_samples**0.5, 
                    alpha=.3, color=colors[2])
    ax.fill_between(t, rm - 1.96*rstd/n_samples**0.5, rm + 1.96*rstd/n_samples**0.5, 
                    alpha=.3, color=colors[3])
    ax.fill_between(t, pm - 1.96*pstd/n_samples**0.5, pm + 1.96*pstd/n_samples**0.5, 
                    alpha=.3, color=colors[4])
    ax.fill_between(t, crm - 1.96*crstd/n_samples**0.5, crm + 1.96*crstd/n_samples**0.5, 
                    alpha=.3, color=colors[5])
    ax.legend(loc='right')
    plt.show()

def compare_and_plot_3(init_state, param, n_days, n_samples):
    ode = ExactSEIR(param, init_state)
    dtmc = DTMCSEIR(param, init_state)
    ctmc = CTMCSEIR(param, init_state)
    # ode
    for i in range(n_days):
        ode.step()
    fig, ax = plt.subplots()
    # ode.plot(ax, linestyle='--')
    colors = ['aqua', 'coral', 'g', 'orange', 'purple', 'hotpink']
    # ctmc
    for i in range(n_days):
        ctmc.step()
    s, e, i, r, p, cr = zip(*ctmc.history)
    t = np.arange(n_days+1)
    ax.plot(t, s, label='Susceptible (CTMC)', c=colors[0])
    ax.plot(t, e, label='Exposed (CTMC)', c=colors[1])
    ax.plot(t, i, label='Infected (CTMC)', c=colors[2])
    ax.plot(t, r, label='Recovered (CTMC)', c=colors[3])
    ax.plot(t, p, label='Deaths (CTMC)', c=colors[4])
    ax.plot(t, cr, label='Cum Recovered (CTMC)', c=colors[5])
    # dtmc
    for i in range(n_days):
        dtmc.step()
    s, e, i, r, p, cr = zip(*dtmc.history)
    t = np.arange(n_days+1)
    ax.plot(t, s, linestyle='-.', label='Susceptible (DTMC)', c=colors[0])
    ax.plot(t, e, linestyle='-.', label='Exposed (DTMC)', c=colors[1])
    ax.plot(t, i, linestyle='-.', label='Infected (DTMC)', c=colors[2])
    ax.plot(t, r, linestyle='-.', label='Recovered (DTMC)', c=colors[3])
    ax.plot(t, p, linestyle='-.', label='Deaths (DTMC)', c=colors[4])
    ax.plot(t, cr, linestyle='-.', label='Cum Recovered (DTMC)', c=colors[5])
    ax.legend()
    plt.show()
    
compare_and_plot({'i': 10}, {'ae': .0001, 'ai': .00001}, 100, 20)
# compare_and_plot_3({'i': 10}, {'ae': .0001, 'ai': .00001}, 100, 20)
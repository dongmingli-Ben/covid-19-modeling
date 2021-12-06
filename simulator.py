import numpy as np

class Model:
    
    def __init__(self, parameters, init_state):
        self.reset(parameters, init_state)
    
    def step(self):
        raise NotImplementedError
    
    def reset(self, parameters, state):
        raise NotImplementedError
    
    def plot(self):
        raise NotImplementedError

class ApproxSEIR(Model):
    
    def __init__(self, parameters, init_state):
        self.history = []
        super().__init__(parameters, init_state)
        # store E, I, R, P, CR (cumulative recovered) only (since system is conservative)
        
    def reset(self, parameters=None, state=None, clear_history=True, keep_init=True):
        if clear_history:
            if keep_init and self.history != []: self.history = [self.history[0]]
            else: self.history = []
        if parameters is not None:
            self.ae = parameters.get('ae', 1)
            self.ai = parameters.get('ai', 1)
            self.y = parameters.get('gamma', 0.01)
            self.k = parameters.get('kappa', 0.1)
            self.b = parameters.get('beta', 0.1)
            self.rho = parameters.get('rho', 0.01)
            self.m = parameters.get('mu', 0.01)
        if state is not None:
            self.s0 = state.get('s', 10000)
            self.e = state.get('e', 0)
            self.i = state.get('i', 1)
            self.r = state.get('r', 0)
            self.p = state.get('p', 0)
            self.cr = state.get('cr', 0)
            self.history.append((self.e, self.i, self.r, self.p, self.cr))
    
    def step(self, steps=10):
        dt = 1/steps
        # ds = -self.ae*self.e - self.ai*self.i + self.y*self.r
        for i in range(steps):
            de = self.ae*self.e + self.ai*self.i - self.k*self.e - self.rho*self.e
            di = self.k*self.e - self.b*self.i - self.m*self.i
            dr = self.rho*self.e + self.b*self.i - self.y*self.r
            dp = self.m*self.i
            self.e += de * dt
            self.i += di * dt
            self.r += dr * dt
            self.p += dp * dt
            self.cr += (self.rho*self.e + self.b*self.i) * dt
            # if np.isnan(self.cr): import pdb; pdb.set_trace()
        self.history.append((self.e, self.i, self.r, self.p, self.cr))
        return self.e, self.i, self.r, self.p
    
    def plot(self, ax):
        t = np.arange(len(self.history))
        e, i, r, p, cr = zip(*self.history)
        ax.plot(t, e, label='Exposed')
        ax.plot(t, i, label='Infected')
        ax.plot(t, r, label='Recovered')
        ax.plot(t, p, label='Deaths')
        ax.plot(t, cr, label='Cum Recovered')
        ax.legend()
        
        
class ExactSEIR(Model):
    
    def __init__(self, parameters, init_state):
        self.history = []
        super().__init__(parameters, init_state)
        # store E, I, R, P, CR (cumulative recovered) only (since system is conservative)
        
    def reset(self, parameters=None, state=None, clear_history=True, keep_init=True):
        if clear_history:
            if keep_init and self.history != []: self.history = [self.history[0]]
            else: self.history = []
        if parameters is not None:
            self.ae = parameters.get('ae', 1)
            self.ai = parameters.get('ai', 1)
            self.y = parameters.get('gamma', 0.01)
            self.k = parameters.get('kappa', 0.1)
            self.b = parameters.get('beta', 0.1)
            self.rho = parameters.get('rho', 0.01)
            self.m = parameters.get('mu', 0.01)
            if 's' in parameters: # initial susceptible
                self.s = parameters['s']
            if 'e' in parameters: # initial exposed
                self.e = parameters['e']
        if state is not None:
            # priority: param > state for s
            if parameters is None or 's' not in parameters:
                self.s = state.get('s', 10000)
            if parameters is None or 'e' not in parameters:
                self.e = state.get('e', 0)
            self.i = state.get('i', 1)
            self.r = state.get('r', 0)
            self.p = state.get('p', 0)
            self.cr = state.get('cr', 0)
            self.history.append((self.s, self.e, self.i, self.r, self.p, self.cr))
    
    def step(self, steps=10):
        dt = 1/steps
        for i in range(steps):
            ds = -self.ae*self.e*self.s - self.ai*self.i*self.s + self.y*self.r
            # if np.isinf(ds): import pdb; pdb.set_trace()
            de = self.ae*self.e*self.s + self.ai*self.i*self.s - self.k*self.e - self.rho*self.e
            di = self.k*self.e - self.b*self.i - self.m*self.i
            dr = self.rho*self.e + self.b*self.i - self.y*self.r
            dp = self.m*self.i
            self.s += ds * dt
            self.e += de * dt
            self.i += di * dt
            self.r += dr * dt
            self.p += dp * dt
            self.cr += (self.rho*self.e + self.b*self.i) * dt
        self.history.append((self.s, self.e, self.i, self.r, self.p, self.cr))
        return self.s, self.e, self.i, self.r, self.p
    
    def plot(self, ax, **wargs):
        t = np.arange(len(self.history))
        s, e, i, r, p, cr = zip(*self.history)
        ax.plot(t, s, label='Susceptible', **wargs)
        ax.plot(t, e, label='Exposed', **wargs)
        ax.plot(t, i, label='Infected', **wargs)
        ax.plot(t, r, label='Recovered', **wargs)
        ax.plot(t, p, label='Deaths', **wargs)
        ax.plot(t, cr, label='Cum Recovered', **wargs)
        ax.legend()
        
        
class CTMCSEIR(Model):
    
    def __init__(self, parameters, init_state):
        self.history = []
        super().__init__(parameters, init_state)
        # store E, I, R, P, CR (cumulative recovered) only (since system is conservative)
        
    def reset(self, parameters=None, state=None, clear_history=True, keep_init=True):
        if clear_history:
            if keep_init and self.history != []: self.history = [self.history[0]]
            else: self.history = []
        if parameters is not None:
            self.ae = parameters.get('ae', 1)
            self.ai = parameters.get('ai', 1)
            self.y = parameters.get('gamma', 0.01)
            self.k = parameters.get('kappa', 0.1)
            self.b = parameters.get('beta', 0.1)
            self.rho = parameters.get('rho', 0.01)
            self.m = parameters.get('mu', 0.01)
            if 's' in parameters: # initial susceptible
                self.s = parameters['s']
            if 'e' in parameters: # initial exposed
                self.e = parameters['e']
        if state is not None:
            # priority: param > state for s
            if parameters is None or 's' not in parameters:
                self.s = state.get('s', 10000)
            if parameters is None or 'e' not in parameters:
                self.e = state.get('e', 0)
            self.i = state.get('i', 1)
            self.r = state.get('r', 0)
            self.p = state.get('p', 0)
            self.cr = state.get('cr', 0)
            self.history.append((self.s, self.e, self.i, self.r, self.p, self.cr))
    
    def step(self):
        t = 0
        while True:
            # import pdb; pdb.set_trace()
            s2e = np.random.exponential(1 / (self.ae*self.s*self.e + self.ai*self.s*self.i)) \
                if self.ae*self.s*self.e + self.ai*self.s*self.i > 0 else 1e10
            e2i = np.random.exponential(1 / (self.k*self.e)) if self.k*self.e > 0 else 1e10
            i2p = np.random.exponential(1 / (self.m*self.i)) if self.m*self.i > 0 else 1e10
            e2r = np.random.exponential(1 / (self.rho*self.e)) if self.rho*self.e > 0 else 1e10
            i2r = np.random.exponential(1 / (self.b*self.i)) if self.b*self.i > 0 else 1e10
            r2s = np.random.exponential(1 / (self.y*self.r)) if self.y*self.r > 0 else 1e10
            min_t = min(s2e, e2i, i2p, e2r, i2r, r2s)
            t += min_t
            if t > 1: break
            if s2e == min_t: self.s, self.e = self.s-1, self.e+1
            elif e2i == min_t: self.e, self.i = self.e-1, self.i+1
            elif i2p == min_t: self.i, self.p = self.i-1, self.p+1
            elif e2r == min_t: self.e, self.r = self.e-1, self.r+1; self.cr += 1
            elif i2r == min_t: self.i, self.r = self.i-1, self.r+1; self.cr += 1
            else: self.r, self.s = self.r-1, self.s+1
        self.history.append((self.s, self.e, self.i, self.r, self.p, self.cr))
        return self.s, self.e, self.i, self.r, self.p
    
    def plot(self, ax, **wargs):
        t = np.arange(len(self.history))
        s, e, i, r, p, cr = zip(*self.history)
        ax.plot(t, s, label='Susceptible', **wargs)
        ax.plot(t, e, label='Exposed', **wargs)
        ax.plot(t, i, label='Infected', **wargs)
        ax.plot(t, r, label='Recovered', **wargs)
        ax.plot(t, p, label='Deaths', **wargs)
        ax.plot(t, cr, label='Cum Recovered', **wargs)
        ax.legend()
        
class DTMCSEIR(Model):
    
    def __init__(self, parameters, init_state):
        self.history = []
        super().__init__(parameters, init_state)
        # store E, I, R, P, CR (cumulative recovered) only (since system is conservative)
        
    def reset(self, parameters=None, state=None, clear_history=True, keep_init=True):
        if clear_history:
            if keep_init and self.history != []: self.history = [self.history[0]]
            else: self.history = []
        if parameters is not None:
            self.ae = parameters.get('ae', 1)
            self.ai = parameters.get('ai', 1)
            self.y = parameters.get('gamma', 0.01)
            self.k = parameters.get('kappa', 0.1)
            self.b = parameters.get('beta', 0.1)
            self.rho = parameters.get('rho', 0.01)
            self.m = parameters.get('mu', 0.01)
            self.max_e = parameters.get('max_e', 1e9)
            self.max_i = parameters.get('max_i', 1e9)
            if 's' in parameters: # initial susceptible
                self.s = parameters['s']
            if 'e' in parameters: # initial exposed
                self.e = parameters['e']
        if state is not None:
            # priority: param > state for s, e
            if parameters is None or 's' not in parameters:
                self.s = state.get('s', 10000)
            if parameters is None or 'e' not in parameters:
                self.e = state.get('e', 0)
            self.i = state.get('i', 1)
            self.r = state.get('r', 0)
            self.p = state.get('p', 0)
            self.cr = state.get('cr', 0)
            self.history.append((self.s, self.e, self.i, self.r, self.p, self.cr))
    
    def step(self):
        t = 0
        # import pdb; pdb.set_trace()
        s2e = np.random.binomial(self.s, self.ae*min(self.e, self.max_e) + self.ai*min(self.i, self.max_i))
        e2r, e2i, _ = np.random.multinomial(self.e, [self.rho, self.k, 1-self.rho-self.k])
        i2p, i2r, _ = np.random.multinomial(self.i, [self.m, self.b, 1-self.m-self.b])
        r2s = np.random.binomial(self.r, self.y)
        self.s += -s2e + r2s
        self.e += s2e - e2r - e2i
        self.i += e2i - i2p - i2r
        self.p += i2p
        self.r += e2r + i2r - r2s
        self.cr += i2r
        self.history.append((self.s, self.e, self.i, self.r, self.p, self.cr))
        return self.s, self.e, self.i, self.r, self.p
    
    def plot(self, ax, **wargs):
        t = np.arange(len(self.history))
        s, e, i, r, p, cr = zip(*self.history)
        # ax.plot(t, s, label='Susceptible', **wargs)
        ax.plot(t, e, label='Exposed', **wargs)
        ax.plot(t, i, label='Infected', **wargs)
        ax.plot(t, r, label='Recovered', **wargs)
        ax.plot(t, p, label='Deaths', **wargs)
        ax.plot(t, cr, label='Cum Recovered', **wargs)
        ax.legend()
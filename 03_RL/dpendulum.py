from pendulum import Pendulum
import numpy as np
from numpy import pi
import time

def plot_V_table(env, V):
    import matplotlib.pyplot as plt
    Q,DQ = np.meshgrid([env.d2cq(i) for i in range(env.nq)], 
                        [env.d2cv(i) for i in range(env.nv)])
    plt.pcolormesh(Q, DQ, V.reshape((env.nv,env.nq)), cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    plt.title('V table')
    plt.xlabel("q")
    plt.ylabel("dq")
    plt.show()
    
def plot_policy(env, pi):
    import matplotlib.pyplot as plt
    Q,DQ = np.meshgrid([env.d2cq(i) for i in range(env.nq)], 
                        [env.d2cv(i) for i in range(env.nv)])
    plt.pcolormesh(Q, DQ, pi.reshape((env.nv,env.nq)), cmap=plt.cm.get_cmap('RdBu'))
    plt.colorbar()
    plt.title('Policy')
    plt.xlabel("q")
    plt.ylabel("dq")
    plt.show()
    
# --- Discretized PENDULUM
class DPendulum:
    def __init__(self, nq=51, nv=21, nu=11, vMax=5, uMax=5, dt=0.2, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(1,noise_stddev)
        self.pendulum.DT  = dt
        self.pendulum.NDT = ndt
        self.nq = nq        # Discretization steps for position
        self.nv = nv        # Discretization steps for velocity
        self.vMax = vMax    # Max velocity (v in [-vmax,vmax])
        self.nu = nu        # Discretization steps for torque
        self.uMax = uMax    # Max torque (u in [-umax,umax])
        self.dt = dt
        self.DQ = 2*pi/nq
        self.DV = 2*vMax/nv
        self.DU = 2*uMax/nu

    @property
    def nqv(self): return [self.nq,self.nv]
    @property
    def nx(self): return self.nq*self.nv
    @property
    def goal(self): return self.x2i(self.c2d([0.,0.]))
    
    # Continuous to discrete
    def c2dq(self, q):
        q = (q+pi)%(2*pi)
        return int(np.floor(q/self.DQ))  % self.nq
    
    def c2dv(self, v):
        v = np.clip(v,-self.vMax+1e-3,self.vMax-1e-3)
        return int(np.floor((v+self.vMax)/self.DV))
    
    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))
    
    def c2d(self, qv):
        '''From continuous to 2d discrete.'''
        return np.array([self.c2dq(qv[0]), self.c2dv(qv[1])])
    
    # Discrete to continuous
    def d2cq(self, iq):
        iq = np.clip(iq,0,self.nq-1)
        return iq*self.DQ - pi + 0.5*self.DQ
    
    def d2cv(self, iv):
        iv = np.clip(iv,0,self.nv-1) - (self.nv-1)/2
        return iv*self.DV
    
    def d2cu(self, iu):
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
        return iu*self.DU
    
    def d2c(self, iqv):
        '''From 2d discrete to continuous'''
        return np.array([self.d2cq(iqv[0]), self.d2cv(iqv[1])])
    
    def x2i(self, x): return x[0]+x[1]*self.nq
    
    ''' From 1d discrete to 2d discrete '''
    def i2x(self, i): return [ i%self.nq, int(np.floor(i/self.nq)) ]

    def reset(self,x=None):
        if x is None:
            x = [ np.random.randint(0,self.nq), np.random.randint(0,self.nv) ]
        else: x = self.i2x(x)
        assert(len(x)==2)
        self.x = x
        return self.x2i(self.x)

    def step(self,iu):
        cost     = -1 if self.x2i(self.x)==self.goal else 0
        self.x     = self.dynamics(self.x,iu)
        return self.x2i(self.x), cost

    def render(self):
        q = self.d2cq(self.x[0])
        self.pendulum.display(np.matrix([q,]))
        time.sleep(self.pendulum.DT)

    def dynamics(self,ix,iu):
        x   = self.d2c(ix)
        u   = self.d2cu(iu)
        
        self.xc,_ = self.pendulum.dynamics(x,u)
        return self.c2d(self.xc)
    
if __name__=="__main__":
    print("Start tests")
    env = DPendulum()
    nq = env.nq
    nv = env.nv
    
    for i in range(nq*nv):
        x = env.i2x(i)
        i_test = env.x2i(x)
        if(i!=i_test):
            print("ERROR! x2i(i2x(i))=", i_test, "!= i=", i)
        
        xc = env.d2c(x)
        x_test = env.c2d(xc)
        if(x_test[0]!=x[0] or x_test[1]!=x[1]):
            print("ERROR! c2d(d2c(x))=", x_test, "!= x=", x)
        xc_test = env.d2c(x_test)
        if(np.linalg.norm(xc-xc_test)>1e-10):
            print("ERROR! xc=", xc, "xc_test=", xc_test)
    print("Tests finished")
    
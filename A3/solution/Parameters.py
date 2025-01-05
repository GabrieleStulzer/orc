class Parameters:
    def __init__(self, index, N, dt, x_init, w_u, u_min, u_max):
        self.index = index
        self.N = N
        self.dt = dt
        self.w_u = w_u
        self.u_min = u_min
        self.u_max = u_max
        self.x_init = x_init
    
    def setXInit(self, x_init):
        self.x_init = x_init

    def __str__(self) -> str:
        return f"Index = {self.index} - Parameters(N={self.N}, dt={self.dt}, x_init={self.x_init}, w_u={self.w_u}, u_min={self.u_min}, u_max={self.u_max})"

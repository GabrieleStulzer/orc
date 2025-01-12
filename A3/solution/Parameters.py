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
    
    def save(self, filename):
        with open (filename, 'w') as f:
            f.write(f"{self.index}\n")
            f.write(f"{self.N}\n")
            f.write(f"{self.dt}\n")
            f.write(f"{self.x_init}\n")
            f.write(f"{self.w_u}\n")
            f.write(f"{self.u_min}\n")
            f.write(f"{self.u_max}\n")
    
    @staticmethod
    def load(filename):
        with open (filename, 'r') as f:
            index = int(f.readline().strip())
            N = int(f.readline().strip())
            dt = float(f.readline().strip())
            x_init = float(f.readline().strip())
            w_u = float(f.readline().strip())
            u_min = float(f.readline().strip())
            u_max = float(f.readline().strip())
            return Parameters(index, N, dt, x_init, w_u, u_min, u_max)

    def __str__(self) -> str:
        return f"Index = {self.index} - Parameters(N={self.N}, dt={self.dt}, x_init={self.x_init}, w_u={self.w_u}, u_min={self.u_min}, u_max={self.u_max})"

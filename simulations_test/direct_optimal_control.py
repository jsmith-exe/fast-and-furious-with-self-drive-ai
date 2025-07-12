import casadi as ca
import matplotlib.pyplot as plt

opti = ca.Opti()

N = 20 # control intervals
dt = 0.1

# State and control trajectories
X = opti.variable(2 , N+1)
U = opti.variable(1, N)

# Dynamic: ẋ₁ = x₂,  ẋ₂ = u
def f(x, u):
    return ca.vertcat(x[1], u)

for k in range(N):
    xk = X[:,k]
    uk = U[k]
    x_next = X[:,k+1]
    opti.subject_to(x_next == xk + dt * f(xk, uk))

# Boundary conditions
opti.subject_to(X[:,0] == [0,0])
opti.subject_to(X[0,-1] == 1)

# Cost: minimize control effort
opti.minimize(ca.sumsqr(U))

# Solve
opti.solver('ipopt')
sol = opti.solve()

# Extract and plot (e.g. with matplotlib)
X1 = sol.value(X[0,:])
X2 = sol.value(X[1,:])
plt.plot(X1, X2, '-o')
plt.xlabel('x₁'); plt.ylabel('x₂')
plt.title('State Trajectory')
plt.show()
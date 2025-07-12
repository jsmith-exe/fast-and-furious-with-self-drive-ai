import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# 1) Build & solve the same MPC as before
opti = ca.Opti()
N, dt = 50, 0.1
X = opti.variable(2, N+1)
U = opti.variable(1, N)
x_ref = opti.parameter(2,1)

def f(x, u):
    return ca.vertcat(x[1], u)

for k in range(N):
    opti.subject_to(X[:,k+1] == X[:,k] + dt*f(X[:,k], U[k]))
opti.subject_to(X[:,0] == [0,0])

Q = ca.diag([10,1]); R = 0.1
cost = 0
for k in range(N):
    e = X[:,k] - x_ref
    cost += e.T@Q@e + R*(U[k]**2)
eN = X[:,N] - x_ref
cost += eN.T@Q@eN
opti.minimize(cost)

opti.solver('ipopt')
opti.set_value(x_ref, [50.0, 0.0])   # constant setpoint [x=1, y=0]
sol = opti.solve()

X_val = sol.value(X)
U_val = sol.value(U)

# 2) Time vectors
t   = np.linspace(0, N*dt,   N+1)
t_u = np.linspace(0, N*dt-dt, N)

# 3) Plot x₁ vs time (with setpoint)
plt.figure()
plt.plot(t,   X_val[0,:],       label='x₁')
plt.plot(t,   np.ones_like(t)*50, label='setpoint')
plt.xlabel('Time [s]')
plt.ylabel('x₁')
plt.title('State x₁ Tracking Setpoint')
plt.legend()
plt.show()

# 4) Plot control input vs time
plt.figure()
plt.plot(t_u, U_val.flatten())
plt.xlabel('Time [s]')
plt.ylabel('u (throttle/accel)')
plt.title('Control Input over Time')
plt.show()


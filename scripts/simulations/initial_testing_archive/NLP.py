import casadi as ca

opti = ca.Opti()

# Decision variables
x = opti.variable(2) # a 2-vector
u = opti.variable(3) # a 3-vector

# Objective: ‖x‖² + ‖u‖²
opti.minimize(ca.sumsqr(x) + ca.sumsqr(u))

# Constraints
opti.subject_to( x[0] + 2*x[1] == 1 )
opti.subject_to( opti.bounded(-1, u, 1))

# Chose solver (default is IPOPT)
opti.solver('ipopt')

# Initial guesses
opti.set_initial(x, [0.0])
opti.set_initial(u, [0,0,0])

# Solve
sol = opti.solve()
print("x = ", sol.value(x))
print("u = ", sol.value(u))

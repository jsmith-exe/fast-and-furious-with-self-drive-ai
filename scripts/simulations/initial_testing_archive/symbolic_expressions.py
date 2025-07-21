import casadi as ca

# Define symbols
x = ca.SX.sym('x')
y = ca.SX.sym('y')

# Build an expression
f = x**2 * y + ca.sin(y)

# Compute its Jacobian
J = ca.jacobian(f, ca.vertcat(x, y))

# Turn into a reusable function
fun = ca.Function('fun', [x, y], [f, J])

# Evaluate at x=2 , y=0.5
val, jac = fun(2, 0.5)
print("f =", val)
print("J =", jac)


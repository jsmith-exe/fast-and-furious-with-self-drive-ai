import os
import sys
# Ensure Turtle can find Tcl/Tk
base_prefix = getattr(sys, 'base_prefix', sys.prefix)
os.environ['TCL_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tk8.6')

import casadi as ca
import numpy as np
import turtle

# ============ USER CONFIGURABLE ============
# List of waypoints: [x, y, psi, v]
waypoints = [
    [2.0, 2.0, np.pi/4, 0.0]
]
# Maximum speed (m/s)
v_max    = 1.5
# Maximum acceleration (m/s^2)
a_max    = 1.0
# Tolerances
tol_xy   = 0.025   # position tolerance (m)
tol_psi  = 0.05   # heading tolerance (rad)
tol_v    = 0.01   # speed tolerance (m/s)
# MPC parameters
N, dt    = 15, 0.15  # horizon length and timestep
L         = 0.18       # wheelbase (m)
max_iters = 100      # maximum simulation steps
# Cost weights (tune these to improve convergence)
Q_weights = [10, 10, 3, 1]  # state tracking weights [x, y, psi, v]
R_weights = [0.05, 0.05]     # control effort weights [a, delta]
# ============================================

# Build a CasADi MPC problem once
def build_mpc(v_max, a_max, Qw, Rw):
    opti = ca.Opti()
    X = opti.variable(4, N+1)  # state: [x,y,psi,v]
    U = opti.variable(2, N)    # control: [a,delta]
    x0 = opti.parameter(4)
    xref = opti.parameter(4, N+1)
    # Dynamics
    xSym, uSym = ca.SX.sym('x',4), ca.SX.sym('u',2)
    f = ca.vertcat(
        xSym[3]*ca.cos(xSym[2]),
        xSym[3]*ca.sin(xSym[2]),
        xSym[3]/L*ca.tan(uSym[1]),
        uSym[0]
    )
    f_fun = ca.Function('f', [xSym,uSym], [f])
    # Constraints
    opti.subject_to(X[:,0] == x0)
    for k in range(N):
        opti.subject_to(X[:,k+1] == X[:,k] + dt*f_fun(X[:,k], U[:,k]))
        opti.subject_to(-v_max <= X[3,k]); opti.subject_to(X[3,k] <= v_max)
        opti.subject_to(-a_max <= U[0,k]); opti.subject_to(U[0,k] <= a_max)
    opti.subject_to(-v_max <= X[3,N]); opti.subject_to(X[3,N] <= v_max)
    # Cost
    Q = ca.diag(Qw); R = ca.diag(Rw)
    cost = 0
    for k in range(N):
        e = X[:,k] - xref[:,k]
        cost += e.T@Q@e + U[:,k].T@R@U[:,k]
    eN = X[:,N] - xref[:,N]
    cost += eN.T@Q@eN
    opti.minimize(cost)
    # Solver
    solver_opts = {'ipopt.print_level':0, 'print_time':False, 'ipopt.max_iter': 50}
    opti.solver('ipopt', solver_opts)
    return opti, X, U, x0, xref, f_fun

# Simulate driving from start_state to a single waypoint
def simulate_leg(opti, X, U, x0, xref, f_fun, start_state, goal):
    x_cur = np.array(start_state, float)
    traj = []
    # warm start
    X_prev = np.tile(x_cur.reshape(4,1),(1,N+1))
    U_prev = np.zeros((2,N))
    for i in range(max_iters):
        # reference trajectory constant at goal
        xr = np.tile(np.array(goal).reshape(4,1),(1,N+1))
        opti.set_value(x0, x_cur); opti.set_value(xref, xr)
        opti.set_initial(X, X_prev); opti.set_initial(U, U_prev)
        try:
            sol = opti.solve()
            X_prev = sol.value(X); U_prev = sol.value(U)
            u0 = U_prev[:,0]
        except RuntimeError:
            # fallback zero control
            u0 = np.zeros(2)
        traj.append(x_cur.copy())
        fval = f_fun(x_cur, u0)
        x_cur = x_cur + dt*np.array(fval).flatten()
        psi_err = ((x_cur[2]-goal[2]+np.pi)%(2*np.pi))-np.pi
        if (np.linalg.norm(x_cur[:2]-goal[:2])<tol_xy and
            abs(psi_err)<tol_psi and abs(x_cur[3]-goal[3])<tol_v):
            break
    traj.append(x_cur.copy())
    return np.array(traj), x_cur

# Draw full trajectory with grid, waypoints
def draw(traj_all):
    scr = turtle.Screen()
    xs, ys = traj_all[:,0], traj_all[:,1]
    m=1; scr.setworldcoordinates(xs.min()-m, ys.min()-m, xs.max()+m, ys.max()+m)
    # grid
    g=turtle.Turtle(); g.hideturtle(); g.speed(0); g.color('lightgrey')
    for x in range(int(xs.min())-1,int(xs.max())+2): g.penup(); g.goto(x,ys.min()-m); g.pendown(); g.goto(x,ys.max()+m)
    for y in range(int(ys.min())-1,int(ys.max())+2): g.penup(); g.goto(xs.min()-m,y); g.pendown(); g.goto(xs.max()+m,y)
    # waypoints
    for wp in waypoints:
        mk=turtle.Turtle(); mk.hideturtle(); mk.shape('arrow'); mk.color('green'); mk.shapesize(2,2)
        mk.penup(); mk.setheading(np.degrees(wp[2])); mk.goto(wp[0],wp[1]); mk.showturtle()
    # trajectory
    pen=turtle.Turtle(); pen.shape('arrow'); pen.color('blue'); pen.width(2); pen.speed(1)
    pen.penup(); pen.setheading(np.degrees(traj_all[0,2])); pen.goto(xs[0],ys[0]); pen.pendown()
    for x,y,psi,_ in traj_all: pen.setheading(np.degrees(psi)); pen.goto(x,y)
    scr.exitonclick()

# Main flow
opti, X, U, x0, xref, f_fun = build_mpc(v_max, a_max, Q_weights, R_weights)
full_traj = []
# sequentially simulate each leg
def initial_state(): return [0,0,0,0]
state = initial_state()
for wp in waypoints:
    leg_traj, state = simulate_leg(opti, X, U, x0, xref, f_fun, state, wp)
    full_traj.append(leg_traj)
# concatenate
traj_all = np.vstack(full_traj)
draw(traj_all)

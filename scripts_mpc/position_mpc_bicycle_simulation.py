import os
import sys
# Ensure Turtle finds Tcl/Tk
base_prefix = getattr(sys, 'base_prefix', sys.prefix)
os.environ['TCL_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tk8.6')

import casadi as ca
import numpy as np
import turtle

# ============ USER CONFIGURABLE PARAMETERS ============
# Waypoints to follow: [x, y, psi, v]
waypoints = [
    [2.0, 2.0, np.pi/4, 0.0],
    [4.0, 1.0, -np.pi/2, 0.0],
    [3.0, -1.0, 0.0, 0.0],
    [3.0, -1.0, np.pi, 0.0]
]
# Kinematic model bounds
v_max     = 2.0       # max speed (m/s)
a_max     = 1.0       # max acceleration (m/s^2)
delta_max = 0.5       # max steering angle (rad)
# MPC horizon
N, dt     = 30, 0.1   # horizon length and timestep
L          = 0.17      # wheelbase of JetRacer (m)
# Convergence tolerances
tol_xy    = 0.05      # position tolerance (m)
tol_psi   = 0.05      # heading tolerance (rad)
tol_v     = 0.01      # speed tolerance (m/s)
max_iters = 100       # maximum sim steps per waypoint
# Cost weights
Q_weights = [20, 20, 2, 3]    # [x, y, psi, v]
R_weights = [0.1, 0.1]        # [a, delta]
# ====================================================

# Kinematic bicycle model dynamics
def f_kin(x, u):
    # State: x=[px, py, psi, v], u=[a, delta]
    px, py, psi, v = x[0], x[1], x[2], x[3]
    a, delta = u[0], u[1]
    dx = ca.vertcat(
        v * ca.cos(psi),
        v * ca.sin(psi),
        v / L * ca.tan(delta),
        a
    )
    return dx

# Build the MPC optimization problem
def build_mpc():
    opti = ca.Opti()
    X = opti.variable(4, N+1)   # state over horizon
    U = opti.variable(2, N)     # controls over horizon
    x0 = opti.parameter(4)      # initial state
    xref = opti.parameter(4, N+1)  # reference states
    # Initial condition
    opti.subject_to(X[:,0] == x0)
    # Dynamics and input bounds
    for k in range(N):
        xk = X[:,k]; uk = U[:,k]
        x_next = xk + dt * f_kin(xk, uk)
        opti.subject_to(X[:,k+1] == x_next)
        opti.subject_to(opti.bounded(-a_max, uk[0], a_max))
        opti.subject_to(opti.bounded(-delta_max, uk[1], delta_max))
        opti.subject_to(opti.bounded(-v_max, X[3,k], v_max))
    opti.subject_to(opti.bounded(-v_max, X[3,N], v_max))
    # Cost function
    Q = ca.diag(Q_weights)
    R = ca.diag(R_weights)
    cost = 0
    for k in range(N):
        e = X[:,k] - xref[:,k]
        cost += e.T @ Q @ e + U[:,k].T @ R @ U[:,k]
    eN = X[:,N] - xref[:,N]
    cost += eN.T @ Q @ eN
    opti.minimize(cost)
    # Solver settings
    opti.solver('ipopt', {'ipopt.print_level':0, 'print_time':False})
    return opti, X, U, x0, xref

# Closed-loop simulation
def simulate():
    opti, X, U, x0, xref = build_mpc()
    state = np.array([0.0, 0.0, 0.0, 0.0])
    traj_all = []
    for wp in waypoints:
        # Build constant reference trajectory for this waypoint
        xr = np.tile(np.array(wp, float).reshape(4,1), (1, N+1))
        for _ in range(max_iters):
            opti.set_value(x0, state)
            opti.set_value(xref, xr)
            sol = opti.solve()
            u0 = sol.value(U[:,0])
            traj_all.append(state.copy())
            # simulate kinematics
            dx = f_kin(state, u0)
            state = state + dt * np.array(dx).flatten()
            # check convergence
            psi_err = ((state[2] - wp[2] + np.pi) % (2*np.pi)) - np.pi
            if (np.linalg.norm(state[:2] - wp[:2]) < tol_xy and
                abs(psi_err) < tol_psi and abs(state[3] - wp[3]) < tol_v):
                break
        traj_all.append(state.copy())
    return np.array(traj_all)

# Visualization with Turtle
def draw(traj):
    scr = turtle.Screen()
    xs, ys = traj[:,0], traj[:,1]
    scr.setworldcoordinates(xs.min()-1, ys.min()-1, xs.max()+1, ys.max()+1)
    # grid
    grid = turtle.Turtle(); grid.hideturtle(); grid.speed(0); grid.color('lightgrey')
    for xg in range(int(xs.min())-1, int(xs.max())+2):
        grid.penup(); grid.goto(xg, ys.min()-1); grid.pendown(); grid.goto(xg, ys.max()+1)
    for yg in range(int(ys.min())-1, int(ys.max())+2):
        grid.penup(); grid.goto(xs.min()-1, yg); grid.pendown(); grid.goto(xs.max()+1, yg)
    # waypoints
    for wp in waypoints:
        mk = turtle.Turtle(); mk.hideturtle(); mk.shape('arrow'); mk.color('green'); mk.shapesize(2,2)
        mk.penup(); mk.goto(wp[0], wp[1]); mk.setheading(np.degrees(wp[2])); mk.showturtle()
    # trajectory
    pen = turtle.Turtle(); pen.shape('arrow'); pen.color('blue'); pen.width(2); pen.speed(1)
    pen.penup(); pen.goto(xs[0], ys[0]); pen.setheading(np.degrees(traj[0,2])); pen.pendown()
    for x,y,psi,v in traj:
        pen.setheading(np.degrees(psi)); pen.goto(x,y)
    scr.exitonclick()

# Main
if __name__=='__main__':
    trajectory = simulate()
    draw(trajectory)

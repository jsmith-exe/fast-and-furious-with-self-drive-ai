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
    [3.0, -1.0, 0.0, 0.0]
]
# Kinematic model bounds
v_max     = 2.0       # max speed (m/s)
a_max     = 1.0       # max acceleration (m/s^2)
delta_max = 0.5       # max steering angle (rad)
# MPC horizon
N, dt     = 15, 0.1   # horizon length and timestep
L          = 0.17      # wheelbase of JetRacer (m)
# Convergence tolerances
tol_xy    = 0.05      # position tolerance (m)
tol_psi   = 0.05      # heading tolerance (rad)
tol_v     = 0.01      # speed tolerance (m/s)
max_iters = 100       # maximum sim steps per waypoint
# Cost weights
Q_weights = [10, 10, 5, 1]    # [x, y, psi, v]
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
    preds = []
    for wp in waypoints:
        xr = np.tile(np.array(wp, float).reshape(4,1), (1, N+1))
        for _ in range(max_iters):
            opti.set_value(x0, state)
            opti.set_value(xref, xr)
            sol = opti.solve()
            # record predicted horizon
            Xpred = sol.value(X)
            preds.append(Xpred)
            u0 = sol.value(U[:,0])
            traj_all.append(state.copy())
            dx = f_kin(state, u0)
            state = state + dt * np.array(dx).flatten()
            psi_err = ((state[2] - wp[2] + np.pi) % (2*np.pi)) - np.pi
            if (np.linalg.norm(state[:2] - wp[:2]) < tol_xy and
                abs(psi_err) < tol_psi and abs(state[3] - wp[3]) < tol_v):
                break
        traj_all.append(state.copy())
    return np.array(traj_all), preds

# Visualization with Turtle with real-time horizon animation
def draw(traj, preds=None):
    scr = turtle.Screen()
    xs, ys = traj[:,0], traj[:,1]
    scr.setworldcoordinates(xs.min()-1, ys.min()-1, xs.max()+1, ys.max()+1)
    # Draw static grid
    grid = turtle.Turtle(); grid.hideturtle(); grid.speed(0); grid.color('lightgrey')
    for xg in range(int(xs.min())-1, int(xs.max())+2):
        grid.penup(); grid.goto(xg, ys.min()-1); grid.pendown(); grid.goto(xg, ys.max()+1)
    for yg in range(int(ys.min())-1, int(ys.max())+2):
        grid.penup(); grid.goto(xs.min()-1, yg); grid.pendown(); grid.goto(xs.max()+1, yg)
    # Draw waypoints
    for wp in waypoints:
        mk = turtle.Turtle(); mk.hideturtle(); mk.shape('arrow'); mk.color('green'); mk.shapesize(2,2)
        mk.penup(); mk.goto(wp[0], wp[1]); mk.setheading(np.degrees(wp[2])); mk.showturtle()
    # Create pen for actual trajectory
    pen = turtle.Turtle(); pen.shape('arrow'); pen.color('blue'); pen.width(2); pen.speed(1)
    pen.penup(); pen.goto(xs[0], ys[0]); pen.setheading(np.degrees(traj[0,2])); pen.pendown()
    # Animate: at each step, draw predicted horizon then move
    pred_turtles = []
    for i in range(len(preds)):
        # clear previous predictions
        for pt in pred_turtles:
            pt.clear(); pt.hideturtle()
        pred_turtles.clear()
        # draw current predicted horizon
        ph = preds[i]
        for k in range(ph.shape[1]):
            if k == 0:
                pt = turtle.Turtle(); pt.hideturtle(); pt.speed(0)
                pt.color('lightgrey'); pt.width(1)
                pt.penup(); pt.goto(ph[0,0], ph[1,0]); pt.pendown()
                pred_turtles.append(pt)
            else:
                pred_turtles[-1].goto(ph[0,k], ph[1,k])
        # step actual pen
        x,y,psi,_ = traj[i]
        pen.setheading(np.degrees(psi)); pen.goto(x, y)
    # Final position
    x,y,psi,_ = traj[-1]
    pen.setheading(np.degrees(psi)); pen.goto(x, y)
    scr.exitonclick()

# Main
if __name__=='__main__':
    traj, preds = simulate()
    draw(traj, preds)

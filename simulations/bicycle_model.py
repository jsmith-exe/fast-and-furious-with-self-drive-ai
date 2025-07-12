import os
import sys
# Set Tcl/Tk library paths for turtle
base_prefix = getattr(sys, 'base_prefix', sys.prefix)
os.environ['TCL_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tk8.6')

import casadi as ca
import numpy as np
import argparse
import turtle

# ==============================
# Global MPC parameters (defaults)
# ==============================
N, dt = 20, 0.1        # horizon length and timestep
L = 0.15               # wheelbase [m]


def build_mpc_problem(N, dt, L, v_max):
    """
    Constructs and returns a CasADi Opti instance for kinematic bicycle MPC
    with a maximum speed constraint.
    """
    opti = ca.Opti()
    # Decision variables: states and controls
    X = opti.variable(4, N+1)       # [X, Y, psi, v]
    U = opti.variable(2, N)         # [a, delta]

    # Parameters: initial state and reference trajectory (X, Y, psi, v)
    x0 = opti.parameter(4)
    xref = opti.parameter(4, N+1)

    # Symbolic bicycle dynamics
    xSym = ca.SX.sym('x', 4)
    uSym = ca.SX.sym('u', 2)
    f_bike_sym = ca.vertcat(
        xSym[3] * ca.cos(xSym[2]),      # Xdot = v*cos(psi)
        xSym[3] * ca.sin(xSym[2]),      # Ydot = v*sin(psi)
        xSym[3]/L * ca.tan(uSym[1]),    # psi_dot = v/L * tan(delta)
        uSym[0]                          # v_dot = a
    )
    f_bike_fun = ca.Function('f_bike', [xSym, uSym], [f_bike_sym])

    # Initial condition constraint
    opti.subject_to(X[:,0] == x0)

    # Dynamics and velocity constraints
    for k in range(N):
        # dynamics
        opti.subject_to(X[:,k+1] == X[:,k] + dt * f_bike_fun(X[:,k], U[:,k]))
        # speed limits
        opti.subject_to(X[3,k] <= v_max)
        opti.subject_to(X[3,k] >= 0)
    # enforce at final step too
    opti.subject_to(X[3,N] <= v_max)

    # Cost: track full state to reference and penalize inputs
    Q = ca.diag([10,10,5,1])
    R = ca.diag([0.1,0.1])
    cost = 0
    for k in range(N):
        e = X[:,k] - xref[:,k]
        cost += ca.mtimes([e.T, Q, e]) + U[:,k].T @ R @ U[:,k]
    # terminal cost
    eN = X[:,N] - xref[:,N]
    cost += ca.mtimes([eN.T, Q, eN])
    opti.minimize(cost)

    # Solver settings
    opti.solver('ipopt', {'ipopt.print_level':0, 'print_time':False})
    return opti, X, U, x0, xref, f_bike_fun


def simulate_mpc(opti, X, U, x0, xref, f_bike_fun, N, dt, goal,
                 tol_xy, tol_psi, tol_v, max_iters):
    """
    Runs closed-loop MPC to reach `goal` with tolerances.
    Returns trajectory array of shape (steps+1,4).
    """
    goal = np.array(goal)
    x_cur = np.array([0.0,0.0,0.0,0.0])
    traj = []

    for i in range(max_iters):
        # reference trajectory constant
        xr = np.tile(goal.reshape(4,1), (1,N+1))
        # set parameters
        opti.set_value(x0, x_cur)
        opti.set_value(xref, xr)
        # solve
        sol = opti.solve()
        # record
        traj.append(x_cur.copy())
        # first control
        u0 = sol.value(U[:,0])
        # step dynamics
        f_val = f_bike_fun(x_cur, u0)
        x_cur = x_cur + dt * np.array(f_val).flatten()
        # heading error
        psi_err = ((x_cur[2]-goal[2]+np.pi)%(2*np.pi))-np.pi
        # check goal
        if (np.linalg.norm(x_cur[:2]-goal[:2])<tol_xy and
            abs(psi_err)<tol_psi and abs(x_cur[3]-goal[3])<tol_v):
            print(f"Reached goal at {x_cur} in {i+1} steps.")
            break
    else:
        print("Did not converge within max iterations.")
    traj.append(x_cur.copy())
    return np.array(traj)


def draw_trajectory(traj, goal):
    """Draws grid and trajectory using turtle with orientation."""
    try:
        screen = turtle.Screen()
    except turtle.TclError as e:
        print("Turtle init failed:", e)
        return
    # Set world coordinates with margin
    xs, ys = traj[:,0], traj[:,1]
    m = 1
    screen.setworldcoordinates(xs.min()-m, ys.min()-m,
                               xs.max()+m, ys.max()+m)
    # Draw grid
    grid = turtle.Turtle()
    grid.hideturtle(); grid.speed(0)
    grid.color('lightgrey')
    for x in range(int(np.floor(xs.min()))-1, int(np.ceil(xs.max()))+2):
        grid.penup(); grid.goto(x, ys.min()-m)
        grid.pendown(); grid.goto(x, ys.max()+m)
    for y in range(int(np.floor(ys.min()))-1, int(np.ceil(ys.max()))+2):
        grid.penup(); grid.goto(xs.min()-m, y)
        grid.pendown(); grid.goto(xs.max()+m, y)
    # Setup pen turtle with arrow shape
    pen = turtle.Turtle()
    pen.shape('arrow')
    pen.color('blue'); pen.width(2); pen.speed(1)
    pen.penup();
    # Move to start and set initial heading
    start_psi = traj[0,2]
    pen.setheading(np.degrees(start_psi))
    pen.goto(xs[0], ys[0])
    pen.pendown()
    # Draw trajectory with heading
    for x, y, psi, _ in traj:
        pen.setheading(np.degrees(psi))
        pen.goto(x, y)
    # Mark goal
    marker = turtle.Turtle(); marker.hideturtle(); marker.penup()
    marker.goto(goal[0], goal[1]); marker.dot(10,'red')
    screen.exitonclick()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', nargs=4, type=float,
                        default=[1,2,np.pi,0],
                        help='[x, y, psi, v] final state')
    parser.add_argument('--v_max', type=float, default=2.0,
                        help='maximum allowed speed')
    parser.add_argument('--tol_xy', type=float, default=0.05)
    parser.add_argument('--tol_psi', type=float, default=0.05)
    parser.add_argument('--tol_v', type=float, default=0.01)
    parser.add_argument('--max_iters', type=int, default=20)
    args = parser.parse_args()
    # build with speed limit
    opti, X, U, x0, xref, f_bike_fun = build_mpc_problem(
        N, dt, L, args.v_max)
    traj = simulate_mpc(opti, X, U, x0, xref, f_bike_fun,
                        N, dt, args.goal,
                        args.tol_xy, args.tol_psi,
                        args.tol_v, args.max_iters)
    draw_trajectory(traj, args.goal)

if __name__=='__main__':
    main()

import os
import sys
# Ensure Turtle finds Tcl/Tk
base_prefix = getattr(sys, 'base_prefix', sys.prefix)
os.environ['TCL_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(base_prefix, 'tcl', 'tk8.6')

import casadi as ca
import numpy as np
import turtle
import time

# === MPC Parameters ===
N = 40          # Prediction horizon
dt = 0.05        # Time step (s)
v = 2.0         # Forward speed (units per step)

# Steering limits
max_steer = np.pi/4  # ±45°

# MPC Weights
w_y = 10        # Cross-track error weight
w_delta = 1     # Steering effort weight

# === Course Definition ===
course_amplitude = 0.8
course_frequency = 0.7

def course_function(x):
    """Desired y position for given x."""
    return course_amplitude * np.sin(course_frequency * x)

# === MPC Solver Setup (prebuilt) ===
opti = ca.Opti()
y = opti.variable(1, N+1)
delta = opti.variable(1, N)
y0 = opti.parameter(1)

# Cost: penalize cross-track error and steering usage
cost = 0
for k in range(N):
    cost += w_y * y[k]**2 + w_delta * delta[k]**2
opti.minimize(cost)

# Simple lateral integrator dynamics
for k in range(N):
    opti.subject_to(y[k+1] == y[k] + delta[k] * dt)

# Steering limits and initial condition
opti.subject_to(opti.bounded(-max_steer, delta, max_steer))
opti.subject_to(y[0] == y0)

# Solver setup (single build)
opti.solver('ipopt', {'print_time': False, 'ipopt': {'print_level': 0}})

# Solve MPC for given cte
# Returns steer angle and latency
def solve_mpc(cte):
    opti.set_value(y0, cte)
    start = time.time()
    sol = opti.solve()
    latency = (time.time() - start) * 1000
    steer = float(sol.value(delta[0]))
    return steer, latency

# === Turtle Visualization ===
screen = turtle.Screen()
screen.setup(800, 600)
screen.title('Simple MPC Line Following')

# Draw reference path
path = turtle.Turtle(visible=False)
path.penup()
path.goto(-400, course_function(-4.0) * 100)
path.pendown()
for pixel_x in range(-400, 401):
    xm = pixel_x / 100.0  # scale to meters
    ym = course_function(xm) * 100
    path.goto(pixel_x, ym)

# Car turtle
car = turtle.Turtle()
car.shape('arrow')
car.color('red')
car.penup()
car.goto(-400, course_function(-4.0) * 100)
car.setheading(0)
car.pendown()

# Simulation loop
x = -4.0  # position in meters
for step in range(200):
    # Compute cross-track error
    y = car.ycor() / 100.0
    cte = y - course_function(x)

        # Compute steering via MPC
    steer, latency = solve_mpc(cte)
    # Print CTE, steering amount, and latency
    print(f"CTE={cte:.2f}, Steer={np.rad2deg(steer):.1f}°, Latency={latency:.1f}ms")
    # Print CTE and steering amount
    print(f"CTE={cte:.2f}, Steer={np.rad2deg(steer):.1f}°")

    # Apply steering and move forward
    car.setheading(np.rad2deg(steer))
    car.forward(v * 100 * dt)  # convert to pixels
    x += v * dt

    time.sleep(dt)

turtle.done()

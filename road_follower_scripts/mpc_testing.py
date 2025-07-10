#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
import numpy as np
import casadi as ca

# Parameters
N = 15
dt = 0.1
L = 0.17
v_max = 2.0
a_max = 1.0
delta_max = 0.5
Qw = [10, 10, 5, 1]
Rw = [0.1, 0.1]

waypoints = [
    [2.0, 2.0, np.pi/4, 0.0],
    [4.0, 1.0, -np.pi/2, 0.0],
    [3.0, -1.0, 0.0, 0.0]
]

tol_xy = 0.05
tol_psi = 0.05
tol_v = 0.01

current_state = np.zeros(4)
current_wp = 0
drive_pub = None

# Kinematic bicycle model
def f_kin(x, u):
    px, py, psi, v = x[0], x[1], x[2], x[3]
    a, delta = u[0], u[1]
    return ca.vertcat(
        v * ca.cos(psi),
        v * ca.sin(psi),
        v / L * ca.tan(delta),
        a
    )

# Build MPC with CasADi v3 syntax
def build_mpc_v3():
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u", 2)
    rhs = f_kin(x, u)
    f_dyn = ca.Function("f_dyn", [x, u], [rhs])

    X = [ca.SX.sym('x0', 4)]
    U = []
    cost = 0
    Q = ca.diag(ca.SX(Qw))
    R = ca.diag(ca.SX(Rw))

    for k in range(N):
        uk = ca.SX.sym('u'+str(k), 2)
        xk = ca.SX.sym('x'+str(k+1), 4)
        U.append(uk)
        X.append(xk)

    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []

    for i in range(N):
        w += [X[i+1], U[i]]
        w0 += [0]*4 + [0]*2
        lbw += [-ca.inf]*4 + [-a_max, -delta_max]
        ubw += [ca.inf]*4 + [a_max, delta_max]
        g += [X[i+1] - (X[i] + dt * f_dyn(X[i], U[i]))]
        lbg += [0]*4
        ubg += [0]*4

    # Terminal cost
    ref = ca.SX.sym("ref", 4)
    for i in range(N):
        e = X[i] - ref
        cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([U[i].T, R, U[i]])
    eN = X[N] - ref
    cost += ca.mtimes([eN.T, Q, eN])

    nlp = {'x': ca.vertcat(*w), 'f': cost, 'g': ca.vertcat(*g), 'p': ca.vertcat(X[0], ref)}
    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': False})
    return solver

solver = build_mpc_v3()

def odom_cb(msg):
    global current_state
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    quat = [q.x, q.y, q.z, q.w]
    yaw = R.from_quat(quat).as_euler('xyz')[2]
    v = msg.twist.twist.linear.x
    current_state[:] = [x, y, yaw, v]

def control_cb(event):
    global current_wp
    if current_wp >= len(waypoints):
        drive_pub.publish(Twist())
        return

    wp = np.array(waypoints[current_wp])
    p = np.concatenate((current_state, wp))

    try:
        sol = solver(x0=[], p=p, lbg=0, ubg=0)
        w_opt = sol['x'].full().flatten()
        a = w_opt[4]  # first U[0]
        delta = w_opt[5]
    except:
        rospy.logwarn("MPC solver failed")
        return

    cmd = Twist()
    cmd.linear.x = float(a)
    cmd.angular.z = float(delta)
    drive_pub.publish(cmd)

    psi_err = ((current_state[2] - wp[2] + np.pi) % (2*np.pi)) - np.pi
    if (np.linalg.norm(current_state[:2] - wp[:2]) < tol_xy and
        abs(psi_err) < tol_psi and abs(current_state[3] - wp[3]) < tol_v):
        current_wp += 1

if __name__ == '__main__':
    rospy.init_node('mpc_controller_py3')
    drive_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.Subscriber('/odom', Odometry, odom_cb)
    rospy.Timer(rospy.Duration(dt), control_cb)
    rospy.spin()

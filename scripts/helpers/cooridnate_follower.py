#!/usr/bin/env python3
"""
linear_mpc_waypoint_follower.py  — CasADi linear MPC using the built‑in
**qrqp** QP solver (no external qpoases plugin required).
"""

import casadi as ca
import numpy as np
import rospy, math
from jetracer.nvidia_racecar import NvidiaRacecar
from car_postion import CarPositionPrinter

# ── Horizon & vehicle settings ──────────────────────────────────────────
NX, NU, N = 4, 2, 15          # state, input, horizon length
DT, L     = 0.1, 0.17         # time step, wheel‑base

# bounds
V_MAX, A_MAX = 2.0, 1.0
DELTA_MAX    = math.radians(45)
DDELTA_MAX   = math.radians(25) * DT

# cost matrices (CasADi DM)
Q  = ca.diag([20, 20, 2, 0.5])
Qf = Q
R  = ca.diag([0.1, 0.02])
Rd = ca.diag([0.05, 1.0])

# block‑diagonals for horizon
Qblk = ca.diagcat(*([Q]*N + [Qf]))     # 64×64
Rblk = ca.diagcat(*([R]*N))            # 30×30

# waypoints  [x, y, yaw, v_ref]
WAYPOINTS = [
    [2.0, 2.0,  math.pi/4, 0.8],
    [4.0, 1.0, -math.pi/2, 0.8],
    [3.0, -1.0,  0.0,      0.8],
    [3.0, -1.0,  math.pi,  0.0]
]
TOL_POS, TOL_YAW, TOL_V = 0.05, 0.05, 0.05
DEADBAND, STEER_SCALE = 0.08, 0.90
pi2pi = lambda a: (a + math.pi) % (2*math.pi) - math.pi

# ----------------------------------------------------------------------

def linear_matrices(v, yaw, delta):
    """Discrete linearised bicycle model around current state."""
    A = np.eye(NX)
    A[0,3] = -DT * v * math.sin(yaw); A[0,2] = DT * math.cos(yaw)
    A[1,3] =  DT * v * math.cos(yaw); A[1,2] = DT * math.sin(yaw)
    A[2,3] =  DT * math.tan(delta) / L

    B = np.zeros((NX, NU))
    B[3,0] = DT
    B[2,1] = DT * v / (L * math.cos(delta)**2)

    C = np.zeros(NX)  # ignore 2nd‑order terms
    return A, B, C


def build_prediction(A, B, C):
    """Build horizon‑wide matrices Ā, S, c for linear MPC."""
    Abar = np.zeros((NX*(N+1), NX))
    S    = np.zeros((NX*(N+1), NU*N))
    c    = np.zeros(NX*(N+1))
    Abar[:NX, :] = np.eye(NX)

    for k in range(N):
        Abar[NX*(k+1):NX*(k+2), :] = A @ Abar[NX*k:NX*(k+1), :]
        for j in range(k+1):
            S[NX*(k+1):NX*(k+2), NU*j:NU*(j+1)] += np.linalg.matrix_power(A, k-j) @ B
        c[NX*(k+1):NX*(k+2)] = C + A @ c[NX*k:NX*(k+1)]

    return ca.DM(Abar), ca.DM(S), ca.DM(c)


def solve_lmpc(state, wp):
    """Solve linear QP for one control step."""
    x0 = ca.DM(state)
    A, B, C = linear_matrices(state[3], state[2], 0.0)
    Abar, S, cvec = build_prediction(A, B, C)
    xref = ca.DM(np.tile(wp, (N+1, 1)).flatten())

    H = 2 * (S.T @ Qblk @ S + Rblk)
    g = 2 * S.T @ Qblk @ (Abar @ x0 + cvec - xref)

    opti = ca.Opti()
    U = opti.variable(NU * N)
    opti.minimize(0.5 * ca.mtimes([U.T, H, U]) + g.T @ U)

    # input and rate constraints
    for k in range(N):
        opti.subject_to(opti.bounded(-A_MAX,     U[NU*k],   A_MAX))
        opti.subject_to(opti.bounded(-DELTA_MAX, U[NU*k+1], DELTA_MAX))
    for k in range(N-1):
        opti.subject_to(opti.bounded(-DDELTA_MAX,
                                     U[NU*(k+1)+1] - U[NU*k+1],
                                     DDELTA_MAX))

    # built‑in sparse QP solver (no external plugin needed)
    opti.solver('qrqp')
    sol = opti.solve()
    return float(sol.value(U[0])), float(sol.value(U[1]))

# ----------------------------------------------------------------------
class LinearMPCFollower:
    def __init__(self):
        rospy.init_node('linear_mpc_waypoint_follower')
        self.car = NvidiaRacecar(); self.pos = CarPositionPrinter()
        self.car.throttle_gain, self.car.steering_gain = 0.3, 1.0
        self.rate = rospy.Rate(1/DT)
        rospy.loginfo('Waiting for /odom …')
        while not rospy.is_shutdown() and self.pos.odom_msg is None:
            rospy.sleep(0.1)

    def state(self):
        x, y, yaw = self.pos.get_odom_values()
        if x is None:
            return None
        v = self.pos.odom_msg.twist.twist.linear.x
        return np.array([x, y, yaw, v])

    def steer_map(self, delta):
        norm = delta / DELTA_MAX
        if 0 < abs(norm) < DEADBAND:
            norm = math.copysign(DEADBAND, norm)
        norm *= STEER_SCALE
        return float(np.clip(norm, -1, 1))

    def run(self):
        for wp in WAYPOINTS:
            wp = np.array(wp)
            while not rospy.is_shutdown():
                st = self.state()
                if st is None:
                    self.rate.sleep(); continue

                a_cmd, d_cmd = solve_lmpc(st, wp)
                self.car.throttle = float(np.clip(a_cmd * self.car.throttle_gain, -1, 1))
                self.car.steering = self.steer_map(d_cmd)
                rospy.loginfo(f"δ={d_cmd:+.3f}  a={a_cmd:+.2f}")

                dx, dy = st[0] - wp[0], st[1] - wp[1]
                if (math.hypot(dx, dy) < TOL_POS and
                    abs(pi2pi(st[2] - wp[2])) < TOL_YAW and
                    abs(st[3] - wp[3]) < TOL_V):
                    rospy.loginfo('✓ Waypoint')
                    break

                self.rate.sleep()

        self.car.throttle = self.car.steering = 0.0
        rospy.loginfo('Mission complete')

if __name__ == '__main__':
    try:
        LinearMPCFollower().run()
    except rospy.ROSInterruptException:
        pass

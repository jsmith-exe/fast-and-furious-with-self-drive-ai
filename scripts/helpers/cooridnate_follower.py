#!/usr/bin/env python3
"""
linear_mpc_waypoint_follower.py  — Original live‑car build
================================================================
Full, un‑trimmed version that **logs every control step** via `rospy.loginfo`,
so you can see x/y/ψ, speed, a, δ, and the raw PWM commands while the car
runs.  This now exactly matches the log behaviour of your first working
track script.
"""
import casadi as ca
import numpy as np
import rospy, math, time
from jetracer.nvidia_racecar import NvidiaRacecar
from car_postion import CarPositionPrinter

# ── MPC / vehicle parameters ───────────────────────────────────────────
NX, NU     = 4, 2             # state and input dimensions
N          = 15               # horizon length
DT         = 0.10             # [s] control period (ROS rate is 1/DT)
L          = 0.17             # [m] wheel‑base (JetRacer chassis)

V_MAX      = 1.2              # [m/s] straight‑line top speed on 2‑cell LiPo
A_MAX      = 1.0              # [m/s²] longitudinal accel bound
DELTA_MAX  = math.radians(45) # [rad] mechanical steering stop
DDELTA_MAX = math.radians(25) * DT  # [rad] max steering rate per step

# Cost weights ----------------------------------------------------------
Q  = ca.diag([20, 20, 2, 0.5])   # state error
Qf = Q                            # terminal state
R  = ca.diag([0.1, 0.02])         # input effort
Rd = ca.diag([0.05, 1.0])         # input change (handled via constraint)

Qblk = ca.diagcat(*([Q]*N + [Qf]))
Rblk = ca.diagcat(*([R]*N))

# Waypoints: [x, y, yaw, v_ref] ----------------------------------------
WAYPOINTS = [
    [2.0,  2.0,  math.pi/4, 0.8],
    [4.0,  1.0, -math.pi/2, 0.8],
    [3.0, -1.0,  0.0,       0.8],
    [3.0, -1.0,  math.pi,   0.0]
]
TOL_POS, TOL_YAW, TOL_V = 0.05, 0.05, 0.05
DEADBAND, STEER_SCALE   = 0.08, 0.90
pi2pi = lambda a: (a + math.pi) % (2*math.pi) - math.pi

# ── Helpers ────────────────────────────────────────────────────────────

def linear_matrices(v, yaw, delta):
    """Linearised bicycle model (ZOH discretised)."""
    A = np.eye(NX)
    A[0,2] =  DT * math.cos(yaw)
    A[0,3] = -DT * v * math.sin(yaw)
    A[1,2] =  DT * math.sin(yaw)
    A[1,3] =  DT * v * math.cos(yaw)
    A[2,3] =  DT * math.tan(delta) / L

    B = np.zeros((NX, NU))
    B[3,0] = DT                                  # a → speed
    B[2,1] = DT * v / (L*math.cos(delta)**2)     # δ → yaw rate
    C = np.zeros(NX)
    return A, B, C


def build_prediction(A, B, C):
    """Condense to one big (Abar, S, c) for horizon N."""
    Abar = np.zeros((NX*(N+1), NX))
    S    = np.zeros((NX*(N+1), NU*N))
    cvec = np.zeros(NX*(N+1))

    Abar[:NX, :] = np.eye(NX)
    for k in range(N):
        Abar[NX*(k+1):NX*(k+2), :] = A @ Abar[NX*k:NX*(k+1), :]
        for j in range(k+1):
            S[NX*(k+1):NX*(k+2), NU*j:NU*(j+1)] += np.linalg.matrix_power(A, k-j) @ B
        cvec[NX*(k+1):NX*(k+2)] = C + A @ cvec[NX*k:NX*(k+1)]
    return ca.DM(Abar), ca.DM(S), ca.DM(cvec)


class LinearMPCFollower:
    def __init__(self):
        rospy.init_node('linear_mpc_waypoint_follower')
        self.car = NvidiaRacecar()
        self.pos = CarPositionPrinter()

        # Outer PID gains ---------------------------------------------------
        self.throttle_kp   = 0.5      # speed error → throttle
        self.throttle_bias = 0.18     # overcome static friction
        self.steering_gain = 1.0

        self.prev_U = np.zeros(NU*N)
        self.rate   = rospy.Rate(int(1/DT))

        rospy.loginfo('Waiting for /odom …')
        while not rospy.is_shutdown() and self.pos.odom_msg is None:
            rospy.sleep(0.05)

    # ------------------------------------------------------------------
    def current_state(self):
        x, y, yaw = self.pos.get_odom_values()
        if x is None:
            return None
        v = self.car.throttle * V_MAX
        return np.array([x, y, pi2pi(yaw), v])

    def steer_map(self, delta):
        norm = delta / DELTA_MAX
        if abs(norm) < DEADBAND:
            norm = math.copysign(DEADBAND, norm)
        norm *= STEER_SCALE
        return float(np.clip(norm, -1, 1))

    # ------------------------------------------------------------------
    def solve_lmpc(self, state, delta_prev, wp):
        x0 = ca.DM(state)
        A, B, C = linear_matrices(state[3], state[2], delta_prev)
        Abar, S, cvec = build_prediction(A, B, C)
        xref = ca.DM(np.tile(wp, (N+1, 1)).reshape(-1))

        H = 2*(S.T @ Qblk @ S + Rblk)
        g = 2*S.T @ Qblk @ (Abar @ x0 + cvec - xref)

        opti = ca.Opti()
        U = opti.variable(NU*N)
        opti.minimize(0.5*ca.mtimes([U.T, H, U]) + g.T@U)

        for k in range(N):
            opti.subject_to(opti.bounded(-A_MAX,     U[NU*k],   A_MAX))
            opti.subject_to(opti.bounded(-DELTA_MAX, U[NU*k+1], DELTA_MAX))
        for k in range(N-1):
            opti.subject_to(opti.bounded(-DDELTA_MAX,
                                         U[NU*(k+1)+1] - U[NU*k+1],
                                         DDELTA_MAX))
        opti.set_initial(U, self.prev_U)

        try:
            opti.solver('qrqp', {'print_iter': False})
        except RuntimeError:
            opti.solver('qpoases', {'print_level': 'none'})

        try:
            sol = opti.solve()
        except RuntimeError as e:
            rospy.logwarn(f"⚠️  MPC infeasible: {e}")
            return 0.0, 0.0, self.prev_U

        U_opt = np.array(sol.value(U)).flatten()
        self.prev_U = U_opt
        return float(U_opt[0]), float(U_opt[1]), U_opt

    # ------------------------------------------------------------------
    def run(self):
        delta_prev = 0.0
        v_des      = 0.0

        for wp in WAYPOINTS:
            wp = np.array(wp)
            rospy.loginfo(f"▶️  New waypoint: x={wp[0]:.2f}, y={wp[1]:.2f}, ψ={math.degrees(wp[2]):.1f}°, v={wp[3]:.2f}")
            while not rospy.is_shutdown():
                st = self.current_state()
                if st is None:
                    self.rate.sleep()
                    continue

                # --- 1. Solve MPC -----------------------------------
                a_cmd, d_cmd, _ = self.solve_lmpc(st, delta_prev, wp)
                delta_prev = d_cmd

                # --- 2. Outer speed loop ----------------------------
                v_des += a_cmd * DT
                v_des = max(min(v_des, V_MAX), -V_MAX)
                v_err = v_des - st[3]
                thr_raw = self.throttle_kp * v_err + math.copysign

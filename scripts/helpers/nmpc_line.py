#!/usr/bin/env python3
import numpy as np
import math
import time
import casadi as ca

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


class NMPC_Terminal:
    def __init__(self, init_pos, DT, N, W_q, W_r, W_v, W_dv):
        self.DT = DT            # time step
        self.N = N              # horizon length
        self.W_q = W_q          # Weight matrix for states
        self.W_r = W_r          # Weight matrix for controls
        self.W_v = W_v          # Weight matrix for Terminal state
        self.W_dv = W_dv   
        self.x_guess = np.ones((self.N+1, 2))*init_pos
        self.u_guess = np.zeros((self.N, 2))
        self.setup_controller()

    def setup_controller(self):
        # states: lateral deviation d, heading error e
        d = ca.SX.sym('d')
        e = ca.SX.sym('e')
        states = ca.vertcat(d, e)
        self.n_states = states.size()[0]
        vx = ca.SX.sym('vx')
        sa = ca.SX.sym('sa')
        controls = ca.vertcat(vx, sa)
        self.n_controls = controls.size()[0]
        
        # dynamics in line frame
        rhs = ca.vertcat(
            vx * ca.sin(e),
            vx * ca.tan(sa) / 0.15
        ) 
                         
        ## function
        f = ca.Function('f', [states, controls], [rhs])                                   
        
        self.U_opt = ca.SX.sym('U', self.n_controls, self.N)
        self.X_opt = ca.SX.sym('X', self.n_states, self.N+1)
        self.U_ref = ca.SX.sym('U_ref', self.n_controls, self.N)
        self.X_ref = ca.SX.sym('X_ref', self.n_states, self.N+1)

        obj = 0 
        g = [] 
        g.append(self.X_opt[:, 0]-self.X_ref[:, 0])

        for i in range(self.N):
            st_e_ = self.X_opt[:, i] - self.X_ref[:, i]
            st_e_[1] = ca.atan2(ca.sin(st_e_[1]), ca.cos(st_e_[1]))
            ct_e_ = self.U_opt[:, i] #- self.U_ref[:, i]
            if i < self.N - 1:
                d_ct = self.U_opt[:, i] - self.U_opt[:, i+1]            
            obj = obj + ca.mtimes([st_e_.T, self.W_q, st_e_]) + ca.mtimes([ct_e_.T, self.W_r, ct_e_]) + ca.mtimes([d_ct.T,self.W_dv,d_ct])
            k1 = f(self.X_opt[:, i],self.U_opt[:, i])
            k2 = f(self.X_opt[:, i] + self.DT/2*k1, self.U_opt[:, i])
            k3 = f(self.X_opt[:, i] + self.DT/2*k2, self.U_opt[:, i])
            k4 = f(self.X_opt[:, i] + self.DT*k3, self.U_opt[:, i])
            x_next = self.X_opt[:, i] + self.DT/6*(k1 + 2*k2 + 2*k3 + k4)
            g.append(self.X_opt[:, i+1] - x_next)   
        st_e_N = self.X_opt[:, self.N] - self.X_ref[:, self.N] 
        st_e_N[1] = ca.atan2(ca.sin(st_e_N[1]), ca.cos(st_e_N[1]))   
        obj = obj + ca.mtimes([st_e_N.T, self.W_v, st_e_N])

        opt_variables = ca.vertcat( ca.reshape(self.U_opt, -1, 1), ca.reshape(self.X_opt, -1, 1))
        opt_params = ca.vertcat(ca.reshape(self.U_ref, -1, 1), ca.reshape(self.X_ref, -1, 1))
        
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []

        for _ in range(self.n_states *(self.N+1)):
            self.lbg.append(0.0)
            self.ubg.append(0.0)        
        for _ in range(self.N):
            self.lbx += [0.1, -np.deg2rad(45)]
            self.ubx += [1.2, np.deg2rad(45)]
        for _ in range(self.N+1): 
            self.lbx += [-10.0, -np.inf] 
            self.ubx += [10.0, np.inf]

        nlp_prob = {'f': obj, 'x': opt_variables, 'p':opt_params, 'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter':300, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
    
    def solve(self, next_trajectories, next_controls):
        try:
            arg_p = np.concatenate((next_controls.reshape(-1, 1), next_trajectories.reshape(-1, 1)))
            arg_X0 = np.concatenate([self.u_guess.reshape(-1, 1), self.x_guess.reshape(-1, 1)], axis=0)
            sol = self.solver(x0=arg_X0, p=arg_p, lbg=self.lbg, lbx=self.lbx, ubg=self.ubg, ubx=self.ubx)
            estimated_opt = sol['x'].full()
            self.u_guess = estimated_opt[:int(self.n_controls*self.N)].reshape(self.N, self.n_controls) 
            self.x_guess = estimated_opt[int(self.n_controls*self.N):int(self.n_controls*self.N+self.n_states*(self.N+1))].reshape(self.N+1, self.n_states)
            return self.u_guess[0,:]
        except RuntimeError as e:
            rospy.logerr(f"[NMPC] Solver failed: {e}")
            return np.zeros(4)

nmpc_active = True
def nmpc_toggle_callback(msg):
    global nmpc_active
    nmpc_active = msg.data

# Feedback state callback
odom = Odometry()
def odom_callback(data):
    global odom
    odom = data

# Global reference pose
pose_ref = None  
def ref_pose_callback(msg):
    global pose_ref
    pose_ref = msg.pose

def line_callback(msg: Float32MultiArray):
    global lateral_dev, heading_err
    if len(msg.data) >= 2:
        lateral_dev = msg.data[0]
        heading_err = msg.data[1]

# Convert quaternion to euler
def quaternion2Yaw(orientation):
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    yaw = math.atan2(2.0*(w*z + x*y), 1.0-2.0*(y**2 + z**2))
    return yaw

def desired_trajectory(fb, N: int, pose_ref):
    goal = pose_ref
    ref_ = np.zeros((N+1, 2))
    ct_ = np.zeros((N, 2))
    delta_yaw = math.atan2(math.sin(goal[1] - fb[1]), math.cos(goal[1] - fb[1]))
    for i in range(N + 1):
        alpha = i / N
        ref_[i, :1] = fb[:1] * (1 - alpha) + goal[:1] * alpha
        interp_yaw = fb[1] + alpha * delta_yaw
        ref_[i, 1] = math.atan2(math.sin(interp_yaw), math.cos(interp_yaw))
    return ref_, ct_

def nmpc_node():
    rospy.init_node("nmpc_node", anonymous=True)
    rospy.Subscriber('/line/offset_yaw', Float32MultiArray, line_callback)
    pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    r = rospy.Rate(20)
    print("[INFO] Init Node...")
    print("[INFO] Wait 2s ...")    
    st = time.time()
    while time.time() - st < 2:
        r.sleep()
        continue
    print("[INFO] Start NMPC simulation!!!")

    # Create the current robot position
    pos_fb = np.array([lateral_dev, heading_err])
    DT = 0.1
    N = 20  
    W_q = np.diag([200, 1])  
    W_r = np.diag([1, 1])  
    W_v = 10**2*np.diag([1, 1]) 
    W_dv = np.diag([100, 100])  
    nmpc = NMPC_Terminal(pos_fb, DT, N, W_q, W_r, W_v, W_dv)

    # compute reference
    target = np.array([0, 0])

    while not rospy.is_shutdown():
        if lateral_dev is None:
            rospy.logwarn_throttle(5, "[NMPC] Waiting for Line Measurements...")
            r.sleep()
            continue

        # Create the current robot position
        pos_fb = np.array([lateral_dev, heading_err])

        next_traj, next_cons = desired_trajectory(pos_fb, N, target)
        st = time.time()
        vel = nmpc.solve(next_traj, next_cons)
        #print("Processing time: {:.2f}s".format(time.time()-st))

        # rospy.loginfo(f"[NMPC] x_fb={x_fb:.2f}, y_fb={y_fb:.2f}, yaw_fb={np.rad2deg(yaw_fb):.1f}°")
        # rospy.loginfo(f"[NMPC] x_ref={x_ref:.2f}, y_ref={y_ref:.2f}, yaw_ref={np.rad2deg(yaw_ref):.1f}°")
        # rospy.loginfo(f"[NMPC] error={error:.3f} m, error_angle={np.rad2deg(error_angle):.1f}°")
        vel_msg = Twist()
        V_MIN = 0.1
        
        if abs(vel[0]) < V_MIN:
            vel[0] = V_MIN if vel[0] > 0 else -V_MIN
        
        # angle_deg_arr = np.rad2deg(next_traj[2])   # still an array
        # angle_deg = float(angle_deg_arr[0])          # now a plain Python float
        # rospy.loginfo("[NMPC] Running: angle=%.3fm", angle_deg)

        rospy.loginfo("[NMPC] Running: Lateral Error=%.3fm, Yaw Error =%.3f°, V=%.2f, ω=%.2f", lateral_dev, np.rad2deg(heading_err), vel[0], vel[1])

        vel_msg.linear.x = vel[0]
        vel_msg.angular.z = vel[1]
        # DEBUG
        # vel_msg.linear.x = 0.0
        # vel_msg.angular.z = 0.0
            
        pub_vel.publish(vel_msg)
 
        r.sleep()
if __name__ == '__main__':    
    try:
        nmpc_node()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("❎ Interrupted — stopping motors")
    finally:
        rospy.loginfo("Motors set to zero. Exiting.")
        
        

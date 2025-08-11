'''
This is my Non-linear MPC class.
'''

import numpy as np
import casadi as ca
import rospy

class NMPC:
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
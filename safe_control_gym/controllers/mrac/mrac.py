import os
import warnings

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
import scipy.linalg
from munch import munchify

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.envs.benchmark_env import Cost, Task
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics
from safe_control_gym.controllers.lqr import lqr_utils


class MRAC(BaseController):
    def __init__(self,
                 env_func,
                 q_lqr=[1],
                 r_lqr=[1],
                 discrete_dynamics=1,
                 task: Task = Task.STABILIZATION,
                 task_info=None,
                 episode_len_sec=10,
                 ctrl_freq=240,
                 pyb_freq=240,
                 output_dir="results/temp",
                 save_data=False,
                 data_dir=None,
                 plot_traj=False,
                 plot_dir=None,
                 save_plot=False,
                 **kwargs):
        self.task = task
        self.env_func = env_func
        self.episode_len_sec = episode_len_sec
        self.task_info = task_info
        self.discrete_dynamics = discrete_dynamics

        self.env = env_func(cost=Cost.QUADRATIC,
                            randomized_inertial_prop=False,
                            episode_len_sec=episode_len_sec,
                            task=task,
                            task_info=task_info,
                            ctrl_freq=ctrl_freq,
                            pyb_freq=pyb_freq)
        self.env = RecordEpisodeStatistics(self.env)

        # Controller parameters

        self.q_lqr = q_lqr
        self.r_lqr = r_lqr

        self.model = self.env.symbolic
        self.Q_lqr = lqr_utils.get_cost_weight_matrix(self.q_lqr, self.model.nx)
        self.R_lqr = lqr_utils.get_cost_weight_matrix(self.r_lqr, self.model.nu)
        self.env.set_cost_function_param(self.Q_lqr, self.R_lqr)
        self.x_0, self.u_0 = self.env.X_GOAL, self.env.U_GOAL

        lqr_gain = lqr_utils.compute_lqr_gain(self.model, self.x_0[0], self.u_0,
                                              self.Q_lqr, self.R_lqr, self.discrete_dynamics)
        self.k_x = -lqr_gain
        self.k_r = lqr_gain

        feature_dim = 1
        self.w_x = np.zeros((feature_dim, self.env.action_dim))

        df = self.model.df_func(self.x_0[0], self.u_0)
        self.A, self.B = df[0].toarray(), df[1].toarray()

        self.Am = np.zeros(self.env.state_dim)
        self.Bm = np.zeros((self.env.state_dim, self.env.action_dim))
        self.Q_ly = np.eye(self.env.state_dim)

        self.gamma_x = 0.01
        self.gamma_r = 0.01
        self.gamma_w = 0.01

        self.reference_state = np.zeros((self.env.state_dim, 1))

        self.k = 0
        self.stepsize = self.model.dt

        self.max_iterations = episode_len_sec / self.stepsize

        # Plotting/output variables

        self.plot_traj = plot_traj
        self.save_plot = save_plot

        # Plot output directory.
        self.plot_dir = plot_dir
        if self.plot_dir:
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
        else:
            self.save_plot = False
            warnings.warn('save_plot is True but no plot directory specified; not saving')

        self.save_data = save_data

        # Data output directory.
        self.data_dir = data_dir
        if self.data_dir:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
        else:
            self.save_data = False
            warnings.warn('save_data is True but no data directory specified; not saving')

    def phi(self, x):
        # return np.array([[1, *x]]).T
        return np.array([[1]])

    def update_reference_state(self):
        # For the reference model, we follow the disturbance-free LQR trajectory
        k_lqr = lqr_utils.compute_lqr_gain(self.model, self.x_0[self.k], self.u_0,
                                           self.Q_lqr, self.R_lqr, self.discrete_dynamics)

        df = self.model.df_func(self.x_0[self.k], self.u_0)
        self.A, self.B = df[0].toarray(), df[1].toarray()

        self.Am = self.A - self.B @ k_lqr
        self.Bm = self.B @ k_lqr

        x_dot = self.Am @ self.reference_state + self.Bm @ self.x_0[self.k] + self.B @ self.u_0
        self.reference_state += x_dot * self.stepsize

    def select_action(self, x):
        u_ad = (self.w_x.T @ self.phi(x)).T
        u_t = self.k_x @ x + self.k_r @ self.x_0[self.k] + self.u_0 - u_ad

        return u_t

    def update_parameters(self, x):
        if self.discrete_dynamics:
            P = scipy.linalg.solve_discrete_lyapunov(self.Am, self.Q_ly)
        else:
            P = scipy.linalg.solve_continuous_lyapunov(self.Am, self.Q_ly)

        err = x - self.reference_state

        e_P_B = err[:, None].T @ P @ self.B

        k_lqr = lqr_utils.compute_lqr_gain(self.model, self.x_0[self.k], self.u_0,
                                           self.Q_lqr, self.R_lqr, self.discrete_dynamics)
        self.k_x = -k_lqr
        self.k_r = k_lqr

        # self.k_x += self.stepsize * self.gamma_x * (x[:, None] @ e_P_B).T
        # self.k_r += self.stepsize * self.gamma_r * (self.x_0[self.k][:, None] @ e_P_B).T
        self.w_x += self.stepsize * self.gamma_w * (self.phi(x) @ e_P_B)

    def run(self, **kwargs):
        self.env.reset()

        ep_returns, ep_lengths = [], []

        self.k = 0
        self.reference_state = self.env.state

        while self.k < self.max_iterations:
            # Current goal.
            if self.task == Task.STABILIZATION:
                current_goal = self.x_0
            elif self.task == Task.TRAJ_TRACKING:
                current_goal = self.x_0[self.k]

            # Select action.
            action = self.select_action(self.env.state)

            # Save initial condition.
            if self.k == 0:
                # Initialize state and input stack.
                state_stack = self.env.state
                input_stack = action
                goal_stack = current_goal
                reference_stack = self.reference_state

            else:
                # Save state and input.
                state_stack = np.vstack((state_stack, self.env.state))
                input_stack = np.vstack((input_stack, action))
                goal_stack = np.vstack((goal_stack, current_goal))
                reference_stack = np.vstack((reference_stack, self.reference_state))

            # Step forward.
            obs, reward, done, info = self.env.step(action)

            # Perform parameter adaptation
            self.update_reference_state()
            self.update_parameters(obs)

            # Update step counter
            self.k += 1

            if done:
                # Push last state and input to stack.
                # Note: the last input is not used.
                state_stack = np.vstack((state_stack, self.env.state))
                input_stack = np.vstack((input_stack, action))
                goal_stack = np.vstack((goal_stack, current_goal))
                reference_stack = np.vstack((reference_stack, self.reference_state))

                # Post analysis.
                if self.plot_traj or self.save_plot or self.save_data:
                    analysis_data = lqr_utils.post_analysis(reference_stack, state_stack,
                                                            input_stack, self.env, 0, 0,
                                                            self.plot_traj,
                                                            self.save_plot,
                                                            self.save_data,
                                                            self.plot_dir, self.data_dir)
                    ep_rmse = np.array([analysis_data["state_rmse_scalar"]])

                ep_returns.append(info["episode"]["r"])
                ep_lengths.append(info["episode"]["l"])

                break

        eval_results = {"ep_returns": ep_returns, "ep_lengths": ep_lengths}
        return eval_results

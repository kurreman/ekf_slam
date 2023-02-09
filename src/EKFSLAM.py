from math import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def observation(x):
    # assume range bearing measurement to landmark
    return np.array([sqrt((x[3] - x[0])**2 + (x[4] - x[1])**2), atan2(x[4] - x[1], x[3] - x[0]) - x[2]])


def H_jac(x):
    # jacobian of observation model
    phi = (x[3] - x[0])**2 + (x[4] - x[1])**2
    return (1 / phi) * np.array([[sqrt(phi) * (x[3] - x[0]), - sqrt(phi) * (x[4] - x[1]), 0, - sqrt(phi) * (x[3] - x[0]), sqrt(phi) * (x[4] - x[1])],
                                 [(x[4] - x[1]), (x[3] - x[0]), -1, - (x[4] - x[1]), - (x[3] - x[0])]])

def observation_direct(x):
    # assume that we get the local landmark measurement
    return np.array([x[3] - x[0], x[4] - x[1]])

def H_direct(x):
    return np.array([[-1, 0., 0., 1., 0.],
                     [0., -1., 0., 0., 1.]])


def motion(x, u, dt):
    # assume velocity based motion model: control input u = [v, w] where v and w are trans. and angular velocity
    return x + dt * np.array([u[0] * cos(x[2]), u[0] * sin(x[2]), u[1], 0., 0.])


def F_jac(x, u, dt):
    # jacobian of motion model
    return np.array([[1., 0., - u[0] * dt * sin(x[2]), 0., 0.],
                     [0., 1., + u[0] * dt * cos(x[2]), 0., 0.],
                     [0., 0., 1., 0., 0.],
                     [0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 1.]])


class EKFSLAM:
    def __init__(self, x0):
        # ---------------------------------------------------------------------
        # ---------------------- initialization -------------------------------
        # ---------------------------------------------------------------------

        # Initialize visualization
        self.visualize = False
        self.init_plot = 1
        self.timer = 0
        self.lm_plot = []
        self.lm_plot_cov = []
        self.robot_pos_cov = []

        # initialize subscribers
        self.landmarks_detected = []
        self.odometry = []
        self.pitch = []

        # initialize Kalman Angle
        self.KalmanAngle = 0

        # set number of landmarks
        self.n_landmarks = 1

        # initialize state (Assuming 2D pose of robot + only positions of landmarks)
        self.x = np.hstack([x0, 1.2,1.9])#np.zeros(2 * self.n_landmarks)])

        # intialize covariance matrix (robot pose: certain, landmark positions: uncertain)
        # self.cov = 10000 * np.ones((5, 5))
        # self.cov[0:3, 0:3] = np.zeros((3, 3))

        diag = [0.02, 0.02, 0.1,10000,10000]
        self.cov = np.diag(diag)

        # process noise (motion_model)
        self.Q = np.diag([0.005, 0.005, 0.002, 0., 0.])

        # measurement noise (single observation)
        self.R = np.diag([0.1, 0.01])

    # ---------------------------------------------------------------------
    # ---------------------- prediction step-------------------------------
    # ---------------------------------------------------------------------

    def predict(self, u, dt):
        # integrate system dynamics for mean vector
        mu = motion(self.x, u, dt)

        # linearized motion model
        F = F_jac(self.x, u, dt)

        self.cov = F @ self.cov @ F.T + dt * self.Q
        self.x = mu

    # ---------------------------------------------------------------------
    # ---------------------- correction step ------------------------------
    # ---------------------------------------------------------------------

    def update(self, z, id=0):
        # measurement z = [r, phi] (distance to landmark and angle expressed in local frame

        if id not in self.landmarks_detected:
            self.landmarks_detected.append(id)
            self.x[3] = self.x[0] + cos(self.x[2] + z[1]) * z[0]
            self.x[4] = self.x[1] + sin(self.x[2] + z[1]) * z[0]

        # predict measurement using observation model
        z_hat = observation_direct(self.x)

        # jacobian of observation model
        H = H_direct(self.x)

        # compute Kalman gain
        K = self.cov @ H.T @ np.linalg.inv(H @ self.cov @ H.T + self.R)

        # correct state
        self.x += K.dot(z - z_hat)
        self.cov = (np.eye(5) - K @ H) @ self.cov
        print(self.cov)

    # ---------------------------------------------------------------------
    # ---------------------- Visualizations -------------------------------
    # ---------------------------------------------------------------------

    def visualize_positions(self, ground_truth=None):
        if self.init_plot == 1:
            plt.ion()
            self.ax = plt.figure().add_subplot(111)
            self.pos_robot, = plt.plot(self.x[0], self.x[1], 'ro', markersize=10)
            self.robot_pos_cov = self.draw_covariance_ellipse(self.x[0:2], self.x[2],
                                                              self.cov[0:2, 0:2], self.ax)

            for j in range(self.n_landmarks):
                pos_lm = self.x[(3 + 2 * j):(3 + 2 * j + 2)]
                # plot position of landmarks
                lm, = plt.plot(pos_lm[0], pos_lm[1], 'bo')
                self.lm_plot.append(lm)
                self.lm_plot_cov.append(self.draw_covariance_ellipse(pos_lm, 0, self.cov[3:5, 3:5], self.ax))

            self.init_plot = 0
            plt.xlim((-2, 10))
            plt.grid(color='gray', linestyle='-', linewidth=1)
            plt.ylim((-3, 3))

        self.pos_robot.set_xdata(self.x[0])
        self.pos_robot.set_ydata(self.x[1])
        try:
            self.robot_pos_cov.remove()
        except ValueError:
            pass
        self.robot_pos_cov = self.draw_covariance_ellipse(self.x[0:2], self.x[2], self.cov[0:2, 0:2], self.ax)

        for jj in range(self.n_landmarks):
            lm_ = self.lm_plot[jj]
            lm_.set_xdata(self.x[3 + 2 * jj])
            lm_.set_ydata(self.x[3 + 2 * jj + 1])
            cov_lm = self.cov[(3 + 2 * jj):(3 + 2 * jj + 2), (3 + 2 * jj):(3 + 2 * jj + 2)]
            try:
                self.lm_plot_cov[jj].remove()
            except ValueError:
                pass
            self.lm_plot_cov[jj] = self.draw_covariance_ellipse(self.x[(3 + 2 * jj):(3 + 2 * jj+2)], 0, cov_lm, self.ax)
        if ground_truth:
            self.ax.plot([g[0] for g in ground_truth], [g[1] for g in ground_truth], color="red")
        plt.draw()
        plt.pause(0.0000000001)

    def draw_covariance_ellipse(self, pos, yaw, cov, ax):
        lambda1 = (cov[0, 0] + cov[1, 1]) / 2 + sqrt(np.power((cov[0, 0] - cov[1, 1]) / 2, 2) + np.power(cov[0, 1], 2))
        lambda2 = (cov[0, 0] + cov[1, 1]) / 2 - sqrt(np.power((cov[0, 0] - cov[1, 1]) / 2, 2) + np.power(cov[0, 1], 2))
        try:
            elps = Ellipse(pos, 3 * sqrt(lambda1), 3 * sqrt(lambda2), yaw, alpha=0.5, facecolor='pink', edgecolor='black')
            ax.add_patch(elps)
            return elps
        except ValueError:
            pass





from math import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from SAM import *


def observation_orientation(x):
    # include orientation of landmark
    return np.array([cos(x[2]) * (x[6] - x[0]) + sin(x[2]) * (x[7] - x[1]),
                     -sin(x[2]) * (x[6] - x[0]) + cos(x[2]) * (x[7] - x[1]),
                     x[8] - x[2]])

def H_orientation(x):
    return np.array([[-cos(x[2]), -sin(x[2]), -sin(x[2]) * (x[6] - x[0]) + cos(x[2]) * (x[7] - x[1]), 0, 0, 0, cos(x[2]), sin(x[2]), 0],
                     [sin(x[2]), -cos(x[2]), -cos(x[2]) * (x[6] - x[0]) - sin(x[2]) * (x[7] - x[1]), 0, 0, 0, -sin(x[2]), cos(x[2]), 0],
                     [0, 0, -1, 0, 0, 0, 0, 0, 1]])


class EKFSLAM:
    def __init__(self, x0):
        # ---------------------------------------------------------------------
        # ---------------------- initialization -------------------------------
        # ---------------------------------------------------------------------

        self.sam = SAM()

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
        self.x = np.hstack([x0, np.zeros(3 * self.n_landmarks)])

        self.cov = np.diag(np.hstack([np.zeros(6), 1000 * np.ones(3 * self.n_landmarks)]))

        # process noise (motion_model)
        self.Q = 0.5 * np.diag([0.01, 0.01, 0.002, 0.01, 0.01, 0.01, 0., 0., 0.])

        # measurement noise (single observation)
        self.R = np.diag([0.1, 0.1, 0.05])

    # ---------------------------------------------------------------------
    # ---------------------- prediction step-------------------------------
    # ---------------------------------------------------------------------

    def predict(self, u, dt):
        # integrate system dynamics for mean vector
        mu = np.hstack([self.sam.motion(self.x.tolist()[0:6], u, dt), self.x[6:9]])

        # linearized motion model
        F = np.block([[self.sam.jacF(self.x.tolist()[0:6], u, dt), np.zeros((6, 3))], [np.zeros((3, 6)), np.eye(3)]])

        if dt >= 1:
            dt = 0.
        self.cov = np.matmul(F, np.matmul(self.cov, F.T)) + dt * self.Q
        self.x = mu

    # ---------------------------------------------------------------------
    # ---------------------- correction step ------------------------------
    # ---------------------------------------------------------------------

    def update(self, z, id=0):
        if id not in self.landmarks_detected:
            self.landmarks_detected.append(id)
            self.x[6] = self.x[0] + cos(self.x[2]) * z[0] - sin(self.x[2]) * z[1]
            self.x[7] = self.x[1] + sin(self.x[2]) * z[0] + cos(self.x[2]) * z[1]
            self.x[8] = self.x[2] + z[2]
            self.cov[6, 6] = 0.1
            self.cov[7, 7] = 0.1
            self.cov[8, 8] = 0.05

        # predict measurement using observation model
        z_hat = observation_orientation(self.x)

        # jacobian of observation model
        H = H_orientation(self.x)

        # compute Kalman gain
        K = np.matmul(self.cov, np.matmul(H.T, np.linalg.inv(np.matmul(H, np.matmul(self.cov, H.T)) + self.R)))

        # correct state
        self.x += K.dot(z - z_hat)
        self.cov = np.matmul((np.eye(9) - np.matmul(K, H)), self.cov)
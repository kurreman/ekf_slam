from EKFSLAM import *
from copy import copy
import time

# simple simulation

# initial state
x0 = np.array([0., 0., 0.])
x_true = np.random.multivariate_normal(np.zeros(3), np.diag([0.01, 0.01, 0.01]))

# unknown landmark position
l = [1., 2.]

# camera FOV
r_max = 3.
phi_max = 90 * pi / 180

# timestep
dt = 0.1

ekf = EKFSLAM(x0)

controls = [0.1, 0.0]
ground_truth = []

for i in range(1000):
    ground_truth.append(copy(x_true))
    x_true = motion(np.hstack([x_true, l]), controls, dt)[0:3] + np.random.multivariate_normal(np.zeros(3), dt * 0.1 * ekf.Q[0:3, 0:3])
    ekf.predict(controls, dt)

    meas = observation_direct(np.hstack([x_true, np.array(l)])) + np.random.multivariate_normal(np.zeros(2), ekf.R)

    # only update belief is landmark is in FOV
    if np.linalg.norm(meas - x_true[0:2]) <= r_max:# and abs(meas[1]) <= phi_max:
        print("Measurement: " + str(meas))
        ekf.update(meas, 1)
    ekf.visualize_positions(ground_truth)
    # time.sleep(0.1)


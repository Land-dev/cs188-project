"""
Simulated localization: EKF pose estimator and sensor noise models.

The drone does not get perfect state. It receives:
- Noisy position (e.g. simulated GPS): used to correct the EKF.
- Optionally noisy lidar ranges: obstacles are placed at perturbed distances.

All planning and mapping use the estimated pose; the true pose is only used
inside the physics sim and to generate the noisy measurements.
"""
import numpy as np


def add_position_noise(true_xy, std=0.08):
    """Simulate noisy position measurement (e.g. GPS). Returns (x, y) with additive Gaussian noise."""
    return true_xy + np.random.normal(0, std, size=2).astype(np.float64)


def add_range_noise(true_range, std=0.03):
    """Simulate noisy lidar range. Returns max(0, true_range + noise)."""
    r = true_range + np.random.normal(0, std)
    return max(0.0, float(r))


class EKFLocalization:
    """
    Extended Kalman filter for 2D position and velocity.
    State: mu = [x, y, vx, vy]. Measurement: noisy (x, y) position.
    """

    def __init__(self, init_xy, position_noise_std=0.08, process_noise_scale=1.0):
        """
        Args:
            init_xy: Initial position [x, y] (e.g. from first noisy measurement or known start).
            position_noise_std: Std of position measurement noise (meters); sets Q.
            process_noise_scale: Scale factor for process noise (higher = more drift between measurements).
        """
        self.mu = np.array([init_xy[0], init_xy[1], 0.0, 0.0], dtype=float)
        # Covariance: small initial uncertainty in position, larger in velocity
        self.P = np.diag([position_noise_std**2, position_noise_std**2, 0.5**2, 0.5**2]).astype(float)
        self.Q = np.diag([position_noise_std**2, position_noise_std**2]).astype(float)  # measurement noise
        # Process noise: position and velocity random walk
        self.R_process = np.diag([
            0.01 * process_noise_scale,
            0.01 * process_noise_scale,
            0.1 * process_noise_scale,
            0.1 * process_noise_scale,
        ]).astype(float)
        self._dt = 1.0 / 48.0  # default control period

    def predict(self, dt):
        """Constant-velocity prediction. Call every control step."""
        x, y, vx, vy = self.mu
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self.mu = F @ self.mu
        # Process noise: Q_k = F not used for R; we add R_process per step
        self.P = F @ self.P @ F.T + self.R_process * (dt / self._dt)

    def update(self, z_xy):
        """Correct with noisy position measurement z_xy = [x_meas, y_meas]."""
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        z = np.array(z_xy, dtype=float)
        z_pred = H @ self.mu
        y = z - z_pred
        S = H @ self.P @ H.T + self.Q
        K = self.P @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    def step(self, dt, z_xy):
        """Predict then update with new measurement. Call once per control step."""
        self.predict(dt)
        self.update(z_xy)

    def get_position(self):
        """Estimated (x, y). Use this for planning and mapping logic."""
        return self.mu[:2].copy()

    def get_velocity(self):
        """Estimated (vx, vy)."""
        return self.mu[2:4].copy()

    def get_pose_covariance_xy(self):
        """2x2 position covariance for uncertainty visualization."""
        return self.P[:2, :2].copy()

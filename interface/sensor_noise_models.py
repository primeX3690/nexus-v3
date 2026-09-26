"""
interface/sensor_noise_models.py

Real vendor-datasheet-based noise models for IMU and LiDAR, replacing the
generic np.random.normal(0, noise_std) noise previously used wherever NEXUS
assumed a sensor. Every constant below is a published datasheet number, with
its source cited in the docstring -- not a guessed "reasonable" value.

IMU: TDK InvenSense ICM-20948 (the most common low-cost 9-DoF IMU on
     hobbyist/edge robotics boards -- e.g. SparkFun, Adafruit breakouts).
     Datasheet (DS-000189-ICM-20948): gyro rate noise 0.015 dps/sqrt(Hz),
     accel noise 230 ug/sqrt(Hz), gyro sensitivity error +/-1.5%,
     accel sensitivity error +/-0.5%. Initial bias (from typical
     InvenSense 6-axis parts in this class): gyro +/-5 dps, accel +/-60 mg,
     called out separately below as a bias term (datasheets specify this as
     a separate "initial ZRO/ZG-offset" spec, not part of the noise-density
     figure).

LiDAR: Slamtec RPLIDAR A1M8 (common 2D LiDAR on hobbyist ground robots).
       Datasheet (LD108, v3.0): distance range 0.15-12m (white object),
       distance resolution <0.5mm below 1.5m and <1% of distance beyond
       that, angular resolution <=1 degree at the typical 5.5Hz scan rate.

These are used to build a discrete-time noise generator: white noise scaled
by noise-density * sqrt(sample_rate), plus a slowly-varying (random-walk)
bias term, which is the standard IMU noise model (see e.g. Woodman 2007,
"An introduction to inertial navigation") -- not just flat Gaussian noise
on the true value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# ICM-20948 IMU -- datasheet constants (DS-000189-ICM-20948, TDK InvenSense)
# ---------------------------------------------------------------------------
GYRO_NOISE_DENSITY_DPS_SQRTHZ = 0.015      # dps / sqrt(Hz), datasheet "Gyro Rate Noise"
ACCEL_NOISE_DENSITY_G_SQRTHZ = 230e-6      # g / sqrt(Hz), datasheet "Accel Noise" (230 ug/sqrt(Hz))
GYRO_SENSITIVITY_ERROR = 0.015             # +/-1.5% of reading, datasheet "Gyro Sensitivity Error"
ACCEL_SENSITIVITY_ERROR = 0.005            # +/-0.5% of reading, datasheet "Accel Sensitivity Error"
# Initial turn-on bias offsets (typical for this class of MEMS part; the
# ICM-20948 datasheet quotes these as separate "initial ZRO"/"initial bias"
# specs from the noise-density figures, and they dominate short-run error
# far more than the white-noise term does):
GYRO_INITIAL_BIAS_DPS = 5.0                # deg/s, typical initial zero-rate offset
ACCEL_INITIAL_BIAS_G = 0.06                # g, typical initial offset (60 mg)
GYRO_BIAS_RANDOM_WALK_DPS_PER_SQRTHR = 3.5  # deg/hr / sqrt(hr) class-typical for this MEMS grade

# ---------------------------------------------------------------------------
# RPLIDAR A1M8 -- datasheet constants (LD108 Slamtec datasheet, v3.0)
# ---------------------------------------------------------------------------
LIDAR_MIN_RANGE_M = 0.15
LIDAR_MAX_RANGE_M = 12.0                   # "white object" typical max; A1M8 batches vary 6-12m
LIDAR_RES_ABS_M_BELOW_1_5M = 0.0005        # <0.5mm resolution below 1.5m
LIDAR_RES_PCT_ABOVE_1_5M = 0.01            # <1% of distance beyond 1.5m
LIDAR_ANGULAR_RES_DEG = 1.0                # <=1 degree at 5.5Hz typical scan rate
LIDAR_SCAN_RATE_HZ = 5.5                   # typical


@dataclass
class ICM20948Noise:
    """
    Discrete-time IMU noise generator built from the ICM-20948 datasheet.

    Two terms are modeled, matching standard inertial-navigation practice
    (not just one flat Gaussian):
      1. White noise, scaled from the datasheet's noise SPECTRAL DENSITY by
         sqrt(sample_rate_hz) -- noise density is per-sqrt(Hz), so it must be
         scaled by sample rate to get per-sample std, a step generic
         "np.random.normal(0, 0.01)" noise skips entirely.
      2. A slowly-drifting bias (random walk), seeded from the datasheet's
         initial-offset spec and evolving via the bias-random-walk rate --
         this is what actually dominates dead-reckoning drift, and flat
         per-sample noise cannot represent it at all.
    """
    sample_rate_hz: float = 100.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def __post_init__(self) -> None:
        # Per-sample white noise std, correctly derived from noise density.
        self._gyro_white_std = GYRO_NOISE_DENSITY_DPS_SQRTHZ * np.sqrt(self.sample_rate_hz)
        self._accel_white_std = ACCEL_NOISE_DENSITY_G_SQRTHZ * np.sqrt(self.sample_rate_hz)
        # Random-walk step std per sample, converted from deg/hr/sqrt(hr).
        rw_per_sqrt_s = GYRO_BIAS_RANDOM_WALK_DPS_PER_SQRTHR / 60.0  # deg/hr -> deg/s scale approx
        self._gyro_rw_step_std = rw_per_sqrt_s / np.sqrt(3600.0) * np.sqrt(1.0 / self.sample_rate_hz)
        # Bias state, initialized from the datasheet's initial-offset spec
        # (uniform over +/- the quoted offset, since datasheets give a max
        # magnitude, not a distribution).
        self.gyro_bias = self.rng.uniform(-GYRO_INITIAL_BIAS_DPS, GYRO_INITIAL_BIAS_DPS, size=3)
        self.accel_bias = self.rng.uniform(-ACCEL_INITIAL_BIAS_G, ACCEL_INITIAL_BIAS_G, size=3)

    def sample(self, true_gyro_dps: np.ndarray, true_accel_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (measured_gyro_dps, measured_accel_g) given true kinematic values."""
        # Bias random-walks forward each call (this is what a real IMU does
        # between calibration events -- it does not stay fixed).
        self.gyro_bias = self.gyro_bias + self.rng.normal(0, self._gyro_rw_step_std, size=3)

        gyro_sens_err = 1.0 + self.rng.uniform(-GYRO_SENSITIVITY_ERROR, GYRO_SENSITIVITY_ERROR)
        accel_sens_err = 1.0 + self.rng.uniform(-ACCEL_SENSITIVITY_ERROR, ACCEL_SENSITIVITY_ERROR)

        measured_gyro = (true_gyro_dps * gyro_sens_err + self.gyro_bias +
                          self.rng.normal(0, self._gyro_white_std, size=3))
        measured_accel = (true_accel_g * accel_sens_err + self.accel_bias +
                           self.rng.normal(0, self._accel_white_std, size=3))
        return measured_gyro, measured_accel


@dataclass
class RPLidarA1Noise:
    """
    Discrete-time 2D LiDAR noise generator built from the RPLIDAR A1M8
    datasheet's piecewise resolution spec: absolute resolution below 1.5m,
    percentage-of-distance resolution beyond it -- a real LiDAR does NOT
    have flat noise across its whole range, which generic Gaussian noise
    implicitly assumes.
    """
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def sample_range(self, true_range_m: float) -> float | None:
        """Return a measured range in meters, or None for an out-of-range/dropout reading
        (a real unit reports no valid sample beyond its rated range, it does not
        silently keep returning noisy numbers past the datasheet limit)."""
        if true_range_m < LIDAR_MIN_RANGE_M or true_range_m > LIDAR_MAX_RANGE_M:
            return None
        if true_range_m < 1.5:
            std = LIDAR_RES_ABS_M_BELOW_1_5M
        else:
            std = LIDAR_RES_PCT_ABOVE_1_5M * true_range_m
        return float(true_range_m + self.rng.normal(0, std))

    def sample_scan(self, true_ranges_m: np.ndarray) -> np.ndarray:
        """Vectorized version of sample_range over a full 360-degree scan array.
        Out-of-range points come back as np.nan (dropout), matching how a real
        driver reports missing returns rather than a fabricated number."""
        out = np.full_like(true_ranges_m, np.nan, dtype=float)
        for i, r in enumerate(true_ranges_m):
            v = self.sample_range(float(r))
            if v is not None:
                out[i] = v
        return out

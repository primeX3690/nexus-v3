"""
tests/test_sensor_noise_models.py

Verifies interface/sensor_noise_models.py's ICM-20948 and RPLIDAR A1M8
noise models behave as their datasheets specify - not just that they
run without crashing.
"""
from __future__ import annotations

import numpy as np
import pytest

from interface.sensor_noise_models import (
    ICM20948Noise, RPLidarA1Noise,
    LIDAR_MIN_RANGE_M, LIDAR_MAX_RANGE_M,
    GYRO_INITIAL_BIAS_DPS, ACCEL_INITIAL_BIAS_G,
)
from brain.perception.sensor_array import DirectionTunedSensorArray


def test_lidar_out_of_range_reports_dropout_not_a_number():
    lidar = RPLidarA1Noise(rng=np.random.default_rng(0))
    assert lidar.sample_range(LIDAR_MIN_RANGE_M - 0.01) is None
    assert lidar.sample_range(LIDAR_MAX_RANGE_M + 0.01) is None


def test_lidar_noise_is_tighter_close_range_than_far_range():
    """Datasheet spec is piecewise: <0.5mm below 1.5m, <1% of distance
    beyond - so noise std at 1m must be much smaller in absolute terms
    than noise std at 10m, not flat across the whole range."""
    lidar = RPLidarA1Noise(rng=np.random.default_rng(1))
    near_samples = [lidar.sample_range(1.0) for _ in range(500)]
    far_samples = [lidar.sample_range(10.0) for _ in range(500)]
    near_std = np.std(near_samples)
    far_std = np.std(far_samples)
    assert near_std < 0.01, f"expected sub-cm noise at 1m, got std={near_std}"
    assert far_std > near_std, "far-range noise must exceed near-range noise (1% of distance vs <0.5mm)"
    # 1% of 10m = 0.1m -> expect far_std in the right ballpark, not orders of magnitude off
    assert 0.03 < far_std < 0.3


def test_imu_bias_is_within_datasheet_initial_offset_bounds():
    imu = ICM20948Noise(rng=np.random.default_rng(2))
    assert np.all(np.abs(imu.gyro_bias) <= GYRO_INITIAL_BIAS_DPS)
    assert np.all(np.abs(imu.accel_bias) <= ACCEL_INITIAL_BIAS_G)


def test_imu_white_noise_scales_with_sample_rate():
    """Noise DENSITY (dps/sqrt(Hz)) must be scaled by sqrt(sample_rate) to
    get per-sample std - a flat noise_std would get this wrong. Higher
    sample rate -> higher per-sample white-noise std, at the same density."""
    slow = ICM20948Noise(sample_rate_hz=10.0, rng=np.random.default_rng(3))
    fast = ICM20948Noise(sample_rate_hz=1000.0, rng=np.random.default_rng(3))
    assert fast._gyro_white_std > slow._gyro_white_std
    ratio = fast._gyro_white_std / slow._gyro_white_std
    assert abs(ratio - np.sqrt(100.0)) < 1e-6


def test_sensor_array_datasheet_noise_matches_contract_of_original_sense():
    """sense_with_datasheet_noise must return the same shape/range contract
    as the original sense() method, so it's a drop-in upgrade, not a
    breaking change."""
    sa = DirectionTunedSensorArray(n_sensors=12, max_range=10.0)
    readings = sa.sense_with_datasheet_noise(np.array([3.0, 0.0]))
    assert readings.shape == (12,)
    assert np.all(readings >= 0.0) and np.all(readings <= 1.0)
    # sensor facing the object (index 0, direction (1,0)) should read
    # meaningfully higher than one facing directly away
    facing_idx = 0
    away_idx = 6
    assert readings[facing_idx] > readings[away_idx]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

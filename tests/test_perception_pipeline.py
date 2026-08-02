import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from utils.seeding import set_global_seed
from brain.perception.sensor_array import DirectionTunedSensorArray
from brain.perception.population_decoder import PopulationVectorDecoder
from brain.perception.perception_pipeline import PerceptionPipeline


def test_sensor_array_direction_tuning():
    arr = DirectionTunedSensorArray(n_sensors=12, max_range=10.0)
    readings = arr.sense(np.array([3.0, 0.0]), noise_std=0.0)
    assert readings[0] == max(readings)


def test_sensor_array_distance_falloff():
    arr = DirectionTunedSensorArray(n_sensors=12, max_range=10.0)
    close = arr.sense(np.array([2.0, 0.0]), noise_std=0.0)
    far = arr.sense(np.array([8.0, 0.0]), noise_std=0.0)
    assert close.max() > far.max()


def test_decoder_raises_before_calibration():
    decoder = PopulationVectorDecoder(n_neurons=12)
    with pytest.raises(RuntimeError):
        decoder.decode(np.ones(12))


def test_perception_pipeline_direction_accuracy():
    """Regression test for a real bug found and fixed: naively assuming
    SNN output neuron i is tuned to the same direction as sensor i
    (identity mapping) gave a mean cosine similarity of only ~0.60
    against ground truth, because the SNN's weight matrix is randomly
    initialized (not identity) - output neurons respond to a mix of all
    inputs. Fixed via calibration: measuring each neuron's actual
    preferred direction empirically before decoding. This test checks
    the fix holds a real accuracy bar, not just 'runs without error'."""
    set_global_seed(0)
    pipeline = PerceptionPipeline(n_sensors=16, max_range=10.0)

    test_cases = [
        np.array([4.0, 0.0]), np.array([0.0, -4.0]), np.array([-4.0, 0.0]),
        np.array([3.0, 3.0]), np.array([1.0, 0.0]), np.array([8.0, 0.0]),
    ]
    cosine_sims = []
    for true_pos in test_cases:
        true_dir = true_pos / np.linalg.norm(true_pos)
        estimate = pipeline.perceive(true_pos, n_timesteps=30, sensor_noise_std=0.02)
        decoded_dir = estimate["direction"]
        if np.linalg.norm(decoded_dir) > 1e-6:
            cosine_sims.append(np.dot(true_dir, decoded_dir))

    mean_cos_sim = np.mean(cosine_sims)
    assert mean_cos_sim > 0.7, f"expected strong direction correlation after calibration fix, got {mean_cos_sim}"


def test_perception_pipeline_proximity_ordering():
    set_global_seed(0)
    pipeline = PerceptionPipeline(n_sensors=16, max_range=10.0)
    close_est = pipeline.perceive(np.array([1.0, 0.0]), n_timesteps=30, sensor_noise_std=0.0)
    far_est = pipeline.perceive(np.array([8.0, 0.0]), n_timesteps=30, sensor_noise_std=0.0)
    assert close_est["proximity_estimate"] > far_est["proximity_estimate"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
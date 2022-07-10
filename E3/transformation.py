"""
Transformations for time series
Based on https://github.com/OxWearables/ssl-wearables, and
https://ai.facebook.com/blog/wav2vec-state-of-the-art-speech-recognition-through-self-supervision/
"""
import numpy as np
from scipy.interpolate import CubicSpline


def identity(sample):
    return sample


def flip(sample):
    return np.flip(sample, 1)


def permute(sample, num_segments=4):
    segment_points_permuted = np.random.choice(sample.shape[1], size=(sample.shape[0], num_segments))
    segment_points = np.sort(segment_points_permuted, axis=1)

    sample_transformed = np.empty(shape=sample.shape)
    for i, (sample_i, segments) in enumerate(zip(sample, segment_points)):
        splitted = np.array(np.split(sample_i, np.append(segments, sample.shape[1])))
        np.random.shuffle(splitted)
        concat = np.concatenate(splitted, axis=0)
        sample_transformed[i] = concat
    return sample_transformed


def time_warp(sample, sigma=0.2):
    sample = np.swapaxes(sample, 0, 1)
    sample = DA_TimeWarp(sample, sigma=sigma)
    sample = np.swapaxes(sample, 0, 1)
    return sample


def negate(sample):
    return sample * -1


def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
    X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])
    return X_new


def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(
        X, sigma
    )  # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [
        (X.shape[0] - 1) / tt_cum[-1, 0],
        (X.shape[0] - 1) / tt_cum[-1, 1],
        (X.shape[0] - 1) / tt_cum[-1, 2],
    ]
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    return tt_cum


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (
        np.ones((X.shape[1], 1))
        * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))
    ).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()

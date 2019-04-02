from random import uniform as random

from scipy.constants import Boltzmann
from tensorflow import random_normal, reduce_mean


def neighbor(x):
    delta = random_normal(shape=x.get_shape(), mean=0.0, stddev=0.001 * reduce_mean(x))
    x = x + delta
    return x


def sa_random(x, y):
    return random(x, y)


def boltzmann_constants():
    return Boltzmann

import torch
import numpy as np
import yaml
from math import sqrt, cos, sin, radians
from random import random

def det(ux, uy, vx, vy):
    return ux*vy - uy*vx

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
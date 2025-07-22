import time
import numpy as np
from typing import Union
import torch
from torch2trt import TRTModule

from utils import preprocess
from road_follower_class import RoadFollower


class CollisionAvoidance(RoadFollower):
    
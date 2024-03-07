from typing import Dict

# Third Party
import carb
import random
import torch
import numpy as np
from tampkit.sim_tools.isaacsim.geometry import Grasp
from tampkit.sim_tools.isaacsim.sim_utils import (
    get_unit_vector,
    multiply,
    get_side_grasps,
    unit_quat,
    compute_grasp_width
)

GRASP_LENGTH = 1.0

def get_grasp_gen(problem, collisions=True, randomize=False):
    def gen_fn(body):
        grasps = []
        arm = 'arm'
        approach_vector = get_unit_vector([0, -1, 0])
        grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g))
                for g in get_side_grasps(body, grasp_length=GRASP_LENGTH))

        filtered_grasps = []
        for grasp in grasps:
            grasp_width = compute_grasp_width(problem.robot, arm, body, grasp.value) if collisions else 0.125 # TODO: modify
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        if randomize:
            random.shuffle(filtered_grasps)
        return [(g,) for g in filtered_grasps]
    return gen_fn

from typing import Dict

# Third Party
import carb
import random
import torch
import numpy as np
from xarm_tamp.tampkit.sim_tools.primitives import BodyGrasp
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    approximate_as_prism,
    get_link_pose,
    get_tool_link,
    get_unit_vector,
    multiply,
    unit_pose,
    unit_quat,
    unit_point,
)


def sample_grasps(body, grasp_length=1.0, max_width=0.5):
    center, (w, l, h) = 10, (0.1, 0.5, 0.6) # approximate_as_prism(body, body_pose=unit_pose())
    reflect_z = np.array([0, np.pi, 0])
    translate_z = np.array([0, 0, h / 2 - grasp_length])
    translate_center = np.array(unit_point()-center)
    grasps = []

    under = 0
    tool_pose = get_link_pose(body, get_tool_link(body, body._end_effector_prim_name))
    if w <= max_width:
        for i in range(1 + under):
            rotate_z = np.array([0, 0, np.pi / 2 + i * np.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, unit_pose())]

    if l <= max_width:
        for i in range(1 + under):
            rotate_z = np.array([0, 0, i * np.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, unit_pose())]

    return grasps

def get_grasp_gen(problem, collisions=True):
    robot = problem.robot
    def gen_fn(body):
        grasps = []
        approach_vector = get_unit_vector([0, -1, 0])
        grasps.extend(BodyGrasp(robot, g, body, multiply((approach_vector, unit_quat()),))
            for g in sample_grasps(body))

        filtered_grasps = []
        for grasp in grasps:
            grasp.grasp_width = 0.15
            filtered_grasps.append(grasp)

        return [(g,) for g in filtered_grasps]
    return gen_fn
from typing import Dict

# Third Party
import carb
import random
import torch
import numpy as np
from tampkit.sim_tools.isaacsim.geometry import Grasp
from tampkit.sim_tools.isaacsim.sim_utils import (
    get_link_pose,
    get_tool_link,
    get_unit_vector,
    multiply,
    unit_pose,
    unit_quat,
    unit_point,
)


def sample_grasps(body, grasp_length=1.0, max_width=0.5):
    center, (w, l, h) = approximate_as_prism(body, body_pose=unit_pose())
    reflect_z = np.array([0, np.pi, 0])
    translate_z = np.array([0, 0, h / 2 - grasp_length])
    translate_center = np.array(unit_point()-center)
    grasps = []

    under = 0
    tool_pose = get_link_pose(get_tool_link(body))
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

# def get_grasp_gen(problem, collisions=True):
#     def gen_fn(body, arm='arm'):
#         grasps = []
#         approach_vector = get_unit_vector([0, -1, 0])
#         grasps.extend(Grasp(body, g, multiply((approach_vector, unit_quat()), g))
#             for g in sample_grasps(body))

#         filtered_grasps = []
#         for grasp in grasps:
#             grasp.grasp_width = 0.15
#             filtered_grasps.append(grasp)

#         return [(g,) for g in filtered_grasps]
#     return gen_fn

def get_grasp_gen(problem, grasp_name='top'):
    grasp_info = GRASP_INFO[grasp_name]
    tool_link = get_tool_link(problem.robot)
    def gen_fn(body):
        grasp_poses = grasp_info.get_grasps(body)
        for grasp_pose in grasp_poses:
            body_grasp = BodyGrasp(body,
                                   grasp_pose, 
                                   grasp_info.approach_pose,
                                   problem.robot,
                                   tool_link)
            yield (body_grasp,)
    return gen_fn
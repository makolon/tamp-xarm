from typing import Dict

# Third Party
import carb
import random
import torch
import numpy as np
from tampkit.sim_tools.isaacsim.primitives import BodyGrasp
from tampkit.sim_tools.isaacsim.sim_utils import (
    approximate_as_prism,
    get_link_pose,
    get_tool_link,
    get_unit_vector,
    multiply,
    unit_pose,
    unit_quat,
    unit_point,
)

def get_side_grasps(body,
                    under=False,
                    tool_pose=unit_pose(),
                    body_pose=unit_pose(),
                    max_width=np.inf,
                    grasp_length=0.0,
                    top_offset=0.03):
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    translate_center = [body_pose[0]-center, unit_quat()]
    grasps = []
    x_offset = h / 2 - top_offset

    for j in range(1 + under):
        swap_xz = [unit_point(), [0, -np.pi / 2 + j * np.pi, 0]]
        if w <= max_width:
            translate_z = [[x_offset, 0, l / 2 - grasp_length], unit_quat()]
            for i in range(2):
                rotate_z = [unit_point(), [np.pi / 2 + i * np.pi, 0, 0]]
                grasps += [multiply(tool_pose, translate_z, rotate_z,
                                    swap_xz, translate_center, body_pose)]  # , np.array([w])

        if l <= max_width:
            translate_z = [[x_offset, 0, w / 2 - grasp_length], unit_quat()]
            for i in range(2):
                rotate_z = [unit_point(), [i * np.pi, 0, 0]]
                grasps += [multiply(tool_pose, translate_z, rotate_z,
                                    swap_xz, translate_center, body_pose)]  # , np.array([l])
    return grasps

def get_top_grasps(body,
                   under=False,
                   tool_pose=unit_pose(),
                   body_pose=unit_pose(),
                   max_width=0.0,
                   grasp_length=0.03):
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    reflect_z = [unit_point(), [0, np.pi, 0]]
    translate_z = [[0, 0, h / 2 - grasp_length], unit_quat()]
    translate_center = [body_pose[0]-center, unit_quat()]
    grasps = []

    if w <= max_width:
        for i in range(1 + under):
            rotate_z = [unit_point(), [0, 0, np.pi / 2 + i * np.pi]]
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]

    if l <= max_width:
        for i in range(1 + under):
            rotate_z = [unit_point(), [0, 0, i * np.pi]]
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]

    return grasps

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

def get_grasp_gen(problem, collisions=True):
    def gen_fn(body):
        grasps = []
        approach_vector = get_unit_vector([0, -1, 0])
        grasps.extend(BodyGrasp(body, g, multiply((approach_vector, unit_quat()), g))
            for g in sample_grasps(body))

        filtered_grasps = []
        for grasp in grasps:
            grasp.grasp_width = 0.15
            filtered_grasps.append(grasp)

        return [(g,) for g in filtered_grasps]
    return gen_fn
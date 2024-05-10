from typing import Dict

# Third Party
import carb
import random
import torch
import numpy as np
from xarm_tamp.tampkit.sim_tools.primitives import BodyGrasp
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    approximate_as_prism,
    get_pose,
    get_link_pose,
    get_tool_link,
    multiply_array,
    pairwise_collision,
    unit_pose,
    unit_point,
    unit_quat,
)


def sample_grasps(body, tool_pose, grasp_length=1.0, max_width=0.5):
    center, (w, l, h) = approximate_as_prism(body, get_pose(body))
    reflect_z = [unit_point(), np.array([0, np.pi, 0])]
    translate_z = [np.array([0, 0, h/2-grasp_length]), unit_quat()]
    translate_center = [np.array(unit_point()-center), unit_quat()]
    grasps = []

    under = 0
    if w <= max_width:
        for i in range(1 + under):
            rotate_z = [unit_point(), np.array([0, 0, np.pi/2+i*np.pi])]
            grasps += [multiply_array(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, unit_pose())]

    if l <= max_width:
        for i in range(1 + under):
            rotate_z = [unit_point(), np.array([0, 0, i*np.pi])]
            grasps += [multiply_array(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, unit_pose())]
    return grasps

def get_grasp_gen(problem, collisions=True):
    robot = problem.robot
    tool_link = get_tool_link(robot, robot._end_effector_prim_name)
    tool_pose = get_link_pose(robot, robot._end_effector_prim_name)

    obstacles = problem.fixed if collisions else []
    def gen_fn(body):
        # TOD: fix
        approach_pose = None
        while True:
            grasp_pose = sample_grasps(body, tool_pose)
            if (len(grasp_pose) == 0) or any(pairwise_collision(robot.gripper, obstacles)):
                continue
            body_grasp = BodyGrasp(robot, tool_link, body, grasp_pose, approach_pose)
            yield (body_grasp,)
    return gen_fn
from typing import Dict

# Third Party
import carb
import random
import torch
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.torch.maths import normalize, scale_transform, unscale_transform
from omni.isaac.core.utils.torch.rotations import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from tampkit.sim_tools.isaacsim.geometry import Grasp


def get_grasp_gen(problem, collisions=True, randomize=False):
    def gen_fn(body):
        grasps = []
        arm = 'arm'
        approach_vector = get_unit_vector([0, -1, 0])
        grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_ARM)
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

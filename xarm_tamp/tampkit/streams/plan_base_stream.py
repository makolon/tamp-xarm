import random
import itertools
import numpy as np
from tampkit.sim_tools.isaacsim.geometry import Pose, Conf
from tampkit.sim_tools.isaacsim.sim_utils import (
    is_placement,
    all_between,
    pairwise_collision,
    iterate_approach_path,
    get_arm_joints,
    get_custom_limits,
    get_base_joints,
    all_between,
    set_joint_positions,
    unit_from_theta
)

CIRCULAR_LIMITS = (10.0, 10.0)

def uniform_pose_generator(robot, gripper_pose, reachable_range=(10.0, 10.0), **kwargs):
    point = gripper_pose[0]
    while True:
        radius = np.random.uniform(*reachable_range)
        x, y = radius * unit_from_theta(np.random.uniform(np.pi, 2*np.pi/2)) + point[:2]
        yaw = np.random.uniform(*CIRCULAR_LIMITS)
        base_values = (x, y, yaw)
        if base_values is None:
            break
        yield base_values


def plan_base_fn(problem, collisions=True, max_attempts=25, custom_limits={}):
    # Sample move_base pose
    robot = problem.robot
    obstacles = problem.fixed if collisions else []

    def gen_fn(base, body, pose):
        pose.assign()

        gripper_pose = pose.value
        base_joints = get_base_joints(robot)

        base_generator = uniform_pose_generator(robot, gripper_pose)

        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            for base_conf in itertools.islice(base_generator, max_attempts):
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                bq.assign()
                yield (bq,)
                break
            else:
                yield None
    return gen_fn
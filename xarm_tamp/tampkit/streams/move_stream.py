import random
import numpy as np
# TODO: fix
from omni.isaac.core.utils.transforms import euler_from_quat
from omni.isaac.core.utils.maths import multiply
from tampkit.sim_tools.isaacsim.geometry import Pose, Conf
from tampkit.sim_tools.isaacsim.sim_utils import (
    islice,
    is_placement,
    all_between,
    max_attempts,
    pairwise_collision,
    iterate_approach_path,
    arm_conf,
    get_arm_joints,
    get_custom_limits,
    get_group_joints,
    set_joint_positions
)


def learned_pose_generator(robot, gripper_pose, arm, grasp_type):
    gripper_from_base_list = load_inverse_reachability(arm, grasp_type)
    random.shuffle(gripper_from_base_list)
    for gripper_from_base in gripper_from_base_list:
        base_point, base_quat = multiply(gripper_pose, gripper_from_base)
        x, y, _ = base_point
        _, _, theta = euler_from_quat(base_quat)
        base_values = (x, y, theta)
        yield base_values

def sample_reachable_base(robot, point, reachable_range=(0.9, 0.95)):
    radius = np.random.uniform(*reachable_range)
    x, y = radius * unit_from_theta(np.random.uniform(np.pi, 2*np.pi/2)) + point[:2]
    yaw = np.random.uniform(*CIRCULAR_LIMITS)
    base_values = (x, y, yaw)
    return base_values

def uniform_pose_generator(robot, gripper_pose, **kwargs):
    point = gripper_pose[0]
    while True:
        base_values = sample_reachable_base(robot, point, **kwargs)
        if base_values is None:
            break
        yield base_values


def get_move_gen(problem, collisions=True, learned=False):
    # Sample move_base pose
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    gripper = problem.get_gripper()

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
            if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
                return

        gripper_pose = pose.value # multiply(pose.value, invert(grasp.value))
        default_conf = arm_conf(arm, grasp.carry)
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_group_joints(robot, 'base')

        if learned:
            base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)

        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            for base_conf in islice(base_generator, max_attempts):
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                yield (bq,)
                break
            else:
                yield None
    return gen_fn
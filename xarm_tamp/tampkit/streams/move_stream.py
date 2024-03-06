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

def learned_pose_generator(robot, gripper_pose, arm='arm'):
    pass

def uniform_pose_generator(robot, gripper_pose, arm='arm'):
    pass


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
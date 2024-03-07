import numpy as np
from tampkit.sim_tools.isaacsim.sim_utils import (
    get_arm_joints, get_initial_conf,
    set_joint_positions,
    is_placement, arm_conf, pairwise_collision,
)


def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=False, resolution=0.05):
    robot = problem.robot
    ik_solver = problem.ik_solver
    obstacles = problem.fixed if collisions else []

    def fn(arm, obj, pose, grasp):
        arm_joints = get_arm_joints(robot, arm)

        # Default confs
        default_arm_conf = get_initial_conf(robot)

        pose.assign()

        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {attachment.child: attachment}

        # Set position to default configuration for grasp action
        assert len(default_arm_conf) == len(robot.arm_joints), "Lengths do not match."
        set_joint_positions(robot, default_arm_conf)

        # TODO: fix this function to computer collision free ik
        target_conf = ik_solver.solve()
        if (target_conf is None):
            return None

        assert len(target_conf) == len(robot.arm_joints), "Lengths do not match."
        set_joint_positions(robot, target_conf)

        cmd = Command()
        return (cmd,)
    return fn

def get_ik_gen(problem, max_attempts=25, learned=False, teleport=False, **kwargs):
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)

    def gen_fn(*inputs):
        a, p, g = inputs
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            ik_outputs = ik_fn(*(inputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            yield ik_outputs
            return
    return gen_fn
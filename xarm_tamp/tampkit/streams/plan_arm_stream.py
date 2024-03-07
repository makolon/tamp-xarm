import carb
import numpy as np
from tampkit.sim_tools.isaacsim.sim_utils import (
    get_arm_joints,
    get_initial_conf,
    set_joint_positions,
)
from tampkit.sim_tools.isaacsim.curobo_utils import (
    get_tensor_device_type,
    get_motion_gen_plan_cfg
)
from tampkit.sim_tools.isaacsim.geometry import Conf, State, Trajectory
from tampkit.sim_tools.isaacsim.control import Commands

from curobo.types.math import Pose
from curobo.types.state import JointState


def create_trajectory(robot, joints, path):
    return Trajectory(Conf(robot, joints, q) for q in path)

def get_motion_fn(problem, collisions=True, teleport=False):
    robot = problem.robot
    motion_planner = problem.motion_planner
    tensor_args = get_tensor_device_type()
    plan_cfg = get_motion_gen_plan_cfg()
    obstacles = problem.fixed if collisions else []

    def fn(arm, body, pose, grasp):
        arm_joints = get_arm_joints(robot, arm)

        # Default confs
        default_arm_conf = get_initial_conf(robot)

        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {attachment.child: attachment}

        # Set position to default configuration for grasp action
        assert len(default_arm_conf) == len(robot.arm_joints), "Lengths do not match."

        # Set ik goal
        ik_goal = Pose(
            position=tensor_args.to_device(pose.position),
            quaternion=tensor_args.to_device(pose.rotation),
        )

        # Plan joint motion for grasp
        curr_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=arm_joints
        )
        curr_js = curr_js.get_ordered_joint_state(motion_planner.kinematics.joint_names)
        result = motion_planner.plan_single(curr_js.unsqueeze(0), ik_goal, plan_cfg.clone())
        succ = result.success.item()
        if succ:
            trajectory = result.get_interpolated_plan()
        else:
            carb.log_warn("Plan did not converge to a solution.")
            return None

        mt = create_trajectory(robot, arm_joints, trajectory)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd,)

    return fn

def plan_arm_fn(problem, max_attempts=25, learned=False, teleport=False, **kwargs):
    ik_fn = get_motion_fn(problem, teleport=teleport, **kwargs)

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
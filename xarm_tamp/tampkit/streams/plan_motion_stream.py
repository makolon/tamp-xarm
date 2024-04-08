import carb
import numpy as np
from xarm_tamp.tampkit.sim_tools.primitives import BodyConf, BodyPath, Command
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    get_arm_joints,
    get_initial_conf,
)
from curobo.types.math import Pose
from curobo.types.state import JointState


def get_motion_fn(problem, collisions=True, teleport=False):
    robot = problem.robot
    tensor_args = problem.tensor_args
    plan_cfg = problem.plan_cfg
    ik_solver = problem.ik_solver
    motion_planner = problem.motion_planner
    obstacles = problem.fixed if collisions else []

    def fn(body, pose, grasp):
        arm_joints = get_arm_joints(robot)

        # Default confs
        default_arm_conf = get_initial_conf(robot)

        # Set position to default configuration for grasp action
        assert len(default_arm_conf) == len(robot.arm_joints), "Lengths do not match."

        # Set ik goal
        ik_goal = Pose(
            position=tensor_args.to_device(pose.position),
            quaternion=tensor_args.to_device(pose.rotation),
        )
        goal_conf = ik_solver.solve_single(ik_goal)

        # Get joint states
        sim_js = robot.get_joints_state()

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

        conf = BodyConf(robot, goal_conf)
        command = Command([BodyPath(robot, trajectory)])
        return (conf, command)

    return fn

def plan_motion_fn(problem, max_attempts=25, teleport=False, **kwargs):
    ik_fn = get_motion_fn(problem, teleport=teleport, **kwargs)

    def gen_fn(*inputs):
        b, p, g = inputs
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
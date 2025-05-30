import carb
import torch
from xarm_tamp.tampkit.sim_tools.primitives import BodyPath
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    compute_configuration_distance,
    get_arm_joints,
    get_initial_conf,
)
from curobo.types.math import Pose
from curobo.types.state import JointState


def get_free_motion_fn(problem, collisions=True, batch_size=64, threshold=0.1):
    robot = problem.robot
    tensor_args = problem.tensor_args
    plan_cfg = problem.plan_cfg
    ik_solver = problem.ik_solver
    motion_planner = problem.motion_planner

    def fn(conf1, conf2):
        arm_joints = get_arm_joints(robot)

        # Default confs
        default_arm_conf = get_initial_conf(robot)

        # Set position to default configuration for grasp action
        assert len(default_arm_conf) == len(arm_joints), "Lengths do not match."

        # Calculate forawrd kinematics
        q = torch.tensor(conf2.value, **(tensor_args.as_torch_dict())).squeeze(0).repeat(batch_size, 1)
        out = ik_solver.fk(q)
        position, rotation = out.ee_position[0], out.ee_quaternion[0]

        # Set ik goal
        ik_goal = Pose(
            position=tensor_args.to_device(position),
            quaternion=tensor_args.to_device(rotation),
        )

        # Filter out same configuration
        conf_diff = compute_configuration_distance(conf1.value, conf2.value)
        if conf_diff <= threshold:
            traj = BodyPath(robot, arm_joints, [conf2.value])
            return (traj,)

        # Get joint states
        sim_js = robot.get_joints_state()

        # Plan joint motion for grasp
        curr_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=robot._arm_dof_names
        )
        curr_js = curr_js.get_ordered_joint_state(motion_planner.kinematics.joint_names)
        result = motion_planner.plan_single(curr_js.unsqueeze(0), ik_goal, plan_cfg.clone())
        succ = result.success.item()
        if succ:
            trajectory = result.get_interpolated_plan()
        else:
            carb.log_warn("Plan did not converge to a solution.")
            return None

        traj = BodyPath(robot, arm_joints, trajectory.position)
        return (traj,)

    return fn

def get_holding_motion_fn(problem, collisions=True, batch_size=64, threshold=0.1):
    robot = problem.robot
    tensor_args = problem.tensor_args
    plan_cfg = problem.plan_cfg
    ik_solver = problem.ik_solver
    motion_planner = problem.motion_planner

    def fn(conf1, conf2, body, grasp):
        arm_joints = get_arm_joints(robot)

        # Default confs
        default_arm_conf = get_initial_conf(robot)

        # Set position to default configuration for grasp action
        assert len(default_arm_conf) == len(arm_joints), "Lengths do not match."

        # Calculate forawrd kinematics
        q = torch.tensor(conf2.value, **(tensor_args.as_torch_dict())).squeeze(0).repeat(batch_size, 1)
        out = ik_solver.fk(q)
        position, rotation = out.ee_position[0], out.ee_quaternion[0]

        # Set ik goal
        ik_goal = Pose(
            position=tensor_args.to_device(position),
            quaternion=tensor_args.to_device(rotation),
        )

        # Filter out same configuration
        conf_diff = compute_configuration_distance(conf1.value, conf2.value)
        if conf_diff <= threshold:
            traj = BodyPath(robot, arm_joints, [conf2.value])
            return (traj,)

        # Get joint states
        sim_js = robot.get_joints_state()

        # Plan joint motion for grasp
        curr_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=robot._arm_dof_names
        )
        curr_js = curr_js.get_ordered_joint_state(motion_planner.kinematics.joint_names)
        result = motion_planner.plan_single(curr_js.unsqueeze(0), ik_goal, plan_cfg.clone())
        succ = result.success.item()
        if succ:
            trajectory = result.get_interpolated_plan()
        else:
            carb.log_warn("Plan did not converge to a solution.")
            return None

        traj = BodyPath(robot, arm_joints, trajectory.position)
        return (traj,)

    return fn
import carb
from xarm_tamp.tampkit.sim_tools.primitives import BodyConf, BodyPath
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    end_effector_from_body,
    get_arm_joints,
)
from curobo.types.math import Pose
from curobo.types.state import JointState


def get_ik_fn(problem, collisions=True):
    robot = problem.robot
    tensor_args = problem.tensor_args
    plan_cfg = problem.plan_cfg
    ik_solver = problem.ik_solver
    motion_planner = problem.motion_planner

    # Get arm joints
    arm_joints = get_arm_joints(robot)
    def fn(body, pose, grasp):
        # Target pose
        target_position, target_rotation = end_effector_from_body(pose.value, grasp.value)
        target_rotation = [0.0, 1.0, 0.0, 0.0]  # TODO: fix

        # Set ik goal
        ik_goal = Pose(
            position=tensor_args.to_device(target_position),
            quaternion=tensor_args.to_device(target_rotation),
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

        conf = BodyConf(robot=robot, configuration=goal_conf.js_solution.position.squeeze().cpu().numpy())
        arm_traj = trajectory.position.cpu().numpy()
        traj = BodyPath(robot, arm_joints, arm_traj)
        return (conf, traj)

    return fn
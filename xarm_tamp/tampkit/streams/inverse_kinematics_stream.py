import carb
from xarm_tamp.tampkit.sim_tools.primitives import BodyConf, BodyPath, Command
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    create_trajectory,
    get_arm_joints,
    get_initial_conf,
)
from curobo.types.math import Pose
from curobo.types.state import JointState


def get_ik_fn(problem, collisions=True):
    robot = problem.robot
    tensor_args = problem.tensor_args
    plan_cfg = problem.plan_cfg
    ik_solver = problem.ik_solver
    motion_planner = problem.motion_planner
    obstacles = problem.fixed if collisions else []

    def fn(body, pose, grasp):
        # TODO: add grasp to attach
        arm_joints = get_arm_joints(robot)

        # Default confs
        default_arm_conf = get_initial_conf(robot)

        # Set position to default configuration for grasp action
        assert len(default_arm_conf) == len(arm_joints), "Lengths do not match."

        # target pose
        position, rotation = grasp.value

        # Set ik goal
        ik_goal = Pose(
            position=tensor_args.to_device(position),
            quaternion=tensor_args.to_device(rotation),
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

        conf = BodyConf(robot=robot, configuration=goal_conf.js_solution.position)

        # create trajectory
        art_traj = create_trajectory(trajectory, arm_joints)
        command = Command([BodyPath(robot, art_traj)])
        return (conf, command)

    return fn
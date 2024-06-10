import hydra
import random
import numpy as np
from omegaconf import DictConfig
from xarm_tamp.tampkit.problems import stacking_problem, fetch_problem, fmb_momo_problem
from xarm_tamp.tampkit.sim_tools.sim_utils import connect, get_pose
from xarm_tamp.tampkit.streams.grasp_stream import get_grasp_gen
from xarm_tamp.tampkit.streams.insert_stream import get_insert_gen
from xarm_tamp.tampkit.streams.inverse_kinematics_stream import get_ik_fn
from xarm_tamp.tampkit.streams.place_stream import get_place_gen
from xarm_tamp.tampkit.streams.plan_motion_stream import get_motion_fn
from xarm_tamp.tampkit.streams.test_stream import (
    get_cfree_pose_pose_test, get_cfree_approach_pose_test,
    get_cfree_traj_pose_test, get_supported, get_inserted
)


### Grasp Stream Test
def grasp_stream_test(problem, assign=False):
    # connect to simulator
    sim_app = connect()

    # grasp pose generator
    grasp_gen = get_grasp_gen(problem, collisions=False)

    world = problem.world
    robot = problem.robot
    bodies = problem.movable
    obstacles = problem.fixed
    print('bodies:', bodies)
    print('obstacles:', obstacles)

    step_index = 0
    while sim_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            print("**** Click Play to start simulation ****")
            continue

        step_index += 1
        if step_index < 2:
            world.reset()
            robot._articulation_view.initialize()

        if step_index < 20:
            default_config = np.array([0.0, -0.698, 0.0, 0.349, 0.0, 1.047, 0.0, 0.0, 0.0])
            idx_list = [robot.get_dof_index(x) for x in robot._arm_dof_names+robot._gripper_dof_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            continue

        # step grasp simulation
        num_trial = 0
        body = random.choice(bodies)
        for grasp, in grasp_gen(body):
            print('grasp value:', grasp.value)
            if assign:
                # assign once
                grasp.assign()
                assign = False

            # step simulation
            world.step(render=True)
            if not world.is_playing():
                continue

            num_trial += 1
            if num_trial > 50:
                break

        # stop simulation
        break


### Place Stream Test
def place_stream_test(problem, assign=False):
    # connect to simulator
    sim_app = connect()

    # place pose generator
    place_gen = get_place_gen(problem, collisions=False)

    world = problem.world
    robot = problem.robot
    bodies = problem.movable
    obstacles = problem.fixed
    surfaces = problem.surfaces
    print('bodies:', bodies)
    print('obstacles:', obstacles)
    print('surfaces:', surfaces)

    step_index = 0
    while sim_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            print("**** Click Play to start simulation ****")
            continue

        step_index += 1
        if step_index < 2:
            world.reset()
            robot._articulation_view.initialize()

        if step_index < 20:
            default_config = np.array([0.0, -0.698, 0.0, 0.349, 0.0, 1.047, 0.0, 0.0, 0.0])
            idx_list = [robot.get_dof_index(x) for x in robot._arm_dof_names+robot._gripper_dof_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            continue

        # step place simulation
        num_trial = 0
        body, surface = random.choice(bodies), random.choice(surfaces)
        for place, in place_gen(body, surface):
            print('place value:', place.value)
            if assign:
                # assign once
                place.assign()
                assign = False

            # step simulation
            world.step(render=True)
            if not world.is_playing():
                continue

            num_trial += 1
            if num_trial > 50:
                break

        # stop simulation
        break


### Insert Stream Test
def insert_stream_test(problem, assign=False):
    # connect to simulator
    sim_app = connect()

    # insert pose generator
    insert_gen = get_insert_gen(problem, collisions=False)

    world = problem.world
    robot = problem.robot
    bodies = problem.movable
    obstacles = problem.fixed
    holes = problem.holes
    print('bodies:', bodies)
    print('obstacles:', obstacles)
    print('holes:', holes)

    step_index = 0
    while sim_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            print("**** Click Play to start simulation ****")
            continue

        step_index += 1
        if step_index < 2:
            world.reset()
            robot._articulation_view.initialize()

        if step_index < 20:
            default_config = np.array([0.0, -0.698, 0.0, 0.349, 0.0, 1.047, 0.0, 0.0, 0.0])
            idx_list = [robot.get_dof_index(x) for x in robot._arm_dof_names+robot._gripper_dof_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            continue

        # step insert simulation
        num_trial = 0
        body, hole = random.choice(bodies), random.choice(holes)
        for insert, in insert_gen(body, hole):
            print('insert value:', insert.value)
            if assign:
                # assign once
                insert.assign()
                assign = False

            # step simulation
            world.step(render=True)
            if not world.is_playing():
                continue

            num_trial += 1
            if num_trial > 50:
                break

        # stop simulation
        break


### IK Stream Test
def ik_stream_test(problem, assign=False):
    # connect to simulator
    sim_app = connect()

    # grasp pose generator
    grasp_gen = get_grasp_gen(problem, collisions=False)

    # ik solver
    ik_fn = get_ik_fn(problem)

    world = problem.world
    robot = problem.robot
    bodies = problem.bodies
    print('bodies:', bodies)

    # target pose
    target_pose = [np.array((0.3, 0.0, 0.06)), np.array((0., 1., 0., 0.))]
    step_index = 0
    while sim_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            print("**** Click Play to start simulation ****")
            continue

        step_index += 1
        if step_index < 2:
            world.reset()
            robot._articulation_view.initialize()

        if step_index < 20:
            default_config = np.array([0.0, -0.698, 0.0, 0.349, 0.0, 1.047, 0.0, 0.0, 0.0])
            idx_list = [robot.get_dof_index(x) for x in robot._arm_dof_names+robot._gripper_dof_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            continue

        # step ik simulation
        num_trial = 0
        body = random.choice(bodies)
        for grasp, in grasp_gen(body):
            result = ik_fn(body, target_pose, grasp)
            if result is None:
                continue

            goal_conf, trajectory = result
            print('goal_conf:', goal_conf.value)
            print('trajectory:', trajectory.body_paths)

            if assign:
                trajectory.control()

            num_trial += 1
            if num_trial > 50:
                break

        # stop simulation
        break


### Plan Motion Stream Test
def plan_motion_stream_test(problem):
    # connect to simulator
    sim_app = connect()

    # grasp pose generator
    grasp_gen = get_grasp_gen(problem, collisions=False)

    # motion planner
    plan_motion_fn = get_motion_fn(problem)

    world = problem.world
    robot = problem.robot
    bodies = problem.bodies
    obstacles = problem.fixed
    print('bodies:', bodies)
    print('obstacles:', obstacles)

    step_index = 0
    while sim_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            print("**** Click Play to start simulation ****")
            continue

        step_index += 1
        if step_index < 2:
            world.reset()
            robot._articulation_view.initialize()

        if step_index < 20:
            default_config = np.array([0.0, -0.698, 0.0, 0.349, 0.0, 1.047, 0.0, 0.0, 0.0])
            idx_list = [robot.get_dof_index(x) for x in robot._arm_dof_names+robot._gripper_dof_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            continue

        body = random.choice(bodies)
        sim_js = robot.get_joints_state()
        for goal_conf, in grasp_gen(body):
            plan = plan_motion_fn(sim_js, goal_conf)
            if plan is None:
                continue
            print('plan:', plan)

        # stop simulation
        break


### Test Stream Test
def test_stream_test(problem):
    # connect to simulator
    sim_app = connect()

    # grasp pose generator
    grasp_gen = get_grasp_gen(problem, collisions=False)

    # test
    cfree_pp_test = get_cfree_pose_pose_test(problem)
    cfree_ap_test = get_cfree_approach_pose_test(problem)
    cfree_tp_test = get_cfree_traj_pose_test(problem)
    supported_test = get_supported(problem)
    inserted_test = get_inserted(problem)

    world = problem.world
    robot = problem.robot
    bodies = problem.bodies
    obstacles = problem.fixed
    print('bodies:', bodies)
    print('obstacles:', obstacles)

    step_index = 0
    while sim_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            print("**** Click Play to start simulation ****")
            continue

        step_index += 1
        if step_index < 2:
            world.reset()
            robot._articulation_view.initialize()

        if step_index < 20:
            default_config = np.array([0.0, -0.698, 0.0, 0.349, 0.0, 1.047, 0.0, 0.0, 0.0])
            idx_list = [robot.get_dof_index(x) for x in robot._arm_dof_names+robot._gripper_dof_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            continue

        body1, body2 = random.sample(bodies, 2)
        pose1, pose2 = get_pose(body1), get_pose(body2)

        grasp = next(grasp_gen(body1))
        conf = get_ik_fn()

        is_cfree_pp = cfree_pp_test(body1, pose1, body2, pose2)
        is_cfree_ap = cfree_ap_test(body1, pose1, grasp, body2, pose2)
        is_cfree_tp = cfree_tp_test(conf, body1, pose1)
        is_supported = supported_test(body1, pose1, body2, pose2)
        is_inserted = inserted_test(body1, pose1, body2, pose2)

        print('is_cfree_pp:', is_cfree_pp)
        print('is_cfree_ap:', is_cfree_ap)
        print('is_cfree_tp:', is_cfree_tp)
        print('is_supported:', is_supported)
        print('is_inserted:', is_inserted)

        # stop simulation
        break


config_file = input("Please input the problem name from (simple_fetch, simple_stacking, fmb_momo): ")
@hydra.main(version_base=None, config_name=config_file, config_path="../configs")
def main(cfg: DictConfig):
    if config_file == 'simple_stacking' or config_file == 'simple_fetch':
        simple_fetch_problem = fetch_problem(cfg.sim, cfg.curobo)

        print('###########################')
        grasp_stream_test(simple_fetch_problem)
        print('Grasp Stream Ready!')
        print('###########################')

        print('###########################')
        place_stream_test(simple_fetch_problem)
        print('Place Stream Ready!')
        print('###########################')

        print('###########################')
        ik_stream_test(simple_fetch_problem)
        print('IK Stream Ready!')
        print('###########################')

        print('###########################')
        plan_motion_stream_test(simple_fetch_problem)
        print('Plan Motion Stream Ready!')
        print('###########################')

        print('###########################')
        test_stream_test(simple_fetch_problem)
        print('Test Stream Ready!')
        print('###########################')

    if config_file == 'fmb_momo':
        assembly_problem = fmb_momo_problem(cfg.sim, cfg.curobo)

        print('###########################')
        insert_stream_test(assembly_problem)
        print('Insert Stream Ready!')
        print('###########################')

    input('Finish!')


if __name__ == "__main__":
    main()
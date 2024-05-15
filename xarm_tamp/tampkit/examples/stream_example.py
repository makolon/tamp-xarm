import hydra
import random
import numpy as np
from omegaconf import DictConfig
from xarm_tamp.tampkit.problems import stacking_problem, fmb_momo_problem
from xarm_tamp.tampkit.sim_tools.sim_utils import connect
from xarm_tamp.tampkit.streams.grasp_stream import get_grasp_gen
from xarm_tamp.tampkit.streams.insert_stream import get_insert_gen
from xarm_tamp.tampkit.streams.place_stream import get_place_gen
from xarm_tamp.tampkit.streams.plan_motion_stream import get_motion_fn
from xarm_tamp.tampkit.streams.test_stream import (
    get_cfree_pose_pose_test, get_cfree_approach_pose_test,
    get_cfree_traj_pose_test, get_supported, get_inserted
)

### Grasp Stream Test
def grasp_stream_test(problem):
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
        for _ in range(100):
            body = random.choice(bodies)
            gen = grasp_gen(body)
            grasp = next(gen)
            print('grasp:', grasp)
            for g in grasp:
                print('grasp value:', g.value)
                g.assign()

                # step simulation
                world.step(render=True)
                if not world.is_playing():
                    continue

        # stop simulation
        break


### Place Stream Test
def place_stream_test(problem):
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

        for _ in range(100):
            body, surface = random.choice(bodies), random.choice(surfaces)
            gen = place_gen(body, surface)
            place = next(gen)
            print('place:', place)
            for p in place:
                print('place value:', p.value)
                p.assign()

                # step simulation
                world.step(render=True)
                if not world.is_playing():
                    continue

        # stop simulation
        break


### Insert Stream Test
def insert_stream_test(problem):
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

        for _ in range(100):
            body, hole = random.choice(bodies), random.choice(holes)
            gen = insert_gen(body, hole)
            insert = next(gen)
            print('insert:', insert)
            for i in insert:
                print('insert value:', i.value)
                i.assign()

                # step simulation
                world.step(render=True)
                if not world.is_playing():
                    continue

        # stop simulation
        break


### Plan Motion Stream Test
def plan_motion_stream_test(problem):
    # connect to simulator
    sim_app = connect()

    # motion planner
    plan_motion_fn = get_motion_fn(problem)

    world = problem.world
    robot = problem.robot
    bodies = problem.bodies
    obstacles = problem.fixed
    print('bodies:', bodies)
    print('obstacles:', obstacles)

    # target pose
    target_pose = [np.array((0.3, 0.0, 0.15)), np.array((0., 1., 0., 0.))]
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
        plan = plan_motion_fn(body, target_pose)
        if plan is None:
            continue
        print('plan:', plan)

        # stop simulation
        break


### Test Stream Test
def test_stream_test(problem):
    pass


config_file = input("Please input the problem name from (simple_fetch, simple_stacking, fmb_momo, fmb_simo): ")
@hydra.main(version_base=None, config_name=config_file, config_path="../configs")
def main(cfg: DictConfig):
    if config_file == 'simple_stacking':
        simple_stacking_problem = stacking_problem(cfg.sim, cfg.curobo)

        print('###########################')
        grasp_stream_test(simple_stacking_problem)
        print('Grasp Stream Ready!')
        print('###########################')

        print('###########################')
        place_stream_test(simple_stacking_problem)
        print('Place Stream Ready!')
        print('###########################')

        print('###########################')
        plan_motion_stream_test(simple_stacking_problem)
        print('Plan Motion Stream Ready!')
        print('###########################')

        print('###########################')
        test_stream_test(simple_stacking_problem)
        print('Test Stream Ready!')
        print('###########################')

    elif config_file == 'fmb_momo' or 'fmb_simo':
        assembly_problem = fmb_momo_problem(cfg.sim, cfg.curobo)

        print('###########################')
        insert_stream_test(assembly_problem)
        print('Insert Stream Ready!')
        print('###########################')

    input('Finish!')


if __name__ == "__main__":
    main()
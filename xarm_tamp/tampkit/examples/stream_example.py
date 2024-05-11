import hydra
import random
import time
from omegaconf import DictConfig
from xarm_tamp.tampkit.problems import *
from xarm_tamp.tampkit.sim_tools.sim_utils import *
from xarm_tamp.tampkit.streams.grasp_stream import get_grasp_gen
from xarm_tamp.tampkit.streams.insert_stream import get_insert_gen
from xarm_tamp.tampkit.streams.place_stream import get_place_gen
from xarm_tamp.tampkit.streams.plan_motion_stream import get_motion_fn
from xarm_tamp.tampkit.streams.test_stream import *


### Grasp Stream Test
def grasp_stream_test(problem):
    grasp_gen = get_grasp_gen(problem, collisions=False)

    bodies = problem.movable
    for _ in range(100):
        body = random.choice(bodies)
        gen = grasp_gen(body)
        grasp = next(gen)
        print('grasp:', grasp)
        for g in grasp:
            print('grasp value:', g.value)
            g.assign()
            time.sleep(0.1)


### Place Stream Test
def place_stream_test(problem):
    place_gen = get_place_gen(problem, collisions=False)

    bodies, surfaces = problem.movable, problem.surfaces
    for _ in range(100):
        body, surface = random.choice(bodies), random.choice(surfaces)
        gen = place_gen(body, surface)
        place = next(gen)
        print('place:', place)
        for p in place:
            print('place value:', p.value)
            p.assign()
            time.sleep(0.1)


### Insert Stream Test
def insert_stream_test(problem):
    insert_gen = get_insert_gen(problem, collisions=False)

    bodies, holes = problem.movable, problem.holes
    for _ in range(100):
        body, hole = random.choice(bodies), random.choice(holes)
        gen = insert_gen(body, hole)
        insert = next(gen)
        print('insert:', insert)
        for i in insert:
            print('insert value:', i.value)
            i.assign()
            time.sleep(0.1)


### Plan Motion Stream Test
def plan_motion_stream_test(problem):
    plan_motion_fn = get_motion_fn(problem)

    bodies = problem.bodies
    print('bodies:', bodies)
    pose = [np.array((0.25, 0.0, 0.3)), np.array((0., 0., 0., 1.))]
    for _ in range(100):
        body = random.choice(bodies)
        plan = plan_motion_fn(body, pose)
        print('plan:', plan)
        time.sleep(0.1)


### Test Stream Test
def test_stream_test(problem):
    pass


@hydra.main(version_base=None, config_name="fmb_momo", config_path="../configs")
def main(cfg: DictConfig):
    assembly_problem = fmb_momo_problem(cfg.sim, cfg.curobo)

    print('###########################')
    grasp_stream_test(assembly_problem)
    print('Grasp Stream Ready!')
    print('###########################')

    print('###########################')
    place_stream_test(assembly_problem)
    print('Place Stream Ready!')
    print('###########################')

    print('###########################')
    insert_stream_test(assembly_problem)
    print('Insert Stream Ready!')
    print('###########################')

    print('###########################')
    plan_motion_stream_test(assembly_problem)
    print('Plan Motion Stream Ready!')
    print('###########################')

    print('###########################')
    test_stream_test(assembly_problem)
    print('Test Stream Ready!')
    print('###########################')

    input('Finish!')


if __name__ == "__main__":
    main()
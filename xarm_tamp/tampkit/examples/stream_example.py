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
def grasp_stream_test(problem, world):
    grasp_gen = get_grasp_gen(problem)

    bodies = problem.bodies
    print('bodies:', bodies)
    for _ in range(100):
        body = random.choise(bodies)
        grasp = grasp_gen(body)
        print('grasp:', grasp)
        print('grasp value:', grasp.value)
        print('approach:', grasp.approach)
        grasp.assign()
        time.sleep(0.1)


### Insert Stream Test
def insert_stream_test(problem, world):
    insert_gen = get_insert_gen(problem)

    bodies = problem.bodies
    for _ in range(100):
        body = random.choise(bodies)
        insert = insert_gen(body)
        print('insert:', insert)
        print('insert value:', insert.value)
        insert.assign()
        time.sleep(0.1)


### Place Stream Test
def place_stream_test(problem, world):
    place_gen = get_place_gen(problem)

    bodies = problem.bodies
    for _ in range(100):
        body = random.choise(bodies)
        place = place_gen(body)
        print('place:', place)
        print('place value:', place.value)
        place.assign()
        time.sleep(0.1)


### Plan Motion Stream Test
def plan_motion_stream_test(problem):
    plan_motion_fn = plan_motion_fn(problem)

    for _ in range(100):
        plan = plan_motion_fn(body, pose, grasp)
        print('plan:', plan)
        time.sleep(0.1)


### Test Stream Test
def test_stream_test(problem):
    pass


@hydra.main(version_base=None, config_name="assembly_config", config_path="../configs")
def main(cfg: DictConfig):
    assembly_problem = fmb_momo_problem(cfg.sim, cfg.curobo)

    print('###########################')
    grasp_stream_test(assembly_problem)
    print('Grasp Stream Ready!')
    print('###########################')

    print('###########################')
    insert_stream_test(assembly_problem)
    print('Insert Stream Ready!')
    print('###########################')

    print('###########################')
    place_stream_test(assembly_problem)
    print('Place Stream Ready!')
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
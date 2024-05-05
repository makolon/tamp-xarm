import hydra
import random
import numpy as np
from collections import namedtuple
from omegaconf import DictConfig

# Initialize isaac sim
import xarm_tamp.tampkit.sim_tools.sim_utils

from xarm_tamp.tampkit.sim_tools.primitives import BodyPose, BodyConf, Command
from xarm_tamp.tampkit.problems import PROBLEMS
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    # Simulation utility
    connect, disconnect, create_world
)


config_file = input("Please input the problem name from (simple_fetch, simple_stacking, fmb_momo, fmb_simo): ")
@hydra.main(version_base=None, config_name=config_file, config_path="../configs")
def main(cfg: DictConfig):
    # connect
    sim_app = connect()

    # create world
    world = create_world()

    # Instanciate problem 
    problem_from_name = {fn.__name__: fn for fn in PROBLEMS}
    if cfg.pddlstream.problem not in problem_from_name:
        raise ValueError(cfg.pddlstream.problem)
    print('Problem:', cfg.pddlstream.problem)
    problem_fn = problem_from_name[cfg.pddlstream.problem]
    tamp_problem = problem_fn(cfg.sim, cfg.curobo)

    while sim_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue
        
    disconnect()


if __name__ == '__main__':
    main()
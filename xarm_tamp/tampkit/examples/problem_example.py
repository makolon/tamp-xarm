import hydra
import random
import numpy as np
from collections import namedtuple
from omegaconf import DictConfig

# Initialize isaac sim
import xarm_tamp.tampkit.sim_tools.sim_utils

from xarm_tamp.tampkit.sim_tools.primitives import BodyPose, BodyConf, Command
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    # Simulation utility
    connect, disconnect,
    # Getter
    get_pose, get_max_limit, get_arm_joints, get_gripper_joints,
    get_joint_positions,
)

from xarm_tamp.tampkit.problems import PROBLEMS
from xarm_tamp.tampkit.streams.plan_motion_stream import plan_motion_fn
from xarm_tamp.tampkit.streams.grasp_stream import get_grasp_gen
from xarm_tamp.tampkit.streams.place_stream import get_place_gen
from xarm_tamp.tampkit.streams.insert_stream import get_insert_gen
from xarm_tamp.tampkit.streams.test_stream import get_cfree_pose_pose_test, get_cfree_approach_pose_test, \
    get_cfree_traj_pose_test, get_supported, get_inserted
from ..tamp_planner import TAMPPlanner

config_name = input("Please input config from (assembly_config, stacking_config, ...)")
@hydra.main(version_base=None, config_name=config_name, config_path="../configs")
def main(cfg: DictConfig):
    tamp_planner = TAMPPlanner(
        algorithm=cfg.pddlstream.algorithm,
        unit=cfg.pddlstream.unit,
        deterministic=cfg.pddlstream.deternimistic,
        problem=cfg.pddlstream.problem,
        cfree=cfg.pddlstream.cfree,
        teleport=cfg.pddlstream.teleport
    )
    tamp_planner.execute(cfg.sim, cfg.curobo)


if __name__ == '__main__':
    main()
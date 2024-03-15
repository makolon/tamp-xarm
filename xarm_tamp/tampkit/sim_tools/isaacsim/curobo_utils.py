import carb
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction

from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.types.base import TensorDeviceType
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.rollout.rollout_base import Goal
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
# Model wrapper
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
# Reacher wrapper
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, \
    MotionGenPlanConfig, MotionGenResult
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

########################

def get_tensor_device_type():
    tensor_args = TensorDeviceType()
    return tensor_args

########################

def get_robot_cfg(cfg: dict):
    # load robot config from curobo content/config/robot
    return load_yaml(join_path(get_robot_configs_path(), cfg.yaml))["robot_cfg"]

def get_world_cfg(cfg: dict):
    # load world config from curobo content/config/world
    cuboid_cfg, mesh_cfg = [], []
    for _, val in cfg.items():
        if val.type == 'cuboid':
            world_cfg_cuboid = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), val.yaml))
            )
            cuboid_cfg.append(world_cfg_cuboid.cuboid)
        elif val.type == 'mesh':
            world_cfg_mesh = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), val.yaml))
            )
            mesh_cfg.append(world_cfg_mesh.mesh)

    world_cfg = WorldConfig(
        cuboid=cuboid_cfg,
        mesh=mesh_cfg,
    )
    return world_cfg

def get_motion_gen_plan_cfg(cfg: dict):
    plan_cfg = MotionGenPlanConfig(
        enable_graph=cfg.enable_graph,
        enable_graph_attempt=cfg.enable_graph_attempt,
        max_attempts=cfg.max_attempts,
        enable_finetune_trajopt=cfg.enable_finetune_trajopt,
        parallel_finetune=cfg.parallel_finetune,
    )
    return plan_cfg

########################
    
def get_robot_world_cfg(cfg: dict = None,
                        world_cfg: WorldConfig = None):
    robot_file = cfg.robot_world_cfg.robot_file
    robot_world_cfg = RobotWorldConfig.load_from_config(
        robot_file,
        world_cfg,
        collision_activation_distance=cfg.robot_world_cfg.activation_distance,
        collision_checker_type=CollisionCheckerType.BLOX \
            if cfg.robot_world_cfg.nvblox else CollisionCheckerType.MESH
    )
    return robot_world_cfg

########################

def get_ik_solver_cfg(cfg: dict,
                      robot_cfg: dict = None,
                      world_cfg: WorldConfig = None,
                      tensor_args: TensorDeviceType = None):
    if robot_cfg is None:
        robot_cfg = get_robot_cfg(cfg.robot_cfg)
    if world_cfg is None:
        world_cfg = get_world_cfg(cfg.world_cfg)
    if tensor_args is None:
        tensor_args = get_tensor_device_type()

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        position_threshold=cfg.ik_solver_cfg.position_threshold,
        rotation_threshold=cfg.ik_solver_cfg.rotation_threshold,
        num_seeds=cfg.ik_solver_cfg.num_seeds,
        self_collision_check=cfg.ik_solver_cfg.self_collision_check,
        self_collision_opt=cfg.ik_solver_cfg.self_collision_opt,
        use_cuda_graph=cfg.ik_solver_cfg.use_cuda_graph,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={
            "obb": cfg.ik_solver_cfg.n_obstacle_cuboids,
            "mesh": cfg.ik_solver_cfg.n_obstacle_mesh},
    )
    return ik_config

def get_motion_gen_cfg(cfg: dict,
                       robot_cfg: dict = None,
                       world_cfg: WorldConfig = None,
                       tensor_args: TensorDeviceType = None):
    if robot_cfg is None:
        robot_cfg = get_robot_cfg(cfg.robot_cfg)
    if world_cfg is None:
        world_cfg = get_world_cfg(cfg.world_cfg)
    if tensor_args is None:
        tensor_args = get_tensor_device_type()

    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=cfg.motion_gen_cfg.num_trajopt_seeds,
        num_graph_seeds=cfg.motion_gen_cfg.num_graph_seeds,
        interpolation_dt=cfg.motion_gen_cfg.interpolation_dt,
        collision_cache={
            "obb": cfg.motion_gen_cfg.n_obstacle_cuboids,
            "mesh": cfg.motion_gen_cfg.n_obstacle_mesh},
        optimize_dt=cfg.motion_gen_cfg.optimize_dt,
        trajopt_dt=cfg.motion_gen_cfg.trajopt_dt,
        trajopt_tsteps=cfg.motion_gen_cfg.trajopt_tsteps,
        trim_steps=cfg.motion_gen_cfg.trim_steps,
    )
    return motion_gen_cfg

def get_mpc_solver_cfg(cfg: dict,
                       robot_cfg: dict = None,
                       world_cfg: WorldConfig = None):
    if robot_cfg is None:
        robot_cfg = get_robot_cfg(cfg.robot_cfg)
    if world_cfg is None:
        world_cfg = get_world_cfg(cfg.world_cfg)

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=cfg.mpc_cfg.use_cuda_graph,
        use_cuda_graph_metrics=cfg.mpc_cfg.use_cuda_graph_metrics,
        use_cuda_graph_full_step=cfg.mpc_cfg.use_cuda_graph_full_step,
        self_collision_check=cfg.mpc_cfg.self_collision_check,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={
            "obb": cfg.mpc_cfg.n_obstackle_cuboids,
            "mesh": cfg.mpc_cfg.n_obstackle_mesh,
        },
        use_mppi=cfg.mpc_cfg.use_mppi,
        use_lbfgs=cfg.mpc_cfg.use_lbfgs,
        store_rollouts=cfg.mpc_cfg.store_rollouts,
        step_dt=cfg.mpc_cfg.step_dt
    )
    return mpc_config

########################

def get_robot_world(robot_world_cfg: RobotWorldConfig = None):
    if robot_world_cfg == None:
        raise ValueError("robot_world_cfg is not specified.")
    return RobotWorld(robot_world_cfg)

########################

def get_ik_solver(ik_cfg: IKSolverConfig = None):
    if ik_cfg == None:
        raise ValueError("ik_cfg is not specified.")
    return IKSolver(ik_cfg)

def get_motion_gen(motion_gen_cfg: MotionGenConfig = None):
    if motion_gen_cfg == None:
        raise ValueError("motion_gen_cfg is not specified.")
    return MotionGen(motion_gen_cfg)

def get_mpc_solver(mpc_cfg: MpcSolverConfig = None):
    if mpc_cfg == None:
        raise ValueError("mpc_cfg is not specified.")
    return MpcSolver(mpc_cfg)
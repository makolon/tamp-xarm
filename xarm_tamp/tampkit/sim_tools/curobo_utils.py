from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sphere_fit import SphereFitType
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
# Model wrapper
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
# Reacher wrapper
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

########################

def get_tensor_device_type():
    tensor_args = TensorDeviceType()
    return tensor_args

def get_usd_helper():
    usd_helper = UsdHelper()
    return usd_helper

########################

def get_robot_cfg(cfg: dict):
    # load robot config from curobo content/config/robot
    cfg_file = load_yaml(join_path(get_robot_configs_path(), cfg.yaml))["robot_cfg"]
    tensor_args = get_tensor_device_type()
    return RobotConfig.from_dict(cfg_file, tensor_args)

def get_world_cfg(cfg: dict):
    # load world config from curobo content/config/world
    world_cfg_cuboid, world_cfg_mesh = None, None
    for _, val in cfg.items():
        if val.type == 'cuboid':
            world_cfg_cuboid = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), val.yaml))
            )
            # lower cuboid a little
            world_cfg_cuboid.cuboid[0].pose[2] -= 0.02
        elif val.type == 'mesh':
            world_cfg_mesh = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), val.yaml))
            )
            # lower mesh to under ground
            world_cfg_mesh.mesh[0].pose[2] = -10.5

    if world_cfg_mesh is None:
        world_cfg = WorldConfig(
            cuboid=world_cfg_cuboid.cuboid
        )
    elif world_cfg_cuboid is None:
        world_cfg = WorldConfig(
            mesh=world_cfg_mesh.mesh
        )
    else:
        world_cfg = WorldConfig(
            cuboid=world_cfg_cuboid.cuboid,
            mesh=world_cfg_mesh.mesh,
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
    
def get_robot_world_cfg(cfg: dict,
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

def get_world_collision_cfg(cfg: dict,
                            world_cfg: WorldConfig = None,
                            tensor_args: TensorDeviceType = None):
    world_collision_cfg = WorldCollisionConfig.load_from_dict(
        world_coll_checker_dict=cfg.world_collision_cfg,
        world_model_dict=world_cfg,
        tensor_args=tensor_args,
    )
    return world_collision_cfg

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

    ik_cfg = IKSolverConfig.load_from_robot_config(
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
    return ik_cfg

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
        use_cuda_graph=cfg.motion_gen_cfg.use_cuda_graph,
        num_trajopt_seeds=cfg.motion_gen_cfg.num_trajopt_seeds,
        num_graph_seeds=cfg.motion_gen_cfg.num_graph_seeds,
        interpolation_dt=cfg.motion_gen_cfg.interpolation_dt,
        collision_cache={
            "obb": cfg.motion_gen_cfg.n_obstacle_cuboids,
            "mesh": cfg.motion_gen_cfg.n_obstacle_mesh},
        optimize_dt=cfg.motion_gen_cfg.optimize_dt,
        trajopt_dt=None, # cfg.motion_gen_cfg.trajopt_dt,
        trajopt_tsteps=cfg.motion_gen_cfg.trajopt_tsteps,
        trim_steps=None # cfg.motion_gen_cfg.trim_steps,
    )
    return motion_gen_cfg

def get_mpc_solver_cfg(cfg: dict,
                       robot_cfg: dict = None,
                       world_cfg: WorldConfig = None):
    if robot_cfg is None:
        robot_cfg = get_robot_cfg(cfg.robot_cfg)
    if world_cfg is None:
        world_cfg = get_world_cfg(cfg.world_cfg)

    mpc_cfg = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=cfg.mpc_cfg.use_cuda_graph,
        use_cuda_graph_metrics=cfg.mpc_cfg.use_cuda_graph_metrics,
        use_cuda_graph_full_step=cfg.mpc_cfg.use_cuda_graph_full_step,
        self_collision_check=cfg.mpc_cfg.self_collision_check,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={
            "obb": cfg.mpc_cfg.n_obstacle_cuboids,
            "mesh": cfg.mpc_cfg.n_obstacle_mesh,
        },
        use_mppi=cfg.mpc_cfg.use_mppi,
        use_lbfgs=cfg.mpc_cfg.use_lbfgs,
        store_rollouts=cfg.mpc_cfg.store_rollouts,
        step_dt=cfg.mpc_cfg.step_dt
    )
    return mpc_cfg

########################

def get_robot_world(robot_world_cfg: RobotWorldConfig = None):
    if robot_world_cfg is None:
        raise ValueError("robot_world_cfg is not specified.")
    return RobotWorld(robot_world_cfg)

########################

def get_collision_checker(world_collision_cfg: WorldCollisionConfig = None):
    if world_collision_cfg is None:
        raise ValueError("world_collision_cfg is not specified.")
    return create_collision_checker(world_collision_cfg)

########################

def get_ik_solver(ik_cfg: IKSolverConfig = None):
    if ik_cfg is None:
        raise ValueError("ik_cfg is not specified.")
    return IKSolver(ik_cfg)

def get_motion_gen(motion_gen_cfg: MotionGenConfig = None):
    if motion_gen_cfg is None:
        raise ValueError("motion_gen_cfg is not specified.")
    return MotionGen(motion_gen_cfg)

def get_mpc_solver(mpc_cfg: MpcSolverConfig = None):
    if mpc_cfg is None:
        raise ValueError("mpc_cfg is not specified.")
    return MpcSolver(mpc_cfg)

########################

def add_fixed_constraint(robot, obj, attach_fn):
    tensor_args = get_tensor_device_type()
    sim_js = robot.get_joints_state()
    cu_js = JointState(
        position=tensor_args.to_device(sim_js.positions),
        velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
        acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
        jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
        joint_names=robot._arm_dof_names
    )

    attach_fn(
        cu_js,
        [obj.prim_path],
        sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], tensor_args)
    )

def remove_fixed_constraint(detach_fn):
    detach_fn()

########################

def update_world(tamp_problem,
                 ignore_substring: str = None,
                 reference_prim_path: str = None):
    obstacles = tamp_problem.usd_helper.get_obstacles_from_stage(
        ignore_substring=ignore_substring, reference_prim_path=reference_prim_path
    ).get_collision_check_world()
    obstacles.add_obstacle(tamp_problem.world_cfg.cuboid[0])
    tamp_problem.motion_planner.update_world(obstacles)
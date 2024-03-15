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
    return load_yaml(join_path(get_robot_configs_path(), cfg.robot_cfg.name))["robot_cfg"]

def get_world_cfg(cfg: dict):
    # load world config from curobo content/config/world
    cuboid_cfg, mesh_cfg = [], []
    for _, val in cfg.world_cfg.items():
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

########################
    
def get_robot_world_cfg(cfg: dict = None,
                        world_cfg: WorldConfig = None):
    robot_file = cfg.robot_file
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
        position_threshold=cfg.inverse_kinematics.position_threshold,
        rotation_threshold=cfg.inverse_kinematics.rotation_threshold,
        num_seeds=cfg.inverse_kinematics.num_seeds,
        self_collision_check=cfg.inverse_kinematics.self_collision_check,
        self_collision_opt=cfg.inverse_kinematics.self_collision_opt,
        use_cuda_graph=cfg.inverse_kinematics.use_cuda_graph,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={
            "obb": cfg.inverse_kinematics.n_obstacle_cuboids,
            "mesh": cfg.inverse_kinematics.n_obstacle_mesh},
    )
    return ik_config

def get_motion_gen_plan_cfg(cfg: dict):
    plan_cfg = MotionGenPlanConfig(
        enable_graph=cfg.motion_generation_plan.enable_graph,
        enable_graph_attempt=cfg.motion_generation_plan.enable_graph_attempt,
        max_attempts=cfg.motion_generation_plan.max_attempts,
        enable_finetune_trajopt=cfg.motion_generation_plan.enable_finetune_trajopt,
        parallel_finetune=cfg.motion_generation_plan.parallel_finetune,
    )
    return plan_cfg

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
        num_trajopt_seeds=cfg.motion_generation.num_trajopt_seeds,
        num_graph_seeds=cfg.motion_generation.num_graph_seeds,
        interpolation_dt=cfg.motion_generation.interpolation_dt,
        collision_cache={
            "obb": cfg.motion_generation.n_obstacle_cuboids,
            "mesh": cfg.motion_generation.n_obstacle_mesh},
        optimize_dt=cfg.motion_generation.optimize_dt,
        trajopt_dt=cfg.motion_generation.trajopt_dt,
        trajopt_tsteps=cfg.motion_generation.trajopt_tsteps,
        trim_steps=cfg.motion_generation.trim_steps,
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
        use_cuda_graph=cfg.mpc.use_cuda_graph,
        use_cuda_graph_metrics=cfg.mpc.use_cuda_graph_metrics,
        use_cuda_graph_full_step=cfg.mpc.use_cuda_graph_full_step,
        self_collision_check=cfg.mpc.self_collision_check,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={
            "obb": cfg.mpc.n_obstackle_cuboids,
            "mesh": cfg.mpc.n_obstackle_mesh,
        },
        use_mppi=cfg.mpc.use_mppi,
        use_lbfgs=cfg.mpc.use_lbfgs,
        store_rollouts=cfg.mpc.store_rollouts,
        step_dt=cfg.mpc.step_dt
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

########################

class CuroboController(BaseController):
    def __init__(
        self,
        my_world: World,
        tensor_args,
        robot_cfg,
        world_cfg,
        plan_cfg,
        motion_gen_cfg,
        motion_gen,
        name: str = "curobo_controller",
    ) -> None:
        BaseController.__init__(self, name=name)

        self.my_world = my_world

        # warmup curobo instance
        self.init_curobo = False
        self.cmd_js_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.usd_help = UsdHelper()
        self.usd_help.load_stage(self.my_world.stage)
        self.cmd_plan = None
        self.cmd_idx = 0
        self.idx_list = None
        
        # Tensor args
        self.tensor_args = tensor_args
        # Config
        self.robot_cfg = robot_cfg
        self.world_cfg = world_cfg
        self.plan_cfg = plan_cfg
        self.motion_gen_cfg = motion_gen_cfg
        # Motion Generator
        self.motion_gen = motion_gen

    def attach_obj(
        self,
        sim_js: JointState,
        js_names: list,
        body_name: str
    ) -> None:

        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )

        self.motion_gen.attach_objects_to_robot(
            cu_js,
            [body_name],
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
        )

    def detach_obj(self) -> None:
        self.motion_gen.detach_object_from_robot()

    def plan(
        self,
        ee_translation_goal: np.array,
        ee_orientation_goal: np.array,
        sim_js: JointState,
        js_names: list,
    ) -> MotionGenResult:
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config.clone())
        return result

    def forward(
        self,
        ee_translation_goal: np.array,
        ee_orientation_goal: np.array,
        sim_js: JointState,
        js_names: list,
    ) -> ArticulationAction:

        if self.cmd_plan is None:
            self.cmd_idx = 0

            # compute curobo solution:
            result = self.plan(ee_translation_goal, ee_orientation_goal, sim_js, js_names)
            succ = result.success.item()
            if succ:
                cmd_plan = result.get_interpolated_plan()
                self.idx_list = [i for i in range(len(self.cmd_js_names))]
                self.cmd_plan = cmd_plan.get_ordered_joint_state(self.cmd_js_names)
            else:
                carb.log_warn("Plan did not converge to a solution.")
                return None

            cmd_state = self.cmd_plan[self.cmd_idx]
            self.cmd_idx += 1

            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy() * 0.0,
                joint_indices=self.idx_list,
            )
            if self.cmd_idx >= len(self.cmd_plan.position):
                self.cmd_idx = 0
                self.cmd_plan = None
        return art_action

    def reached_target(self, curr_ee_pose, target_ee_pose) -> bool:
        return True if np.linalg.norm(target_ee_pose - curr_ee_pose) < 0.04 \
            and (self.cmd_plan is None) else False

    def reset(
        self,
        ignore_substring: str,
        robot_prim_path: str,
    ) -> None:
        # init
        self.update(ignore_substring, robot_prim_path)
        self.init_curobo = True
        self.cmd_plan = None
        self.cmd_idx = 0

    def update(
        self,
        ignore_substring: str,
        robot_prim_path: str,
    ) -> None:
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=robot_prim_path
        ).get_collision_check_world()

        # add ground plane as it's not readable
        # TODO: fix
        obstacles.add_obstacle(self.world_cfg.cuboid[0])
        self.motion_gen.update_world(obstacles)
        self._world_cfg = obstacles
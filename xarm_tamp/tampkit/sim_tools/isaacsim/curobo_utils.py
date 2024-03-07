import carb
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.tasks import BaseTask
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

def get_robot_cfg(robot_cfg: dict):
    return load_yaml(join_path(robot_cfg.robot_cfg_path, robot_cfg.robot)["robot_cfg"])

def get_world_cfg(world_cfg: dict):
    cuboid_cfg, mesh_cfg = [], []
    for _, cfg in world_cfg.items():
        if cfg.type == 'cuboid':
            world_cfg_cuboid = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), cfg.yaml))
            )
            cuboid_cfg.append(world_cfg_cuboid.cuboid)
        elif cfg.type == 'mesh':
            world_cfg_mesh = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), cfg.yaml))
            )
            mesh_cfg.append(world_cfg_mesh.mesh)

    world_cfg = WorldConfig(
        cuboid=cuboid_cfg,
        mesh=mesh_cfg,
    )
    return world_cfg

########################
    
def get_robot_world_cfg(robot_cfg, world_cfg, sim_cfg):
    robot_file = robot_cfg.robot_path
    robot_world_cfg = RobotWorldConfig.load_from_config(
        robot_file,
        world_cfg,
        collision_activation_distance=sim_cfg.robot_world_cfg.activation_distance,
        collision_checker_type=CollisionCheckerType.BLOX \
            if sim_cfg.robot_world_cfg.nvblox else CollisionCheckerType.MESH
    )
    return robot_world_cfg

def get_motion_gen_plan_cfg(sim_cfg):
    plan_cfg = MotionGenPlanConfig(
        enable_graph=sim_cfg.motion_generation_plan.enable_graph,
        enable_graph_attempt=sim_cfg.motion_generation_plan.enable_graph_attempt,
        max_attempts=sim_cfg.motion_generation_plan.max_attempts,
        enable_finetune_trajopt=sim_cfg.motion_generation_plan.enable_finetune_trajopt,
        parallel_finetune=sim_cfg.motion_generation_plan.parallel_finetune,
    )
    return plan_cfg

def get_ik_solver_cfg(robot_cfg, world_cfg, tensor_args, sim_cfg):
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        position_threshold=sim_cfg.inverse_kinematics.position_threshold,
        rotation_threshold=sim_cfg.inverse_kinematics.rotation_threshold,
        num_seeds=sim_cfg.inverse_kinematics.num_seeds,
        self_collision_check=sim_cfg.inverse_kinematics.self_collision_check,
        self_collision_opt=sim_cfg.inverse_kinematics.self_collision_opt,
        use_cuda_graph=sim_cfg.inverse_kinematics.use_cuda_graph,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={
            "obb": sim_cfg.inverse_kinematics.n_obstacle_cuboids,
            "mesh": sim_cfg.inverse_kinematics.n_obstacle_mesh},
    )
    return ik_config

def get_motion_gen_cfg(robot_cfg, world_cfg, tensor_args, sim_cfg):
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=sim_cfg.motion_generation.num_trajopt_seeds,
        num_graph_seeds=sim_cfg.motion_generation.num_graph_seeds,
        interpolation_dt=sim_cfg.motion_generation.interpolation_dt,
        collision_cache={
            "obb": sim_cfg.motion_generation.n_obstacle_cuboids,
            "mesh": sim_cfg.motion_generation.n_obstacle_mesh},
        optimize_dt=sim_cfg.motion_generation.optimize_dt,
        trajopt_dt=sim_cfg.motion_generation.trajopt_dt,
        trajopt_tsteps=sim_cfg.motion_generation.trajopt_tsteps,
        trim_steps=sim_cfg.motion_generation.trim_steps,
    )
    return motion_gen_cfg

def get_mpc_solver_cfg(robot_cfg, world_cfg, sim_cfg):
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=sim_cfg.mpc.use_cuda_graph,
        use_cuda_graph_metrics=sim_cfg.mpc.use_cuda_graph_metrics,
        use_cuda_graph_full_step=sim_cfg.mpc.use_cuda_graph_full_step,
        self_collision_check=sim_cfg.mpc.self_collision_check,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={
            "obb": sim_cfg.mpc.n_obstackle_cuboids,
            "mesh": sim_cfg.mpc.n_obstackle_mesh,
        },
        use_mppi=sim_cfg.mpc.use_mppi,
        use_lbfgs=sim_cfg.mpc.use_lbfgs,
        store_rollouts=sim_cfg.mpc.store_rollouts,
        step_dt=sim_cfg.mpc.step_dt
    )
    return mpc_config

########################

def get_motion_gen(motion_gen_cfg: MotionGenConfig = None):
    if motion_gen_cfg == None:
        motion_gen_cfg = get_motion_gen_cfg()
    return MotionGen(motion_gen_cfg)

def get_robot_world(robot_world_cfg: RobotWorldConfig = None):
    if robot_world_cfg == None:
        robot_world_cfg = get_robot_world_cfg()
    return RobotWorld(robot_world_cfg)

def get_ik_solver(ik_cfg: IKSolverConfig = None):
    if ik_cfg == None:
        ik_cfg = get_ik_solver_cfg()
    return IKSolver(ik_cfg)

def get_mpc_solver(mpc_cfg: MpcSolverConfig = None):
    if mpc_cfg == None:
        mpc_cfg = get_mpc_solver_cfg()
    return MpcSolver(mpc_cfg)

########################

class CuroboController(BaseController):
    def __init__(
        self,
        my_world: World,
        my_task: BaseTask,
        name: str = "curobo_controller",
        constrain_grasp_approach: bool = False,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._save_log = False
        self.my_world = my_world
        self.my_task = my_task
        self._step_idx = 0
        n_obstacle_cuboids = 20
        n_obstacle_mesh = 2
        # warmup curobo instance
        self.usd_help = UsdHelper()
        self.init_curobo = False
        self.world_file = "collision_table.yml"
        self.cmd_js_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.tensor_args = TensorDeviceType()
        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.robot_cfg["kinematics"][
            "base_link"
        ] = "panda_link0"  # controls which frame the controller is controlling

        self.robot_cfg["kinematics"]["ee_link"] = "panda_hand"  # controls which frame the controller is controlling
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}
        self.robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_collision_mesh.yml"

        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        self._world_cfg_table = world_cfg_table

        world_cfg1 = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        ).get_mesh_world()
        world_cfg1.mesh[0].pose[2] = -10.5

        self._world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self._world_cfg,
            self.tensor_args,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.01,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            store_ik_debug=self._save_log,
            store_trajopt_debug=self._save_log,
            velocity_scale=0.75,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up...")
        self.motion_gen.warmup(parallel_finetune=True)
        pose_metric = None
        # if constrain_grasp_approach:
        #     pose_metric = PoseCostMetric.create_grasp_approach_metric(
        #         offset_position=0.1, tstep_fraction=0.6
        #     )

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            max_attempts=10,
            enable_graph_attempt=None,
            enable_finetune_trajopt=True,
            partial_ik_opt=False,
            parallel_finetune=True,
            pose_cost_metric=pose_metric,
        )
        self.usd_help.load_stage(self.my_world.stage)
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.idx_list = None

    def attach_obj(
        self,
        sim_js: JointState,
        js_names: list,
    ) -> None:
        cube_name = self.my_task.get_cube_prim(self.my_task.target_cube)

        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )

        self.motion_gen.attach_objects_to_robot(
            cu_js,
            [cube_name],
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
        if self._save_log:  # and not result.success.item(): # logging for debugging
            UsdHelper.write_motion_gen_log(
                result,
                {"robot_cfg": self.robot_cfg},
                self._world_cfg,
                cu_js,
                ik_goal,
                join_path("log/usd/", "cube") + "_debug",
                write_ik=False,
                write_trajopt=True,
                visualize_robot_spheres=True,
                link_spheres=self.motion_gen.kinematics.kinematics_config.link_spheres,
                grid_space=2,
                write_robot_usd_path="log/usd/assets",
            )
        return result

    def forward(
        self,
        sim_js: JointState,
        js_names: list,
    ) -> ArticulationAction:
        assert self.my_task.target_position is not None
        assert self.my_task.target_cube is not None

        if self.cmd_plan is None:
            self.cmd_idx = 0
            self._step_idx = 0
            # Set EE goals
            ee_translation_goal = self.my_task.target_position
            ee_orientation_goal = np.array([0, 0, -1, 0])
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
        if self._step_idx % 3 == 0:
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
        else:
            art_action = None
        self._step_idx += 1
        return art_action

    def reached_target(self, observations: dict) -> bool:
        curr_ee_position = observations["my_franka"]["end_effector_position"]
        if np.linalg.norm(
            self.my_task.target_position - curr_ee_position
        ) < 0.04 and (  # This is half gripper width, curobo succ threshold is 0.5 cm
            self.cmd_plan is None
        ):
            if self.my_task.cube_in_hand is None:
                print("reached picking target: ", self.my_task.target_cube)
            else:
                print("reached placing target: ", self.my_task.target_cube)
            return True
        else:
            return False

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
        # print("updating world...")
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=robot_prim_path
        ).get_collision_check_world()
        # add ground plane as it's not readable:
        obstacles.add_obstacle(self._world_cfg_table.cuboid[0])
        self.motion_gen.update_world(obstacles)
        self._world_cfg = obstacles
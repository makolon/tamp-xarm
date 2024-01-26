import carb
import hydra
import torch
from xarm_rl.robots.articulations.xarm import xArm
from xarm_rl.robots.articulations.views.xarm_view import xArmView
from xarm_rl.tasks.base.rl_task import RLTask
from xarm_rl.tasks.utils.scene_utils import spawn_dynamic_object, spawn_static_object
from xarm_rl.tasks.utils.ik_utils import DifferentialInverseKinematics, DifferentialInverseKinematicsCfg

from omni.isaac.core.prims import GeometryPrimView, RigidPrimView, XFormPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, print_stage_prim_paths
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.torch.rotations import quat_diff_rad, quat_mul, normalize
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.dynamic_control import _dynamic_control
from omni.physx.scripts import utils, physicsUtils
from pxr import Gf, Sdf, UsdGeom, PhysxSchema, UsdPhysics


class xArmFMBBaseTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._device = self._cfg["sim_device"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # Get dt for integrating velocity commands and checking limit violations
        self._control_frequency = torch.tensor(1/self._sim_config.task_config["env"]["controlFrequencyInv"], device=self._device)
        self._dt = torch.tensor(self._sim_config.task_config["sim"]["dt"] * self._sim_config.task_config["env"]["controlFrequencyInv"], device=self._device)

        # Set environment properties
        self._table_height = self._task_cfg["env"]["table_height"]
        self._table_width = self._task_cfg["env"]["table_width"]
        self._table_depth = self._task_cfg["env"]["table_depth"]

        # Set physics parameters for gripper
        self._gripper_static_friction = self._sim_config.task_config["sim"]["gripper"]["static_friction"]
        self._gripper_dynamic_friction = self._sim_config.task_config["sim"]["gripper"]["dynamic_friction"]
        self._gripper_restitution = self._sim_config.task_config["sim"]["gripper"]["restitution"]

        # Choose num_obs and num_actions based on task.
        self._num_observations = self._task_cfg["env"]["num_observations"]
        self._num_actions = self._task_cfg["env"]["num_actions"]

        # Set inverse kinematics configurations
        self._action_type = self._task_cfg["env"]["action_type"]
        self._target_space = self._task_cfg["env"]["target_space"]

        # Set learning hyper-parameters
        self._pick_success = self._task_cfg["rl"]["pick_threshold"]
        self._place_success = self._task_cfg["rl"]["place_threshold"]
        self._insert_success = self._task_cfg['rl']['insert_threshold']
        self._action_speed_scale = self._task_cfg["env"]["actionSpeedScale"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        # Set up environment from loaded demonstration
        self.set_up_environment()

        # Joint & body names
        self._arm_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self._gripper_names = ["hand_l_proximal_joint", "hand_r_proximal_joint"]

        # Values are set in post_reset after model is loaded
        self.arm_dof_idxs = []
        self.gripper_dof_idxs = []

        # Dof joint position limits
        self.arm_dof_lower = []
        self.arm_dof_upper = []
        self.gripper_dof_lower = []
        self.gripper_dof_upper = []

        # Reward settings
        self.is_collided = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.is_failure = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.is_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.pick_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.place_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.insert_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # Gripper settings
        self.gripper_close = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self.gripper_open = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self.gripper_hold = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

        RLTask.__init__(self, name, env)
        return

    def set_up_environment(self) -> None:
        # Environment object settings
        self._xarm_position = torch.tensor([0.0, 0.0, 0.0], device=self._device)
        self._xarm_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        self._table_position = torch.tensor([1.2, -0.2, self._table_height/2], device=self._device)
        self._table_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

    def set_up_scene(self, scene) -> None:
        # Create gripper materials
        self.create_gripper_material()

        # Set up scene
        super().set_up_scene(scene, replicate_physics=False)

        self.add_xarm()
        self.add_table()

        # Add robot to scene
        self._robots = xArmView(prim_paths_expr="/World/envs/.*/xarm", name="xarm_view")
        scene.add(self._robots)
        scene.add(self._robots._hands)
        scene.add(self._robots._lfingers)
        scene.add(self._robots._rfingers)
        scene.add(self._robots._fingertip_centered)

        # Add parts to scene
        self._block1 = RigidPrimView(prim_paths_expr="/World/envs/.*/shaft1/shaft1", name="shaft1_view", reset_xform_properties=False)
        self._block2 = RigidPrimView(prim_paths_expr="/World/envs/.*/shaft2/shaft2", name="shaft2_view", reset_xform_properties=False)
        self._block3 = RigidPrimView(prim_paths_expr="/World/envs/.*/gear1/gear1", name="gear1_view", reset_xform_properties=False)
        self._block4 = RigidPrimView(prim_paths_expr="/World/envs/.*/gear2/gear2", name="gear2_view", reset_xform_properties=False)
        self._block5 = RigidPrimView(prim_paths_expr="/World/envs/.*/gear3/gear3", name="gear3_view", reset_xform_properties=False)

        scene.add(self._shaft1)
        scene.add(self._shaft2)
        scene.add(self._gear1)
        scene.add(self._gear2)
        scene.add(self._gear3)

        self._parts = {'block1': self._block1, 'shaft2': self._block2,
                       'gear1': self._block3, 'gear2': self._block4, 'gear3': self._block5}

    def add_xarm(self):
        # Add xArm
        for i in range(self._num_envs):
            xarm = xArm(prim_path=f"/World/envs/env_{i}/xarm",
                       name="xarm",
                       translation=self._xarm_position,
                       orientation=self._xarm_rotation)
            self._sim_config.apply_articulation_settings("xarm", get_prim_at_path(xarm.prim_path), self._sim_config.parse_actor_config("xarm"))

            # Add physics material
            physicsUtils.add_physics_material_to_prim(
                self._stage,
                self._stage.GetPrimAtPath(f"/World/envs/env_{i}/xarm/hand_r_distal_link/collisions/mesh_0"), # TODO: Fix name
                self.gripperPhysicsMaterialPath
            )
            physicsUtils.add_physics_material_to_prim(
                self._stage,
                self._stage.GetPrimAtPath(f"/World/envs/env_{i}/xarm/hand_l_distal_link/collisions/mesh_0"), # TODO: Fix name
                self.gripperPhysicsMaterialPath
            )

    def add_table(self):
        # Add table
        for i in range(self._num_envs):
            table = FixedCuboid(prim_path=f"/World/envs/env_{i}/table",
                                name="table",
                                translation=self.env_info['initial_object_pose'][i]['table'][0],
                                orientation=self.init_static_orn['table'],
                                size=1.0,
                                color=torch.tensor([0.75, 0.75, 0.75]),
                                scale=torch.tensor([self._table_width, self._table_depth, self._table_height]))
            self._sim_config.apply_articulation_settings("table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table"))

    def create_gripper_material(self):
        self._stage = get_current_stage()
        self.gripperPhysicsMaterialPath = "/World/Physics_Materials/GripperMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.gripperPhysicsMaterialPath,
            density=self._gripper_density,
            staticFriction=self._gripper_static_friction,
            dynamicFriction=self._gripper_dynamic_friction,
            restitution=self._gripper_restitution
        )

    def get_observations(self):
        raise NotImplementedError()

    def pre_physics_step(self, actions) -> None:
        raise NotImplementedError()

    def reset_idx(self, env_ids):
        raise NotImplementedError()

    def post_reset(self):
        raise NotImplementedError()

    def calculate_metrics(self) -> None:
        raise NotImplementedError()

    def post_physics_step(self):
        raise NotImplementedError()

    def is_done(self) -> None:
        self.reset_buf = torch.where(
            self.progress_buf[:] >= self._max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def set_dof_idxs(self):
        [self.arm_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._arm_names]
        [self.gripper_dof_idxs.append(self._robots.get_dof_index(name)) for name in self._gripper_names]

        # Movable joints
        self.actuated_dof_indices = torch.LongTensor(self.arm_dof_idxs+self.gripper_dof_idxs).to(self._device)
        self.movable_dof_indices = torch.LongTensor(self.arm_dof_idxs).to(self._device)

    def set_dof_limits(self): # dof position limits
        # (num_envs, num_dofs, 2)
        dof_limits = self._robots.get_dof_limits()
        dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        # Set relevant joint position limit values
        self.arm_dof_lower = dof_limits_lower[self.arm_dof_idxs]
        self.arm_dof_upper = dof_limits_upper[self.arm_dof_idxs]
        self.gripper_dof_lower = dof_limits_lower[self.gripper_dof_idxs]
        self.gripper_dof_upper = dof_limits_upper[self.gripper_dof_idxs]

        self.robot_dof_lower_limits, self.robot_dof_upper_limits = torch.t(dof_limits[0].to(device=self._device))

    def set_default_state(self):
        # Start at 'home' positions
        self.arm_start = self.initial_dof_positions[:, self.arm_dof_idxs]
        self.gripper_start = self.initial_dof_positions[:, self.gripper_proximal_dof_idxs]

        # Set default joint state
        joint_states = self._robots.get_joints_default_state()
        jt_pos = joint_states.positions
        jt_pos[:, self.arm_dof_idxs] = self.arm_start.float()
        jt_pos[:, self.gripper_proximal_dof_idxs] = self.gripper_start.float()

        jt_vel = joint_states.velocities
        jt_vel[:, self.arm_dof_idxs] = torch.zeros_like(self.arm_start, device=self._device, dtype=torch.float)
        jt_vel[:, self.gripper_proximal_dof_idxs] = torch.zeros_like(self.gripper_start, device=self._device, dtype=torch.float)

        self._robots.set_joints_default_state(positions=jt_pos, velocities=jt_vel)

        # Initialize target positions
        self.dof_position_targets = jt_pos

    def set_joint_gains(self, joint_kps: torch.Tensor, joint_kds: torch.Tensor):
        self._robots.set_gains(kps=joint_kps, kds=joint_kds)

    def set_joint_frictions(self, friction_coefficients: torch.Tensor):
        self._robots.set_friction_coefficients(values=friction_coefficients)

    def _close_gripper(self, env_ids, sim_steps=None):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        # Set gripper target force
        gripper_dof_effort = torch.tensor([-30., -30.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, indices=env_ids_32, joint_indices=self.gripper_dof_idxs)

        # Step sim
        if sim_steps != None:
            for _ in range(sim_steps):
                SimulationContext.step(self._env._world, render=True)

        self.dof_position_targets[env_ids_64[:, None], self.gripper_dof_idxs] = self._robots.get_joint_positions(indices=env_ids_32, joint_indices=self.gripper_dof_idxs)
        self.gripper_hold[env_ids_64] = True

    def _open_gripper(self, env_ids, sim_steps=None):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        # Remove gripper force
        gripper_dof_effort = torch.tensor([0., 0.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, indices=env_ids_32, joint_indices=self.gripper_dof_idxs)

        # Set gripper target angle
        gripper_dof_pos = torch.tensor([0.5, 0.5], device=self._device)
        self._robots.set_joint_position_targets(gripper_dof_pos, indices=env_ids_32, joint_indices=self.gripper_dof_idxs)

        # Step sim
        if sim_steps != None:
            for _ in range(sim_steps):
                SimulationContext.step(self._env._world, render=True)

        self.dof_position_targets[env_ids_64[:, None], self.gripper_dof_idxs] = self._robots.get_joint_positions(indices=env_ids_32, joint_indices=self.gripper_dof_idxs)
        self.gripper_hold[env_ids_64] = False

    def _hold_gripper(self, env_ids):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        gripper_dof_effort = torch.tensor([-30., -30.], device=self._device)
        self._robots.set_joint_efforts(gripper_dof_effort, indices=env_ids_32, joint_indices=self.gripper_dof_idxs)
        self.dof_position_targets[env_ids_64[:, None], self.gripper_dof_idxs] = self._robots.get_joint_positions(indices=env_ids_32, joint_indices=self.gripper_dof_idxs)

    def load_exp_dataset(self):
        raise NotImplementedError()

    def set_ik_controller(self):
        command_type = "pose_rel" if self._action_type == 'relative' else "pose_abs"

        ik_control_cfg = DifferentialInverseKinematicsCfg(
            command_type=command_type,
            ik_method="dls",
            position_offset=(0.0, 0.0, 0.0),
            rotation_offset=(1.0, 0.0, 0.0, 0.0),
        )
        return DifferentialInverseKinematics(ik_control_cfg, self._num_envs, self._device)

    def enable_gravity(self):
        """Enable gravity."""

        gravity = [0.0, 0.0, -9.81]
        self._env._world._physics_sim_view.set_gravity(carb.Float3(gravity[0], gravity[1], gravity[2]))

    def disable_gravity(self):
        """Disable gravity."""

        gravity = [0.0, 0.0, 0.0]
        self._env._world._physics_sim_view.set_gravity(carb.Float3(gravity[0], gravity[1], gravity[2]))


@torch.jit.script
def norm_diff_pos(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    # Calculate norm
    diff_norm = torch.norm(p1 - p2, p=2, dim=-1)

    return diff_norm

@torch.jit.script
def norm_diff_rot(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # Calculate norm
    diff_norm1 = torch.norm(q1 - q2, p=2, dim=-1)
    diff_norm2 = torch.norm(q2 - q1, p=2, dim=-1)

    diff_norm = torch.min(diff_norm1, diff_norm2)

    return diff_norm

@torch.jit.script
def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

@torch.jit.script
def calc_diff_pos(p1, p2):
    return p1 - p2

@torch.jit.script
def calc_diff_rot(q1, q2):
    # Normalize the input quaternions
    q1 = normalize(q1)
    q2 = normalize(q2)

    # Calculate the quaternion product between q2 and the inverse of q1
    scaling = torch.tensor([1, -1, -1, -1], device=q1.device)
    q1_inv = q1 * scaling
    q_diff = quaternion_multiply(q2, q1_inv)

    return q_diff
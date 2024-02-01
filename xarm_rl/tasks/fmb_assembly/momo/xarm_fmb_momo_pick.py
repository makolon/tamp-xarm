import torch
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.maths import tensor_clamp
from omni.physx.scripts import physicsUtils
from xarm_rl.tasks.utils.scene_utils import spawn_dynamic_object, spawn_static_object
from xarm_rl.tasks.fmb_assembly.fmb_base.xarm_fmb_base import xArmFMBBaseTask
from xarm_rl.robots.articulations.views.xarm_view import xArmView
from pxr import Usd, UsdGeom


class xArmFMBMOMOPick(xArmFMBBaseTask):
    def __init__(self, name, sim_config, env) -> None:
        xArmFMBBaseTask.__init__(self, name, sim_config, env)

        self._parts = dict()
        self._parts_names = ['block1', 'block2', 'block3', 'block4']

        self._base_translation = torch.tensor([0.4, -0.1, 0.1], device=self._device)
        self._base_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)
        self._parts_translation = {
            'block1': torch.tensor([0.5, 0.1, 0.1], device=self._device),
            'block2': torch.tensor([0.6, 0.2, 0.1], device=self._device),
            'block3': torch.tensor([0.7, 0.1, 0.1], device=self._device),
            'block4': torch.tensor([0.5, 0.2, 0.1], device=self._device),
        }
        self._parts_orientation = {
            'block1': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'block2': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'block3': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
            'block4': torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device),
        }

    def set_up_scene(self, scene) -> None:
        # Set up scene
        super().set_up_scene(scene)

        # Add FMB MOMO objects
        self.add_base()
        self.add_parts()

        # Add parts to scene
        for obj_name in self._parts_names:
            parts = RigidPrimView(
                prim_paths_expr=f"/World/envs/.*/{obj_name}/{obj_name}",
                name=f"{obj_name}_view",
                reset_xform_properties=False
            )
            scene.add(parts)
            self._parts[obj_name] = parts

    def add_base(self) -> None:
        # Add static base
        base = spawn_static_object(name='base',
                                    task_name='fmb/momo/assembly1',
                                    prim_path=self.default_zero_env_path,
                                    object_translation=self._base_translation,
                                    object_orientation=self._base_orientation)
        self._sim_config.apply_articulation_settings('base',
                                            get_prim_at_path(base.prim_path),
                                            self._sim_config.parse_actor_config('base'))

    def add_parts(self) -> None:
        # Add movable parts
        for obj_name in self._parts_names:
            parts = spawn_dynamic_object(name=obj_name,
                                            task_name='fmb/momo/assembly1',
                                            prim_path=self.default_zero_env_path,
                                            object_translation=self._parts_translation[obj_name],
                                            object_orientation=self._parts_orientation[obj_name])
            self._sim_config.apply_articulation_settings(obj_name,
                                            get_prim_at_path(parts.prim_path),
                                            self._sim_config.parse_actor_config(obj_name))

    def get_observations(self) -> dict:
        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._robots._hands.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos
        end_effector_orientations = end_effector_orientations[:, [3, 0, 1, 2]]

        # Get dof positions
        dof_pos = self._robots.get_joint_positions(clone=False)

        self.obs_buf[..., 0:12] = dof_pos
        self.obs_buf[..., 12:15] = end_effector_positions
        self.obs_buf[..., 15:19] = end_effector_orientations

        observations = {self._robots.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Calculate targte pose
        self.actions = actions.clone().to(self._device)
        targets = self.xarm_dof_targets + self._dt * self.actions * self._action_scale

        # Clamp action
        self.xarm_dof_targets[:] = tensor_clamp(targets, self.xarm_dof_lower_limits, self.xarm_dof_upper_limits)

        # Set target pose
        env_ids_int32 = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
        self._robots.set_joint_position_targets(self.xarm_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        # Reset DOF states for robots in selected envs
        self._robots.set_joint_position_targets(self.initial_dof_positions, indices=env_ids_32)
        self._robots.set_joint_positions(self.initial_dof_positions, indices=env_ids_32)
        self._robots.set_joint_velocities(self.initial_dof_velocities, indices=env_ids_32)

        # Reset root state for robots in selected envs
        self._robots.set_world_poses(
            self.initial_robot_pos[env_ids_64],
            self.initial_robot_rot[env_ids_64],
            indices=env_ids_32
        )

        # Reset parts state
        for obj_name in self._parts_names:
            self._parts[obj_name].set_world_poses(
                self.initial_parts_pos[obj_name][env_ids_64],
                self.initial_parts_rot[obj_name][env_ids_64],
                indices=env_ids_32
            )

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.num_xarm_dofs = self._robots.num_dof
        self.xarm_dof_pos = torch.zeros((self._num_envs, self.num_xarm_dofs), device=self._device)
        
        dof_limits = self._robots.get_dof_limits()
        self.xarm_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.xarm_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.xarm_dof_targets = torch.zeros(
            (self._num_envs, self.num_xarm_dofs), dtype=torch.float, device=self._device
        )

        # Initialize robot positions / velocities
        self.initial_robot_pos, self.initial_robot_rot = self._robots.get_world_poses()
        self.initial_dof_positions = self._robots.get_joint_positions()
        self.initial_dof_velocities = torch.zeros_like(self.initial_dof_positions, device=self._device)

        # Initialize parts positions / velocities
        self.initial_parts_pos, self.initial_parts_rot = dict(), dict()
        for obj_name in self._parts_names:
            initial_parts_pos, initial_parts_rot = self._parts[obj_name].get_world_poses()
            self.initial_parts_pos[obj_name] = initial_parts_pos
            self.initial_parts_rot[obj_name] = initial_parts_rot

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def check_pick_success(self):
        return True

    def calculate_metrics(self) -> None:
        return 1
import math

import numpy as np
import torch
from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.maths import tensor_clamp
from xarm_rl.tasks.base.rl_task import RLTask
from xarm_rl.robots.articulations.xarm import xArm
from xarm_rl.robots.articulations.views.xarm_view import xArmView
from pxr import Usd, UsdGeom


class xArmFMBAssembly(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.dt = 1 / 60.0
        self._num_observations = 19
        self._num_actions = 12

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.num_props = self._task_cfg["env"]["numProps"]

    def set_up_scene(self, scene) -> None:
        self.get_xarm()
        self.get_props()

        super().set_up_scene(scene, filter_collisions=False)

        # Add xarm view to the scene
        self._xarms = xArmView(prim_paths_expr="/World/envs/.*/xarm6_with_gripper", name="xarm_view")
        scene.add(self._xarms)
        scene.add(self._xarms._hands)
        scene.add(self._xarms._lfingers)
        scene.add(self._xarms._rfingers)

        # Add props view to the scene
        self._props = RigidPrimView(prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False)
        scene.add(self._props)

        return

    def get_xarm(self):
        # NOTE: Basically self.default_zero_env_path is /World/envs/env_0
        xarm = xArm(prim_path=self.default_zero_env_path + "/xarm7_with_gripper", name="xarm")
        self._sim_config.apply_articulation_settings(
            "xarm", get_prim_at_path(xarm.prim_path), self._sim_config.parse_actor_config("xarm")
        )

    def get_props(self):
        prop = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/prop/prop_0",
            name="prop",
            translation=torch.tensor([0.2, 0.0, 0.1]),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            color=torch.tensor([0.2, 0.4, 0.6]),
            size=0.08,
            density=100.0,
        )
        self._sim_config.apply_articulation_settings(
            "prop", get_prim_at_path(prop.prim_path), self._sim_config.parse_actor_config("prop")
        )

    def get_observations(self) -> dict:
        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._xarms._hands.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos
        end_effector_orientations = end_effector_orientations[:, [3, 0, 1, 2]]

        # Get dof positions
        dof_pos = self._xarms.get_joint_positions(clone=False)

        self.obs_buf[..., 0:12] = dof_pos
        self.obs_buf[..., 12:15] = end_effector_positions
        self.obs_buf[..., 15:19] = end_effector_orientations

        observations = {self._xarms.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Calculate targte pose
        self.actions = actions.clone().to(self._device)
        targets = self.xarm_dof_targets + self.dt * self.actions * self.action_scale

        # Clamp action
        self.xarm_dof_targets[:] = tensor_clamp(targets, self.xarm_dof_lower_limits, self.xarm_dof_upper_limits)

        # Set target pose
        env_ids_int32 = torch.arange(self._xarms.count, dtype=torch.int32, device=self._device)
        self._xarms.set_joint_position_targets(self.xarm_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        # Reset DOF states for robots in selected envs
        self._xarms.set_joint_position_targets(self.initial_dof_positions, indices=env_ids_32)
        self._xarms.set_joint_positions(self.initial_dof_positions, indices=env_ids_32)
        self._xarms.set_joint_velocities(self.initial_dof_velocities, indices=env_ids_32)

        # Reset root state for robots in selected envs
        self._xarms.set_world_poses(
            self.initial_robot_pos[env_ids_64],
            self.initial_robot_rot[env_ids_64],
            indices=env_ids_32
        )

        # reset props
        self._props.set_world_poses(
            self.initial_prop_pos[env_ids_64],
            self.initial_prop_rot[env_ids_64],
            indices=env_ids_32
        )

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.num_xarm_dofs = self._xarms.num_dof
        self.xarm_dof_pos = torch.zeros((self._num_envs, self.num_xarm_dofs), device=self._device)
        
        dof_limits = self._xarms.get_dof_limits()
        self.xarm_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.xarm_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.xarm_dof_targets = torch.zeros(
            (self._num_envs, self.num_xarm_dofs), dtype=torch.float, device=self._device
        )

        self.initial_robot_pos, self.initial_robot_rot = self._xarms.get_world_poses()
        self.initial_dof_positions = self._xarms.get_joint_positions()
        self.initial_dof_velocities = torch.zeros_like(self.initial_dof_positions, device=self._device)

        self.initial_prop_pos, self.initial_prop_rot = self._props.get_world_poses()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # TODO: change
        self.rew_buf[:] = 1

    def is_done(self) -> None:
        # TODO: change
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )
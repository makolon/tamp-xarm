import torch
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.maths import tensor_clamp
from xarm_rl.tasks.fmb_assembly.fmb_base.xarm_fmb_base import xArmFMBBaseTask
from pxr import Usd, UsdGeom


class xArmFMBMOMOPick(xArmFMBBaseTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.parts_names = None

        xArmFMBBaseTask.__init__(self, name, sim_config, env)

    def set_up_scene(self, scene) -> None:
        # Set up scene
        super().set_up_scene(scene, replicate_physics=False)

        # Add FMB MOMO objects
        self.add_parts()

        # Add parts to scene
        for parts_name in self.parts_names:
            parts = RigidPrimView(
                prim_paths_expr=f"/World/envs/.*/{parts_name}/{parts_name}",
                name=f"{parts_name}_view",
                reset_xform_properties=False
            )
            scene.add(parts)
            self._parts[parts_name] = parts

    def add_parts(self) -> None:
        # Add movable gearbox parts
        for i in range(self._num_envs):
            object_translation = self.env_info['initial_object_pose'][i][self._dynamic_obj_names[i]][0]
            object_orientation = self.env_info['initial_object_pose'][i][self._dynamic_obj_names[i]][1]
            parts = spawn_dynamic_object(name=self._dynamic_obj_names[i],
                                         prim_path=f"/World/envs/env_{i}",
                                         object_translation=object_translation,
                                         object_orientation=object_orientation)
            self._sim_config.apply_articulation_settings(f"{self._dynamic_obj_names[i]}",
                                            get_prim_at_path(parts.prim_path),
                                            self._sim_config.parse_actor_config(self._dynamic_obj_names[i]))

            # Add physics material
            physicsUtils.add_physics_material_to_prim(
                self._stage,
                self._stage.GetPrimAtPath(f"/World/envs/env_{i}/{self._dynamic_obj_names[i]}/{self._dynamic_obj_names[i]}/collisions/mesh_0"),
                self.shaftPhysicsMaterialPath
            )

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
        self.num_xarm_dofs = self._robots.num_dof
        self.xarm_dof_pos = torch.zeros((self._num_envs, self.num_xarm_dofs), device=self._device)
        
        dof_limits = self._robots.get_dof_limits()
        self.xarm_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.xarm_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.xarm_dof_targets = torch.zeros(
            (self._num_envs, self.num_xarm_dofs), dtype=torch.float, device=self._device
        )

        self.initial_robot_pos, self.initial_robot_rot = self._robots.get_world_poses()
        self.initial_dof_positions = self._robots.get_joint_positions()
        self.initial_dof_velocities = torch.zeros_like(self.initial_dof_positions, device=self._device)

        self.initial_prop_pos, self.initial_prop_rot = self._props.get_world_poses()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        end_effector_positions, _ = self._robots._hands.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos

        # Get current parts positions and orientations
        curr_parts_positions = []
        for i in range(self._num_envs):
            parts_pos, _ = self._parts[self._dynamic_obj_names[i]].get_world_poses()
            parts_pos -= self._env_pos
            curr_parts_positions.append(parts_pos[i, :])

        curr_parts_positions = torch.stack(curr_parts_positions)

        # Distance from hand to the target object
        dist = torch.norm(end_effector_positions - curr_parts_positions, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + dist ** 2)
        dist_reward *= dist_reward
        self.rew_buf[:] = dist_reward * self._task_cfg['rl']['distance_scale']

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self._max_episode_length - 1)
        if is_last_step:
            # Check if block is picked up and close to target pose
            pick_success = self._check_pick_success()
            self.rew_buf[:] += pick_success * self._task_cfg['rl']['pick_success_bonus']
            self.extras['pick_successes'] = torch.mean(pick_success.float())
            self.pick_success = torch.where(
                pick_success[:] == 1,
                torch.ones_like(pick_success),
                -torch.ones_like(pick_success)
            )

import torch
from omni.isaac.core.prims import RigidPrimView, GeometryPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.maths import tensor_clamp
from omni.physx.scripts import physicsUtils
from xarm_rl.tasks.fmb_assembly.fmb_base.xarm_fmb_base import xArmFMBBaseTask
from xarm_rl.robots.articulations.views.xarm_view import xArmView
from pxr import Usd, UsdGeom


class xArmFMBMOMOReach(xArmFMBBaseTask):
    def __init__(self, name, sim_config, env) -> None:
        xArmFMBBaseTask.__init__(self, name, sim_config, env)

    def set_up_scene(self, scene) -> None:
        # Create gripper materials
        self.create_gripper_material()

        self.add_xarm()
        self.add_table()

        # Set up scene
        super().set_up_scene(scene, replicate_physics=False)

        # Add robot to scene
        self._robots = xArmView(prim_paths_expr="/World/envs/.*/xarm7", name="xarm_view")
        scene.add(self._robots)
        scene.add(self._robots._hands)
        scene.add(self._robots._lfingers)
        scene.add(self._robots._rfingers)
        scene.add(self._robots._fingertip_centered)

    def get_observations(self) -> dict:
        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._robots._hands.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos
        end_effector_orientations = end_effector_orientations[:, [3, 0, 1, 2]]

        # Get dof positions
        dof_pos = self._robots.get_joint_positions(joint_indices=self.movable_dof_indices, clone=False)

        self.obs_buf[..., 0:7] = dof_pos
        self.obs_buf[..., 7:10] = end_effector_positions
        self.obs_buf[..., 10:14] = end_effector_orientations

        observations = {self._robots.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Calculate targte pose
        ik_action = self._dt * self._action_scale * actions.to(self._device)
        self.ik_controller.set_command(ik_action)

        # Calculate end effector pose & jacobian
        ee_pos, ee_rot = self._robots._fingertip_centered.get_world_poses()
        ee_pos -= self._env_pos
        robot_jacobian = self._robots.get_jacobians(clone=False)[:, self._robots._body_indices['xarm_gripper_base_link']-1, :, self.movable_dof_indices]

        self.xarm_dof_targets[..., self.movable_dof_indices] = self._robots.get_joint_positions(joint_indices=self.movable_dof_indices)
        self.xarm_dof_targets[..., self.movable_dof_indices] += self.ik_controller.compute_delta(ee_pos, ee_rot, robot_jacobian)
        self.xarm_dof_targets[:] = tensor_clamp(
            self.xarm_dof_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

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

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.set_dof_idxs()
        self.set_dof_limits()

        # Initialize dof position & dof targets
        self.num_xarm_dofs = self._robots.num_dof
        self.xarm_dof_pos = torch.zeros((self._num_envs, self.num_xarm_dofs), device=self._device)
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

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""
        self.progress_buf[:] += 1
        if self._env._world.is_playing():
            # In this policy, episode length is constant
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.is_done()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def check_pick_success(self):
        return True

    def calculate_metrics(self) -> None:
        return 1
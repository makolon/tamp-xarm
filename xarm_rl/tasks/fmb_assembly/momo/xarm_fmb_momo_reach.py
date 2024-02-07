import torch
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import quat_mul, quat_conjugate
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.maths import tensor_clamp, torch_rand_float
from omni.physx.scripts import physicsUtils
from xarm_rl.tasks.fmb_assembly.fmb_base.xarm_fmb_base import xArmFMBBaseTask, axis_angle_from_quat
from xarm_rl.robots.articulations.views.xarm_view import xArmView
from pxr import Usd, UsdGeom


class xArmFMBMOMOReach(xArmFMBBaseTask):
    def __init__(self, name, sim_config, env) -> None:
        xArmFMBBaseTask.__init__(self, name, sim_config, env)

        self._ball_translation = torch.tensor([0.3, 0.0, 0.2], device=self._device)
        self._ball_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

    def set_up_scene(self, scene) -> None:
        # Create gripper materials
        self.create_gripper_material()

        self.add_ball()
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

        # Add ball to scene
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/Ball/ball", name="ball_view", reset_xform_properties=False)
        scene.add(self._balls)

    def add_ball(self):
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/Ball/ball",
            translation=self._ball_translation,
            orientation=self._ball_orientation,
            name="ball",
            radius=0.02,
            color=torch.tensor([0.2, 0.4, 0.6])
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))

    def get_observations(self) -> dict:
        # Get ball positions and orientations
        ball_positions, ball_orientations = self._balls.get_world_poses(clone=False)
        ball_positions  = ball_positions[:, 0:3] - self._env_pos

        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._robots._fingertip_centered.get_world_poses(clone=False)
        end_effector_positions = end_effector_positions[:, 0:3] - self._env_pos

        # Get dof positions
        dof_pos = self._robots.get_joint_positions(joint_indices=self.movable_dof_indices, clone=False)

        # Calculate position and orientation difference
        diff_pos = end_effector_positions - ball_positions
        ball_quat_norm = quat_mul(ball_orientations, quat_conjugate(ball_orientations))[:, 0]
        ball_quat_inv = quat_conjugate(ball_orientations) / ball_quat_norm.unsqueeze(-1)
        diff_quat = quat_mul(end_effector_orientations, ball_quat_inv).to(self._device)
        diff_rot = axis_angle_from_quat(diff_quat)

        self.obs_buf[..., 0:7] = dof_pos
        self.obs_buf[..., 7:10] = diff_pos
        self.obs_buf[..., 10:13] = diff_rot

        observations = {self._robots.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Calculate targte pose
        self.xarm_dof_targets[..., self.movable_dof_indices] = self._robots.get_joint_positions(joint_indices=self.movable_dof_indices)
        self.xarm_dof_targets[..., self.movable_dof_indices] += self._dt * self._action_scale * actions.to(self._device)
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

        ball_x = torch_rand_float(0.2, 0.5, (self._num_envs, 1), self._device)
        ball_y = torch_rand_float(-0.3, 0.3, (self._num_envs, 1), self._device)
        ball_z = torch_rand_float(0.2, 0.5, (self._num_envs, 1), self._device)
        ball_pos = torch.cat([ball_x, ball_y, ball_z], dim=1)
        ball_pos += self._env_pos[:, 0:3]
        ball_rot = self.initial_ball_rot.clone()

        self._balls.set_world_poses(ball_pos[env_ids_64], ball_rot[env_ids_64], indices=env_ids_32)

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

        # Initialize ball positions / velocities
        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # Distance from hand to the ball
        dist = torch.norm(self.obs_buf[..., 7:10], p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + dist ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(dist <= 0.1, dist_reward * 5, dist_reward)

        self.rew_buf[:] = dist_reward

        is_last_step = (self.progress_buf[0] >= self._max_episode_length - 1)
        if is_last_step:
            reach_success = self.check_reach_success()
            self.rew_buf[:] += reach_success * self._task_cfg['rl']['reach_success_bonus']
            self.extras['reach_success'] = torch.mean(reach_success.float())

    def check_reach_success(self):
        dist = torch.norm(self.obs_buf[..., 7:10], p=2, dim=-1)

        reach_success = torch.where(
            dist < torch.tensor([0.03], device=self._device),
            torch.ones((self._num_envs,), device=self._device),
            torch.zeros((self._num_envs,), device=self._device)
        )
        return reach_success

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
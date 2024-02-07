import torch
import numpy as np
from gym import spaces
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import quat_mul, quat_conjugate, quat_from_euler_xyz
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.maths import tensor_clamp, torch_rand_float
from omni.physx.scripts import physicsUtils
from xarm_rl.tasks.fmb_assembly.fmb_base.xarm_fmb_base import xArmFMBBaseTask, axis_angle_from_quat
from xarm_rl.robots.articulations.views.xarm_view import xArmView
from pxr import Usd, UsdGeom


class xArmFMBMOMOCameraReach(xArmFMBBaseTask):
    def __init__(self, name, sim_config, env) -> None:
        xArmFMBBaseTask.__init__(self, name, sim_config, env)

        self._box_translation = torch.tensor([0.3, 0.0, 0.2], device=self._device)
        self._box_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

        self._camera_width = self._task_cfg["env"]["cameraWidth"]
        self._camera_height = self._task_cfg["env"]["cameraHeight"]
        self._camera_position = (-4.2, 0.0, 3.0)
        self._camera_rotation = (0, -6.1155, -180)

        self.observation_space = spaces.Box(
            np.ones((self._camera_width, self._camera_height, 3), dtype=np.float32) * -np.Inf,
            np.ones((self._camera_width, self._camera_height, 3), dtype=np.float32) * np.Inf
        )

    def set_up_scene(self, scene) -> None:
        # Create gripper materials
        self.create_gripper_material()

        self.add_box()
        self.add_xarm()
        self.add_table()
        self.add_camera()

        # Set up scene
        super().set_up_scene(scene, replicate_physics=False)

        # start replicator to capture image data
        self.rep.orchestrator._orchestrator._is_started = True

        # Set up camera
        self.set_up_camera()

        # Add robot to scene
        self._robots = xArmView(prim_paths_expr="/World/envs/.*/xarm7", name="xarm_view")
        scene.add(self._robots)
        scene.add(self._robots._hands)
        scene.add(self._robots._lfingers)
        scene.add(self._robots._rfingers)
        scene.add(self._robots._fingertip_centered)

        # Add box to scene
        self._boxes = RigidPrimView(prim_paths_expr="/World/envs/.*/Box/box", name="box_view", reset_xform_properties=False)
        scene.add(self._boxes)

    def add_box(self):
        box = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/Box/box",
            translation=self._box_translation,
            orientation=self._box_orientation,
            name="box",
            scale=torch.tensor([0.03, 0.03, 0.06]),
            color=torch.tensor([0.2, 0.4, 0.6])
        )
        self._sim_config.apply_articulation_settings("box", get_prim_at_path(box.prim_path), self._sim_config.parse_actor_config("box"))

    def add_camera(self) -> None:
        stage = get_current_stage()
        camera_path = self.default_zero_env_path + "/Camera"
        camera_xform = stage.DefinePrim(f'{camera_path}_Xform', 'Xform')

        # Set up transforms for parent and camera prims
        UsdGeom.Xformable(camera_xform).AddTranslateOp()
        UsdGeom.Xformable(camera_xform).AddRotateXYZOp()

        camera_xform.GetAttribute('xformOp:translate').Set(self._camera_position)
        camera_xform.GetAttribute('xformOp:rotateXYZ').Set(self._camera_rotation)
        camera = stage.DefinePrim(f'{camera_path}_Xform/Camera', 'Camera')
        UsdGeom.Xformable(camera).AddRotateXYZOp()
        camera.GetAttribute("xformOp:rotateXYZ").Set((90, 0, 90))

        # Set camera properties
        camera.GetAttribute('focalLength').Set(24)
        camera.GetAttribute('focusDistance').Set(400)

        # Hide other environments in the background
        camera.GetAttribute("clippingRange").Set((0.01, 20.0))

    def set_up_camera(self):
        # Set up cameras
        self.render_products = []
        for i in range(self._num_envs):
            camera_path = f"/World/envs/env_{i}/Camera_Xform/Camera"
            render_product = self.rep.create.render_product(camera_path, resolution=(self._camera_width, self._camera_height))
            self.render_products.append(render_product)

        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = self.PytorchListener()
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda")
        self.pytorch_writer.attach(self.render_products)

    def get_observations(self) -> dict:
        # Retrieve RGB data from all render products
        images = self.pytorch_listener.get_rgb_data()
        if images is not None:
            self.obs_buf = torch.swapaxes(images, 1, 3).clone().float() / 255.0

        observations = {self._robots.name: {"obs_buf": self.obs_buf}}
        return observations

    def get_states(self):
        # Get box positions and orientations
        box_positions, box_orientations = self._boxes.get_world_poses(clone=False)
        box_positions -= self._env_pos[:, 0:3]

        # Get end effector positions and orientations
        end_effector_positions, end_effector_orientations = self._robots._fingertip_centered.get_world_poses(clone=False)
        end_effector_positions -= self._env_pos[:, 0:3]

        # Get dof positions
        dof_pos = self._robots.get_joint_positions(joint_indices=self.movable_dof_indices, clone=False)

        # Calculate position and orientation difference
        diff_pos = end_effector_positions - box_positions
        box_quat_norm = quat_mul(box_orientations, quat_conjugate(box_orientations))[:, 0]
        box_quat_inv = quat_conjugate(box_orientations) / box_quat_norm.unsqueeze(-1)
        diff_quat = quat_mul(end_effector_orientations, box_quat_inv).to(self._device)
        diff_rot = axis_angle_from_quat(diff_quat)

        self.states_buf[..., 0:7] = dof_pos
        self.states_buf[..., 7:10] = diff_pos
        self.states_buf[..., 10:13] = diff_rot

        return self.states_buf

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

        # Randomize box position
        box_px = torch_rand_float(0.2, 0.5, (self._num_envs, 1), self._device)
        box_py = torch_rand_float(-0.3, 0.3, (self._num_envs, 1), self._device)
        box_pz = torch_rand_float(0.2, 0.4, (self._num_envs, 1), self._device)
        box_pos = torch.cat([box_px, box_py, box_pz], dim=1)
        box_pos += self._env_pos[:, 0:3]

        # Randomize box rotation
        box_rx = torch_rand_float(-np.pi, np.pi, (self._num_envs, 1), self._device).squeeze(-1)
        box_ry = torch_rand_float(-np.pi, np.pi, (self._num_envs, 1), self._device).squeeze(-1)
        box_rz = torch_rand_float(-np.pi, np.pi, (self._num_envs, 1), self._device).squeeze(-1)
        box_rot = quat_from_euler_xyz(box_rx, box_ry, box_rz)

        # Reset robo state for boxes in selected envs
        self._boxes.set_world_poses(box_pos[env_ids_64], box_rot[env_ids_64], indices=env_ids_32)

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

        # Initialize box positions / velocities
        self.initial_box_pos, self.initial_box_rot = self._boxes.get_world_poses()

        # Randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # Distance from hand to the box
        dist = torch.norm(self.states_buf[..., 7:10], p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + dist ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(dist <= 0.1, dist_reward * 2, dist_reward)

        self.rew_buf[:] = dist_reward

        is_last_step = (self.progress_buf[0] >= self._max_episode_length - 1)
        if is_last_step:
            reach_success = self.check_reach_success()
            self.rew_buf[:] += reach_success.float() * self._task_cfg['rl']['reach_success_bonus']

    def check_reach_success(self):
        pos_dist = torch.norm(self.states_buf[..., 7:10], p=2, dim=-1)
        rot_dist = torch.norm(self.states_buf[..., 10:13], p=2, dim=-1)

        reach_success = torch.where(pos_dist < torch.tensor([0.03], device=self._device), True, False)
        reach_success = torch.where(rot_dist < torch.tensor([0.03], device=self._device), reach_success, False)
        return reach_success

    def get_extras(self):
        is_last_step = (self.progress_buf[0] >= self._max_episode_length - 1)
        if is_last_step:
            reach_success = self.check_reach_success()
            self.extras['reach_success'] = torch.mean(reach_success.float())

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
import carb
import math
import torch
import numpy as np
from typing import Optional, List
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from xarm_rl.tasks.utils.usd_utils import set_drive # TODO: fix this
from pxr import PhysxSchema


class xArm(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "xarm7",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name

        if not prim.IsValid():
            # TODO: fix
            usd_path = "/home/makolon/Codes/tamp-xarm/xarm_tamp/tampkit/models/usd/xarm_with_sphere_collision/xarm7.usd"
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

            # end effector
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/fingertip_centered"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name

            # gripper
            if gripper_dof_names is None:
                gripper_dof_names = ["left_drive_joint", "right_drive_joint"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
            if gripper_closed_position is None:
                gripper_closed_position = np.array([0.0, 0.0])
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=translation,
            orientation=orientation,
            articulation_controller=None
        )

        self.arm_dof_idxs = []
        self.gripper_dof_idxs = []

        # if gripper_dof_names is not None:
        #     if deltas is None:
        #         deltas = np.array([0.05, 0.05]) / get_stage_units()
        #     self._gripper = ParallelGripper(
        #         end_effector_prim_path=self._end_effector_prim_path,
        #         joint_prim_names=gripper_dof_names,
        #         joint_opened_positions=gripper_open_position,
        #         joint_closed_positions=gripper_closed_position,
        #         action_deltas=deltas,
        #     )

    def set_drive_property(self):
        dof_paths = [
            "link_base/joint1",
            "link1/joint2",
            "link2/joint3",
            "link3/joint4",
            "link4/joint5",
            "link5/joint6",
            "link6/joint7",
            "xarm_gripper_base_link/left_drive_joint",
            "xarm_gripper_base_link/right_drive_joint",
        ]

        drive_type = ["angular"] * 9
        default_dof_pos = [math.degrees(x) for x in [0.0, -0.698, 0.0, 0.349, 0.0, 1.047, 0.0, 0.0, 0.0]]
        stiffness = [400 * np.pi / 180] * 7 + [80 * np.pi / 180] * 2
        damping = [80 * np.pi / 180] * 7 + [160 * np.pi / 180] * 2
        max_force = [87, 87, 87, 87, 12, 12, 12, 200, 200]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 2.0, 2.0]]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )

    def set_xarm_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)

    def set_dof_idxs(self):
        [self.arm_dof_idxs.append(self._articulation_view.get_dof_index(name)) for name in self._arm_names]
        [self.gripper_dof_idxs.append(self._articulation_view.get_dof_index(name)) for name in self._gripper_proximal_names]

        # Movable joints
        self.actuated_dof_indices = torch.LongTensor(self.arm_dof_idxs+self.gripper_dof_idxs)
        self.movable_dof_indices = torch.LongTensor(self.arm_dof_idxs)

    @property
    def arm_joints(self):
        self.arm_dof_idxs

    @property
    def gripper_joints(self):
        self.gripper_dof_idxs

    @property
    def end_effector(self) -> RigidPrim:
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name="fingertip_centered")
        self._end_effector.initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )

    def post_reset(self) -> None:
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
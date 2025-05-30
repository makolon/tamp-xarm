import carb
import math
import torch
import numpy as np
from typing import Optional
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from xarm_rl.tasks.utils.usd_utils import set_drive
from xarm_rl.utils.files import get_usd_path
from pxr import PhysxSchema


class xArm(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "xarm7",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            self._usd_path = (get_usd_path() / 'xarm7' / 'xarm_instanceable' / 'xarm7.usd').as_posix()

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

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
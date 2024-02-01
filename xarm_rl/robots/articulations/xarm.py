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
        name: Optional[str] = "xarm",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            self._usd_path = (get_usd_path() / 'xarm7' / 'xarm7.usd').as_posix()

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

    def set_xarm_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)
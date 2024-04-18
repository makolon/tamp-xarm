import carb
import math
import torch
import numpy as np
from typing import Optional
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage


class Block(XFormPrim):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "block1",
        type: Optional[str] = 'assembly1',
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ) -> None:
        
        self._usd_path = usd_path
        self._name = name
        
        if self._usd_path is None:
            # TODO: fix this
            self._usd_path = f"/home/makolon/Codes/tamp-xarm/xarm_rl/models/usd/fmb/momo/{type}/{name}/{name}.usd"

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
        )
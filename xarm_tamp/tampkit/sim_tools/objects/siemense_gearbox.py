import numpy as np
from typing import Optional
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage


class Shaft(XFormPrim):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "shaft1",
        usd_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ) -> None:
        
        self.usd_path = usd_path
        self.urdf_path = urdf_path
        
        if self.usd_path is None:
            # TODO: fix this
            self.usd_path = f"/home/makolon/Codes/tamp-xarm/xarm_rl/models/usd/siemense_gearbox/{name}/{name}.usd"
        if self.urdf_path is None:
            # TODO: fix this
            self.urdf_path = f"/home/makolon/Codes/tamp-xarm/xarm_rl/models/urdf/siemense_gearbox/{name}/{name}.urdf"

        add_reference_to_stage(self.usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            scale=scale,
        )

class Gear(XFormPrim):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "gear1",
        usd_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ) -> None:
        
        self.usd_path = usd_path
        self.urdf_path = urdf_path
        
        if self.usd_path is None:
            # TODO: fix this
            self.usd_path = f"/home/makolon/Codes/tamp-xarm/xarm_rl/models/usd/siemense_gearbox/{name}/{name}.usd"
        if self.urdf_path is None:
            # TODO: fix this
            self.urdf_path = f"/home/makolon/Codes/tamp-xarm/xarm_rl/models/urdf/siemense_gearbox/{name}/{name}.urdf"

        add_reference_to_stage(self.usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            scale=scale,
        )

class GearboxBase(XFormPrim):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "gearbox_base",
        usd_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ) -> None:
        
        self.usd_path = usd_path
        self.urdf_path = urdf_path
        
        if self.usd_path is None:
            # TODO: fix this
            self.usd_path = f"/home/makolon/Codes/tamp-xarm/xarm_rl/models/usd/siemense_gearbox/{name}/{name}.usd"
        if self.urdf_path is None:
            # TODO: fix this
            self.urdf_path = f"/home/makolon/Codes/tamp-xarm/xarm_rl/models/urdf/siemense_gearbox/{name}/{name}.urdf"

        add_reference_to_stage(self.usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            scale=scale,
        )
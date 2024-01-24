from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class xArmView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "xArmView",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        # Add hand as RigidPrimView
        self._hands = RigidPrimView(
            prim_paths_expr="/World/envs/.*/xarm7_with_gripper/xarm_gripper_base_link",
            name="hands_view",
            reset_xform_properties=False
        )
        # Add left finger as RigidPrimView
        self._lfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/xarm7_with_gripper/left_finger",
            name="lfingers_view",
            reset_xform_properties=False
        )
        # Add right finger as RigidPrimView
        self._rfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/xarm7_with_gripper/right_finger",
            name="rfingers_view",
            reset_xform_properties=False,
        )

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self._gripper_indices = [self.get_dof_index("left_finger_joint"), self.get_dof_index("right_finger_joint")]

    @property
    def gripper_indices(self):
        return self._gripper_indices
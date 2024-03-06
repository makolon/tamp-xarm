from dataclasses import dataclass, field
from typing import Union


@dataclass
class Problem:
    # Objects
    robot: object
    arms: tuple = field(default_factory=tuple)
    movable: tuple = field(default_factory=tuple)
    bodies: tuple = field(default_factory=tuple)
    fixed: tuple = field(default_factory=tuple)
    holes: tuple = field(default_factory=tuple)
    surfaces: tuple = field(default_factory=tuple)
    sinks: tuple = field(default_factory=tuple)
    stoves: tuple = field(default_factory=tuple)
    buttons: tuple = field(default_factory=tuple)
    # Static predicates
    init_placeable: tuple = field(default_factory=tuple)
    init_insertable: tuple = field(default_factory=tuple)
    goal_conf: object = None
    goal_holding: tuple = field(default_factory=tuple)
    goal_on: tuple = field(default_factory=tuple)
    goal_inserted: tuple = field(default_factory=tuple)
    goal_cleaned: tuple = field(default_factory=tuple)
    goal_cooked: tuple = field(default_factory=tuple)
    body_names: dict = field(default_factory=dict)
    body_types: list = field(default_factory=list)
    base_limits: object = None
    gripper: object = None
    costs: bool = False
    # Config
    robot_cfg: dict = field(default_factory=dict)
    world_cfg: dict = field(default_factory=dict)
    plan_cfg: dict = field(default_factory=dict)
    # Planner
    ik_solver: object = None
    motion_planner: object = None
    mpc: object = None

    def get_gripper(self, gripper_name: str, visual: bool=True):
        if self.gripper is None:
            import omni.isaac.core.utils.prims as prims_utils
            # self.gripper = prims_utils.create_prim(
            #     prim_path=f"/World/{self.robot.name}/{gripper_name}",
            #     prim_type="Xform",
            #     usd_path=""
            # )
            # TODO: fix this
            self.gripper = prims_utils.define_prim(
                prim_path=f"/World/{self.robot.name}/{gripper_name}",
                prim_type="Xform",
            )
        return self.gripper

    def remove_gripper(self):
        if self.gripper is not None:
            import omni.isaac.core.utils.prims as prims_utils
            # TODO: must be prim_path
            prims_utils.delete_prim(self.gripper.prim_path)
            self.gripper = None

    def __repr__(self):
        return repr(self.__dict__)
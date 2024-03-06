import torch
import argparse

from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless_mode", type=str, default=None, help="To run headless, use one of [native, websocket], webrtc might not work.")
parser.add_argument("--width", type=int, default=1920, help="Set window width")
parser.add_argument("--height", type=int, default=1080, help="Set window height")
parser.add_argument("--robot", type=str, default="xarm", help="robot configuration to load")
args = parser.parse_args()

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": args.width if args.width is not None else 1920,
        "height": args.height if args.height is not None else 1080,
    }
)

# Third party
import carb
import numpy as np
from typing import Union, Optional, Tuple, List
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from sim_tools.robots import xarm
from xarm_rl.tasks.utils.scene_utils import spawn_dynamic_object, spawn_static_object


def connect():
    global simulation_app
    return simulation_app

def disconnect():
    global simulation_app
    simulation_app.close()

### Create simulation environment API

def create_world():
    my_world = World(stage_units_in_meters=1.0)
    return my_world

def create_floor(world, plane_cfg):
    plane = world.scene.add_default_ground_plane(
        size=plane_cfg.size,
        static_friction=plane_cfg.static_friction,
        dynamic_friction=plane_cfg.dynamic_friction,
        restitution=plane_cfg.restitution,
    )
    return plane

def create_table(table_cfg):
    table = cuboid.VisualCuboid(
        "/World/table",
        position=np.array([table_cfg.position]),
        orientation=np.array([table_cfg.orientation]),
        color=np.array([table_cfg.color]),
        size=table_cfg.size,
    )
    return table

def create_robot(robot_cfg):
    if "xarm" in robot_cfg.name:
        robot = xarm.xArm(
            "/World/xarm7",
            position=np.array([robot_cfg.position]),
            orientation=np.array([robot_cfg.orientation]),
        )
    else:
        raise ValueError("Need to give known robot_name")
    return robot

def create_fmb(fmb_cfg):
    # NOTE: deprecated
    # block = fmb_momo.Block(
    #     f"/World/{fmb_cfg.name}",
    #     position=np.array([fmb_cfg.position]),
    #     orientation=np.array([fmb_cfg.orientation])
    # )
    block = spawn_dynamic_object(
        name=fmb_cfg.name,
        task_name=fmb_cfg.task_name,
        prim_path=f"/World/{fmb_cfg.name}",
        object_translation=fmb_cfg.translation,
        object_orientation=fmb_cfg.orientation,
    )
    return block


### Getter API

def get_initial_conf(robot: Robot, joint_indices: Optional[Union[List, numpy.ndarray, torch.Tensor]]):
    default_pos, default_vel = robot.get_joints_default_state(joint_indices=joint_indices)
    if default_pos is None:
        default_pos = robot.get_joint_positions(joint_indices=joint_indices)
    return default_pos

def get_group_conf(body_id, body_name):
    return 

def get_target_path():
    pass

def get_gripper_joints():
    pass

def get_joint_positions(robot: Robot, joint_indices: Optional[Union[List, numpy.ndarray, torch.Tensor]]):
    joint_positions = robot.get_jonit_positions(joint_indices=joint_indices)
    return joint_positions

def get_link_pose(prim: Tuple[np.ndarray, np.ndarray]):
    if prim.is_valid():
        pos, orn = prim.get_local_pose()
    else:
        raise NotImplementedError()
    return pos, orn

def get_min_limit(robot: Robot) -> np.ndarray:
    return robot.dof_properties.lower

def get_max_limit(robot: Robot) -> np.ndarray:
    return robot.dof_properties.upper

def get_name():
    pass

def get_pose():
    pass

def get_extend_fn(robot: Robot, joints):
    def fn(start_conf: torch.Tensor, end_conf: torch.Tensor):
        pass
    return fn

def get_distance_fn():
    def fn(q1: torch.Tensor, q2: torch.Tensor):
        pass
    return fn

def get_active_joints(robot: Robot):
    joint_indices = robot.get_active_joints()
    return joint_indices


### Setter API

def set_joint_positions(robot: Robot,
                        positions: Optional[Union[np.ndarray, torch.Tensor]],
                        joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
    if joint_indices is None:
        joint_indices = [robot.get_joint_index(name) \
            for name in robot.active_joints] # TODO: fix?
    robot.set_joint_positions(positions, joint_indices)

def set_pose():
    pass

def set_point(obj: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
              translation: Optional[Union[np.ndarray, torch.Tensor]],
              orientation: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
    obj.set_local_pose(translation=translation, orientation=orientation,)

def set_arm_conf(robot: Robot,
                 conf: Optional[Union[np.ndarray, torch.Tensor]],
                 joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
    if joint_indices is None:
        joint_indices = [robot.get_joint_index(name) \
            for name in robot.active_joints] # TODO: fix?
    robot.set_joint_positions(conf, joint_indices=joint_indices)


### Utility API

def add_segments():
    pass

def step_simulation():
    pass

def joint_controller_hold():
    pass

def waypoints_from_path():
    pass

def link_from_name():
    pass

def create_attachment():
    pass

def add_fixed_constraint():
    pass

def joint_from_names():
    pass

def remove_fixed_constraint():
    pass

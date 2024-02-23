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
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from xarm_rl.robots.articulations import xarm
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

def get_initial_conf():
    pass

def get_group_conf(body_id, body_name):
    return 

def get_target_path():
    pass

def get_gripper_joints():
    pass

def get_joint_positions():
    pass

def get_link_pose():
    pass

def get_min_limit():
    pass

def get_max_limit():
    pass

def get_name():
    pass

def get_pose():
    pass

def get_extend_fn(robot, joints):
    def fn(start_conf: torch.Tensor, end_conf: torch.Tensor):
        pass
    return fn

def get_distance_fn():
    def fn(q1: torch.Tensor, q2: torch.Tensor):
        pass
    return fn


### Setter API

def set_joint_positions():
    pass

def set_pose():
    pass

def set_point(obj, position, rotation):
    obj.set_world_position(
        position=position,
        rotation=rotation,
    ) # TODO: Not implemented set_world_position API on RigidPrim or GeomPrim

def set_arm_conf(arm, conf):
    arm.set_joint_positions(conf)


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

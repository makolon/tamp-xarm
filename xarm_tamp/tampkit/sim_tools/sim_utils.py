import torch
import argparse

import omni
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
import copy
import math
import os
import time
import trimesh
import numpy as np
import omni.isaac.core.utils.bounds as bounds_utils
import omni.isaac.core.utils.mesh as mesh_utils
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.stage as stage_utils
from collections import namedtuple
from itertools import product
from typing import Dict, List, Tuple, Optional, Sequence, Union, Callable, Iterable
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from omni.isaac.core.utils.numpy.rotations import xyzw2wxyz, wxyz2xyzw
from omni.isaac.core.utils.torch.rotations import quat_diff_rad
from omni.isaac.core.utils.types import ArticulationAction

# Simulation API

def connect() -> 'SimulationApp':
    """
    Connect to the simulation application.

    Returns:
        SimulationApp: The simulation application.
    """
    global simulation_app
    return simulation_app

def disconnect() -> None:
    """
    Disconnect from the simulation application.
    """
    global simulation_app
    simulation_app.close()

def step_simulation(world: 'World') -> None:
    """
    Step the simulation forward.

    Args:
        world (World): The simulation world.
    """
    world.step(render=True)

def loop_simulation(world: 'World') -> None:
    """
    Continuously loop the simulation.

    Args:
        world (World): The simulation world.
    """
    sim_app = connect()
    while sim_app.is_running():
        world.step(render=True)
        print('Loop Simulation')

# Create Simulation Environment API

def create_world() -> 'World':
    """
    Create a new simulation world.

    Returns:
        World: The simulation world.
    """
    return World(stage_units_in_meters=1.0)

def create_floor(world: 'World', plane_cfg: 'PlaneConfig') -> 'GroundPlane':
    """
    Create a ground plane in the simulation world.

    Args:
        world (World): The simulation world.
        plane_cfg (PlaneConfig): Configuration for the ground plane.

    Returns:
        GroundPlane: The created ground plane.
    """
    return world.scene.add_default_ground_plane(
        static_friction=plane_cfg.static_friction,
        dynamic_friction=plane_cfg.dynamic_friction,
        restitution=plane_cfg.restitution,
    )

def create_block(block_name: str, translation: np.ndarray, orientation: np.ndarray) -> 'DynamicCuboid':
    """
    Create a dynamic block in the simulation world.

    Args:
        block_name (str): Name of the block.
        translation (np.ndarray): Translation of the block.
        orientation (np.ndarray): Orientation of the block.

    Returns:
        DynamicCuboid: The created block.
    """
    return cuboid.DynamicCuboid(
        prim_path=f"/World/{block_name}",
        name=block_name,
        translation=translation,
        orientation=orientation,
        color=np.array([1.0, 1.0, 1.0]),
        size=0.03,
    )

def create_surface(surface_name: str, translation: np.ndarray, orientation: np.ndarray) -> 'VisualCuboid':
    """
    Create a visual surface in the simulation world.

    Args:
        surface_name (str): Name of the surface.
        translation (np.ndarray): Translation of the surface.
        orientation (np.ndarray): Orientation of the surface.

    Returns:
        VisualCuboid: The created surface.
    """
    return cuboid.VisualCuboid(
        prim_path=f"/World/{surface_name}",
        name=surface_name,
        translation=translation,
        orientation=orientation,
        color=np.array([0.0, 0.0, 0.0]),
        size=0.01,
        visible=False,
    )

def create_hole(hole_name: str, translation: np.ndarray, orientation: np.ndarray) -> 'VisualCuboid':
    """
    Create a visual hole in the simulation world.

    Args:
        hole_name (str): Name of the hole.
        translation (np.ndarray): Translation of the hole.
        orientation (np.ndarray): Orientation of the hole.

    Returns:
        VisualCuboid: The created hole.
    """
    return cuboid.VisualCuboid(
        prim_path=f"/World/{hole_name}",
        name=hole_name,
        translation=translation,
        orientation=orientation,
        color=np.array([0.0, 0.0, 0.0]),
        size=0.01,
        visible=False,
    )

def create_table(table_cfg: 'TableConfig') -> 'FixedCuboid':
    """
    Create a fixed table in the simulation world.

    Args:
        table_cfg (TableConfig): Configuration for the table.

    Returns:
        FixedCuboid: The created table.
    """
    return cuboid.FixedCuboid(
        prim_path=f"/World/{table_cfg.table_name}",
        name=table_cfg.table_name,
        translation=np.array(table_cfg.translation),
        orientation=np.array(table_cfg.orientation),
        color=np.array(table_cfg.color),
        scale=np.array([table_cfg.width, table_cfg.depth, table_cfg.height]),
        size=table_cfg.size,
    )

def create_robot(robot_cfg: 'RobotConfig') -> 'Robot':
    """
    Create a robot in the simulation world.

    Args:
        robot_cfg (RobotConfig): Configuration for the robot.

    Returns:
        Robot: The created robot.

    Raises:
        ValueError: If the robot name is not recognized.
    """
    if "xarm" in robot_cfg.name:
        from xarm_tamp.tampkit.sim_tools.robots import xarm
        return xarm.xArm(
            prim_path="/World/xarm7",
            translation=np.array(robot_cfg.translation),
            orientation=np.array(robot_cfg.orientation),
        )
    else:
        raise ValueError("Unknown robot name")

def create_fmb(fmb_cfg: 'FmbConfig') -> 'Block':
    """
    Create a functional block module (FMB) in the simulation world.

    Args:
        fmb_cfg (FmbConfig): Configuration for the FMB.

    Returns:
        Block: The created FMB.
    """
    if fmb_cfg.task == 'momo':
        from xarm_tamp.tampkit.sim_tools.objects import fmb_momo
        return fmb_momo.Block(
            prim_path=f"/World/{fmb_cfg.name}",
            name=fmb_cfg.name,
            translation=np.array(fmb_cfg.translation),
            orientation=np.array(fmb_cfg.orientation),
            scale=np.array(fmb_cfg.scale),
        )
    elif fmb_cfg.task == 'simo':
        from xarm_tamp.tampkit.sim_tools.objects import fmb_simo
        return fmb_simo.Block(
            prim_path=f"/World/{fmb_cfg.name}",
            name=fmb_cfg.name,
            translation=np.array(fmb_cfg.translation),
            orientation=np.array(fmb_cfg.orientation),
            scale=np.array(fmb_cfg.scale),
        )
    return None

# Unit API

def unit_point() -> np.ndarray:
    """
    Get a unit point.

    Returns:
        np.ndarray: A unit point [0.0, 0.0, 0.0].
    """
    return np.array([0.0, 0.0, 0.0])

def unit_quat() -> np.ndarray:
    """
    Get a unit quaternion.

    Returns:
        np.ndarray: A unit quaternion [0.0, 0.0, 0.0, 1.0].
    """
    return np.array([0.0, 0.0, 0.0, 1.0])

def unit_pose() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a unit pose.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A unit pose.
    """
    return unit_point(), unit_quat()

def get_point(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']]) -> np.ndarray:
    """
    Get the position of a body.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to get the position of.

    Returns:
        np.ndarray: The position of the body.
    """
    return get_pose(body)[0]

def get_quat(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']]) -> np.ndarray:
    """
    Get the orientation of a body.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to get the orientation of.

    Returns:
        np.ndarray: The orientation of the body.
    """
    return get_pose(body)[1]

### Rigid Body API

def get_bodies(world: 'World', body_types: List[str] = ['all']) -> Optional[List['XFormPrim']]:
    """
    Get bodies from the simulation world.

    Args:
        world (World): The simulation world.
        body_types (List[str], optional): The types of bodies to get. Defaults to ['all'].

    Returns:
        Optional[List[XFormPrim]]: List of bodies.
    """
    all_objects = world.scene._scene_registry._all_object_dicts
    bodies = []

    if 'all' in body_types:
        bodies.extend(usd_obj for object_dict in all_objects for name, usd_obj in object_dict.items() if 'plane' not in name)
        return bodies

    if 'rigid' in body_types:
        bodies.extend(usd_obj for name, usd_obj in all_objects[0])
    if 'geom' in body_types:
        bodies.extend(usd_obj for name, usd_obj in all_objects[1] if 'plane' not in name)
    if 'robot' in body_types:
        bodies.extend(usd_obj for name, usd_obj in all_objects[3])
    if 'xform' in body_types:
        bodies.extend(usd_obj for name, usd_obj in all_objects[4])

    return bodies

def get_body_name(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']]) -> str:
    """
    Get the name of a body.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to get the name of.

    Returns:
        str: The name of the body.
    """
    return body.name

def get_pose(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the pose of a body.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to get the pose of.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The position and orientation of the body.
    """
    pos, rot = body.get_world_pose()
    return pos, wxyz2xyzw(rot)

def get_velocity(body: Optional['RigidPrim']) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the velocity of a rigid body.

    Args:
        body (Optional[RigidPrim]): The rigid body to get the velocity of.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The linear and angular velocity of the body.
    """
    return body.linear_velocity, body.angular_velocity

def get_transform_local(prim: 'Usd.Prim', 
                        time_code: Optional['Usd.TimeCode'] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the transform of the given prim relative to the parent prim.

    Args:
        prim (Usd.Prim): The prim to get the transform of.
        time_code (Optional[Usd.TimeCode], optional): The time code. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The local position and orientation of the prim.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    xform = UsdGeom.Xformable(prim)
    local_transformation = xform.GetLocalTransformation(time_code)
    local_pos = np.array(local_transformation.ExtractTranslation())
    _quat = local_transformation.ExtractRotationQuat()
    local_quat = np.array([_quat.GetReal()] + list(_quat.GetImaginary()))
    return local_pos, wxyz2xyzw(local_quat)

def get_transform_world(prim: 'Usd.Prim',
                        time_code: Optional['Usd.TimeCode'] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the transform of the given prim relative to the world frame.

    Args:
        prim (Usd.Prim): The prim to get the transform of.
        time_code (Optional[Usd.TimeCode], optional): The time code. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The world position and orientation of the prim.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    xform = UsdGeom.Xformable(prim)
    world_transformation = xform.ComputeLocalToWorldTransform(time_code)
    world_pos = np.array(world_transformation.ExtractTranslation())
    _quat = world_transformation.ExtractRotationQuat()
    world_quat = np.array([_quat.GetReal()] + list(_quat.GetImaginary()))
    return world_pos, wxyz2xyzw(world_quat)

def get_transform_relative(prim: 'Usd.Prim', 
                           other_prim: 'Usd.Prim', 
                           time_code: Optional['Usd.TimeCode'] = None
                          ) -> np.ndarray:
    """
    Get the transform of the given prim relative to the other prim.

    Args:
        prim (Usd.Prim): The prim to get the transform of.
        other_prim (Usd.Prim): The other prim to get the transform relative to.
        time_code (Optional[Usd.TimeCode], optional): The time code. Defaults to None.

    Returns:
        np.ndarray: The relative transform.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    world2prim = get_transform_world(prim, time_code)
    world2other = get_transform_world(other_prim, time_code)
    other2world = np.linalg.inv(world2other)
    return other2world @ world2prim

def set_pose(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
             pose: Tuple[np.ndarray, np.ndarray]
            ) -> None:
    """
    Set the pose of a body.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to set the pose of.
        pose (Tuple[np.ndarray, np.ndarray]): The pose to set. Tuple containing position (np.ndarray) and orientation (np.ndarray).
    """
    position, orientation = pose
    assert isinstance(body, (GeometryPrim, RigidPrim, XFormPrim)), "Invalid body type."
    body.set_world_pose(position=position, orientation=xyzw2wxyz(orientation))

def set_velocity(body: Optional['RigidPrim'], 
                 translation: np.ndarray = np.array([0.0, 0.0, 0.0]), 
                 rotation: np.ndarray = np.array([0.0, 0.0, 0.0])
                ) -> None:
    """
    Set the velocity of a rigid body.

    Args:
        body (Optional[RigidPrim]): The rigid body to set the velocity of.
        translation (np.ndarray, optional): The linear velocity to set. Defaults to [0.0, 0.0, 0.0].
        rotation (np.ndarray, optional): The angular velocity to set. Defaults to [0.0, 0.0, 0.0].
    """
    assert isinstance(body, RigidPrim), "Invalid body type."
    body.set_linear_velocity(velocity=translation)
    body.set_angular_velocity(velocity=rotation)

def set_transform_local(prim: 'Usd.Prim', 
                        transform: np.ndarray, 
                        time_code: Optional['Usd.TimeCode'] = None
                       ) -> None:
    """
    Set the transform for the given prim relative to the parent prim.

    Args:
        prim (Usd.Prim): The prim to set the transform for.
        transform (np.ndarray): The transformation matrix to set.
        time_code (Optional[Usd.TimeCode], optional): The time code. Defaults to None.
    """
    animation_mode = time_code is not None
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    if prim.GetAttribute("xformOp:transform").IsValid():
        prim.GetAttribute("xformOp:transform").Set(Gf.Matrix4d(transform.T), time_code)
    else:
        translation = transform[:3, 3].tolist()
        rot = Rotation.from_matrix(transform[:3, :3])
        rot_quat = rot.as_quat()
        rot_quat_real = rot_quat[-1]
        rot_quat_imag = Gf.Vec3f(rot_quat[:3].tolist())
        orientation = Gf.Quatf(rot_quat_real, rot_quat_imag)
        if prim.GetAttribute("xformOp:translate").IsValid():
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(translation), time_code)
        else:
            UsdGeom.Xformable(prim).AddTranslateOp()
            UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(translation), time_code)
        if prim.GetAttribute("xformOp:orient").IsValid():
            attribute = prim.GetAttribute("xformOp:orient")
            attribute.SetTypeName(Sdf.ValueTypeNames.Quatf)
            attribute.Set(orientation, time_code)
        else:
            orient_attr = UsdGeom.Xformable(prim).AddOrientOp()
            orient_attr.Set(orientation, time_code)
    if animation_mode:
        kinematics_enable_attr = prim.GetAttribute("physics:kinematicEnabled")
        if not kinematics_enable_attr.IsValid():
            kinematics_enable_attr = prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool)
        kinematics_enable_attr.Set(True)
        rigid_body_enable_attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if not rigid_body_enable_attr.IsValid():
            rigid_body_enable_attr = prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool)
        rigid_body_enable_attr.Set(False)

def set_transform_world(prim: 'Usd.Prim', 
                        transform: np.ndarray, 
                        time_code: Optional['Usd.TimeCode'] = None
                       ) -> None:
    """
    Set the transform for the given prim relative to the world frame.

    Args:
        prim (Usd.Prim): The prim to set the transform for.
        transform (np.ndarray): The transformation matrix to set.
        time_code (Optional[Usd.TimeCode], optional): The time code. Defaults to None.

    Raises:
        RuntimeError: If the parent of the prim is not a valid prim.
    """
    if not prim.GetParent().IsValid():
        raise RuntimeError(f"The given prim's ('{prim.GetPath()}') parent is not a valid prim, thus its world transform cannot be set.")
    world2prim = transform
    world2parent = get_transform_world(prim.GetParent(), time_code)
    parent2world = np.linalg.inv(world2parent)
    parent2prim = parent2world @ world2prim
    set_transform_local(prim, parent2prim, time_code)

def set_transform_relative(prim: 'Usd.Prim', 
                           other_prim: 'Usd.Prim', 
                           transform: np.ndarray, 
                           time_code: Optional['Usd.TimeCode'] = None
                          ) -> None:
    """
    Set the transform for the given prim relative to the other prim.

    Args:
        prim (Usd.Prim): The prim to set the transform for.
        other_prim (Usd.Prim): The other prim to set the transform relative to.
        transform (np.ndarray): The transformation matrix to set.
        time_code (Optional[Usd.TimeCode], optional): The time code. Defaults to None.

    Raises:
        RuntimeError: If the parent of the prim is not a valid prim.
    """
    if not prim.GetParent().IsValid():
        raise RuntimeError(f"The given prim's ('{prim.GetPath()}') parent is not a valid prim, thus its relative transform cannot be set.")
    other2prim = transform
    parent2other = get_transform_relative(other_prim, prim.GetParent(), time_code)
    parent2prim = parent2other @ other2prim
    set_transform_local(prim, parent2prim, time_code)

# Link Utils

def get_link(robot: 'Robot', name: str) -> Optional['Usd.Prim']:
    """Get the prim of the link according to the given name in the robot.

    Args:
        robot (Robot): The robot object.
        name (str): The name of the link.

    Returns:
        Optional[Usd.Prim]: The prim of the link if found, otherwise None.
    """
    for link_prim in robot.prim.GetChildren():
        if link_prim.GetName() == name:
            return link_prim
    return None

def get_tool_link(robot: 'Robot', tool_name: str) -> Optional['Usd.Prim']:
    """Get the tool link in the robot.

    Args:
        robot (Robot): The robot object.
        tool_name (str): The name of the tool link.

    Returns:
        Optional[Usd.Prim]: The prim of the tool link if found, otherwise None.
    """
    return get_link(robot, tool_name)

def get_all_links(robot: 'Robot') -> List['Usd.Prim']:
    """Get all links in the robot.

    Args:
        robot (Robot): The robot object.

    Returns:
        List[Usd.Prim]: List of all link prims in the robot.
    """
    return list(robot.prim.GetChildren())

def get_moving_links(robot: 'Robot') -> List['Usd.Prim']:
    """Get moving links in the robot.

    Args:
        robot (Robot): The robot object.

    Returns:
        List[Usd.Prim]: List of all moving link prims in the robot.
    """
    return get_all_links(robot)  # TODO: Add movable filter

def get_parent(prim: 'Usd.Prim') -> Optional['Usd.Prim']:
    """Get the parent of prim if it exists.

    Args:
        prim (Usd.Prim): The prim object.

    Returns:
        Optional[Usd.Prim]: The parent prim if it exists, otherwise None.
    """
    parent_prim = prim.GetParent()
    return parent_prim if parent_prim.IsValid() else None

def get_child(prim: 'Usd.Prim', child_name: str) -> Optional['Usd.Prim']:
    """Get the child of prim if it exists.

    Args:
        prim (Usd.Prim): The prim object.
        child_name (str): The name of the child.

    Returns:
        Optional[Usd.Prim]: The child prim if found, otherwise None.
    """
    for child_prim in prim.GetChildren():
        if child_prim.GetName() == child_name:
            return child_prim
    return None

def get_children(prim: 'Usd.Prim') -> List['Usd.Prim']:
    """Get the children of prim if it exists.

    Args:
        prim (Usd.Prim): The prim object.

    Returns:
        List[Usd.Prim]: List of child prims.
    """
    return list(prim.GetChildren())

def get_all_link_parents(robot: 'Robot') -> Dict[str, Optional['Usd.Prim']]:
    """Get all parent links in the robot.

    Args:
        robot (Robot): The robot object.

    Returns:
        Dict[str, Optional[Usd.Prim]]: Dictionary of link names and their parent prims.
    """
    return {link.GetName(): get_parent(link) for link in get_all_links(robot)}

def get_all_link_children(robot: 'Robot') -> Dict[str, List['Usd.Prim']]:
    """Get all children links in the robot.

    Args:
        robot (Robot): The robot object.

    Returns:
        Dict[str, List[Usd.Prim]]: Dictionary of link names and their child prims.
    """
    return {link.GetName(): get_children(link) for link in get_all_links(robot)}

def get_link_parents(robot: 'Robot', link: 'Usd.Prim') -> Optional['Usd.Prim']:
    """Get parent link of the specified link.

    Args:
        robot (Robot): The robot object.
        link (Usd.Prim): The link prim.

    Returns:
        Optional[Usd.Prim]: The parent link prim if found, otherwise None.
    """
    return get_all_link_parents(robot).get(link.GetName())

def get_link_children(robot: 'Robot', link: 'Usd.Prim') -> List['Usd.Prim']:
    """Get child links of the specified link.

    Args:
        robot (Robot): The robot object.
        link (Usd.Prim): The link prim.

    Returns:
        List[Usd.Prim]: List of child link prims.
    """
    return get_all_link_children(robot).get(link.GetName(), [])

def get_link_descendants(robot: 'Robot', link: 'Usd.Prim', test: Callable[['Usd.Prim'], bool] = lambda l: True) -> List['Usd.Prim']:
    """Get descendant links of the specified link.

    Args:
        robot (Robot): The robot object.
        link (Usd.Prim): The link prim.
        test (Callable[[Usd.Prim], bool], optional): A test function to filter links. Defaults to a function that returns True.

    Returns:
        List[Usd.Prim]: List of descendant link prims.
    """
    descendants = []
    for child in get_link_children(robot, link):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(robot, child, test))
    return descendants

def get_link_subtree(robot: 'Robot', link: 'Usd.Prim') -> List['Usd.Prim']:
    """Get subtree of the specified link.

    Args:
        robot (Robot): The robot object.
        link (Usd.Prim): The link prim.

    Returns:
        List[Usd.Prim]: List of link prims in the subtree.
    """
    return [link] + get_link_descendants(robot, link)

def get_link_pose(robot: 'Robot', link_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get the pose of the specified link.

    Args:
        robot (Robot): The robot object.
        link_name (str): The name of the link.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The local transform (position, orientation) of the link.

    Raises:
        ValueError: If the specified link does not exist.
    """
    for link_prim in robot.prim.GetChildren():
        if link_name == link_prim.GetName():
            return get_transform_world(link_prim)
    raise ValueError("Specified link does not exist.")

def get_tool_pose(robot: 'Robot') -> Tuple[np.ndarray, np.ndarray]:
    """Get the pose of the end effector.

    Args:
        robot (Robot): The robot object.mro

    Returns:
        Tuple[np.ndarray, np.ndarray]: The world transform (position, orientation)
    """
    end_effector = robot.end_effector
    return end_effector.get_world_pose()

# Joint Utils

def get_ancestor_prims(prim: 'Usd.Prim') -> Iterable['Usd.Prim']:
    """
    Get the ancestors of a prim.

    Args:
        prim (Usd.Prim): The prim to get the ancestors for.

    Yields:
        Usd.Prim: The ancestor prims.
    """
    parent_prim = prim.GetParent()
    if parent_prim.IsValid():
        yield parent_prim
        yield from get_ancestor_prims(parent_prim)

def get_links_for_joint(prim: 'Usd.Prim') -> Tuple[Optional['Usd.Prim'], Optional['Usd.Prim']]:
    """
    Get all link prims from the given joint prim.

    Args:
        prim (Usd.Prim): The joint prim.

    Returns:
        Tuple[Optional[Usd.Prim], Optional[Usd.Prim]]: The link prims associated with the joint.
    """
    stage = prim.GetStage()
    joint_api = UsdPhysics.Joint(prim)

    rel0_targets = joint_api.GetBody0Rel().GetTargets()
    if len(rel0_targets) > 1:
        raise NotImplementedError(
            "`get_links_for_joint` does not currently handle more than one relative"
            f" body target in the joint. joint_prim: {prim}, body0_rel_targets:"
            f" {rel0_targets}"
        )
    link0_prim = stage.GetPrimAtPath(rel0_targets[0]) if rel0_targets else None

    rel1_targets = joint_api.GetBody1Rel().GetTargets()
    if len(rel1_targets) > 1:
        raise NotImplementedError(
            "`get_links_for_joint` does not currently handle more than one relative"
            f" body target in the joint. joint_prim: {prim}, body1_rel_targets:"
            f" {rel1_targets}"
        )
    link1_prim = stage.GetPrimAtPath(rel1_targets[0]) if rel1_targets else None

    return link0_prim, link1_prim

def get_joints_for_articulated_root(prim: 'Usd.Prim', 
                                    joint_selector_func: Optional[Callable[['Usd.Prim'], bool]] = None
                                   ) -> List['Usd.Prim']:
    """
    Get all the child joint prims from the given articulated root prim.

    Args:
        prim (Usd.Prim): The root prim.
        joint_selector_func (Optional[Callable[[Usd.Prim], bool]], optional): Function to select joint prims.

    Returns:
        List[Usd.Prim]: List of joint prims.
    """
    if joint_selector_func is None:
        def joint_selector_func(p: 'Usd.Prim') -> bool:
            return p.IsA(UsdPhysics.Joint)
    stage = prim.GetStage()
    joint_prims = [
        joint for joint in filter(joint_selector_func, stage.Traverse())
        if any(link is not None and is_ancestor(prim, link) for link in get_links_for_joint(joint))
    ]
    return joint_prims

def get_joints(robot: 'Robot', group: str = 'arm') -> Tuple['Usd.Prim']:
    """
    Get joint prims from the given group.

    Args:
        robot (Robot): The robot object.
        group (str, optional): The group to get joints for. Defaults to 'arm'.

    Returns:
        Tuple[Usd.Prim]: Tuple of joint prims.
    """
    joint_prims = list(filter(is_a_movable_joint, get_joints_for_articulated_root(robot.prim)))
    if group == 'arm':
        return tuple(joint_prim for joint_prim in joint_prims if joint_prim.GetName() in robot._arm_dof_names)
    elif group == 'whole_body':
        arm_joint_prims = [joint_prim for joint_prim in joint_prims if joint_prim.GetName() in robot._arm_dof_names]
        gripper_joint_prims = [joint_prim for joint_prim in joint_prims if joint_prim.GetName() in robot._gripper_dof_names]
        return tuple(arm_joint_prims + gripper_joint_prims)
    return tuple()

def get_arm_joints(robot: 'Robot') -> Optional[Union[list, np.ndarray, torch.Tensor]]:
    """
    Get arm joint indices.

    Args:
        robot (Robot): The robot object.

    Returns:
        Optional[Union[list, np.ndarray, torch.Tensor]]: Arm joint indices.
    """
    return robot.arm_joints

def get_base_joints(robot: 'Robot') -> Optional[Union[list, np.ndarray, torch.Tensor]]:
    """
    Get base joint indices.

    Args:
        robot (Robot): The robot object.

    Returns:
        Optional[Union[list, np.ndarray, torch.Tensor]]: Base joint indices.
    """
    return robot.base_joints

def get_gripper_joints(robot: 'Robot') -> Optional[Union[list, np.ndarray, torch.Tensor]]:
    """
    Get gripper joint indices.

    Args:
        robot (Robot): The robot object.

    Returns:
        Optional[Union[list, np.ndarray, torch.Tensor]]: Gripper joint indices.
    """
    return robot.gripper_joints

def get_movable_joints(robot: 'Robot', use_gripper: bool = False) -> Optional[Union[list, np.ndarray, torch.Tensor]]:
    """
    Get movable joint indices.

    Args:
        robot (Robot): The robot object.
        use_gripper (bool, optional): Whether to include gripper joints. Defaults to False.

    Returns:
        Optional[Union[list, np.ndarray, torch.Tensor]]: Movable joint indices.
    """
    return robot.arm_joints + robot.gripper_joints if use_gripper else robot.arm_joints

def get_joint_positions(robot: 'Robot', 
                        joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None
                       ) -> np.ndarray:
    """
    Get joint positions.

    Args:
        robot (Robot): The robot object.
        joint_indices (Optional[Union[list, np.ndarray, torch.Tensor]], optional): Joint indices. Defaults to None.

    Returns:
        np.ndarray: Joint positions.
    """
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    return robot.get_joint_positions(joint_indices=joint_indices)

def get_joint_velocities(robot: 'Robot', 
                         joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None
                        ) -> np.ndarray:
    """
    Get joint velocities.

    Args:
        robot (Robot): The robot object.
        joint_indices (Optional[Union[list, np.ndarray, torch.Tensor]], optional): Joint indices. Defaults to None.

    Returns:
        np.ndarray: Joint velocities.
    """
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    return robot.get_joint_velocities(joint_indices=joint_indices)

def get_min_limit(robot: 'Robot', 
                  joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None
                 ) -> np.ndarray:
    """
    Get joint lower limit.

    Args:
        robot (Robot): The robot object.
        joint_indices (Optional[Union[np.ndarray, torch.Tensor]], optional): Joint indices. Defaults to None.

    Returns:
        np.ndarray: Joint lower limits.
    """
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    return robot.dof_properties["lower"][joint_indices]

def get_max_limit(robot: 'Robot', 
                  joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None
                 ) -> np.ndarray:
    """
    Get joint upper limit.

    Args:
        robot (Robot): The robot object.
        joint_indices (Optional[Union[np.ndarray, torch.Tensor]], optional): Joint indices. Defaults to None.

    Returns:
        np.ndarray: Joint upper limits.
    """
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    return robot.dof_properties["upper"][joint_indices]

def get_joint_limits(robot: 'Robot') -> Tuple[np.ndarray, np.ndarray]:
    """
    Get joint upper and lower limits.

    Args:
        robot (Robot): The robot object.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Joint upper and lower limits.
    """
    return get_min_limit(robot), get_max_limit(robot)

def get_custom_limits(robot: 'Robot', 
                      joint_names: Optional[Union[list, np.ndarray, torch.Tensor]],
                      custom_limits: dict = {}
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get custom limits.

    Args:
        robot (Robot): The robot object.
        joint_names (Optional[Union[list, np.ndarray, torch.Tensor]]): Joint names.
        custom_limits (dict, optional): Custom limits dictionary. Defaults to {}.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Custom joint limits.
    """
    joint_limits = [
        custom_limits[joint] if joint in custom_limits else get_joint_limits(robot)
        for joint in joint_names
    ]
    return zip(*joint_limits)

def get_initial_conf(robot: 'Robot', 
                     joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None
                    ) -> np.ndarray:
    """
    Get joint initial configuration.

    Args:
        robot (Robot): The robot object.
        joint_indices (Optional[Union[list, np.ndarray, torch.Tensor]], optional): Joint indices. Defaults to None.

    Returns:
        np.ndarray: Joint initial configuration.
    """
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    state = robot.get_joints_default_state()
    if state is None:
        initial_pos = get_joint_positions(robot, joint_indices)
        initial_vel = get_joint_velocities(robot, joint_indices)
        robot.set_joints_default_state(initial_pos, initial_vel)
    else:
        initial_pos = state.positions[joint_indices]
    return initial_pos

def get_group_conf(robot: 'Robot', group: str = 'arm') -> np.ndarray:
    """
    Get joint configuration corresponding to group.

    Args:
        robot (Robot): The robot object.
        group (str, optional): The group to get configuration for. Defaults to 'arm'.

    Returns:
        np.ndarray: Joint configuration.
    """
    if group == 'arm':
        joint_indices = robot.arm_joints
    elif group == 'gripper':
        joint_indices = robot.gripper_joints
    elif group == 'base':
        joint_indices = robot.base_joints
    elif group == 'whole_body':
        joint_indices = get_movable_joints(robot, use_gripper=True)
    return robot.get_joint_positions(joint_indices=joint_indices)

def set_joint_positions(robot: 'Robot', 
                        positions: Optional[Union[np.ndarray, torch.Tensor]],
                        joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None
                       ) -> None:
    """
    Set joint positions.

    Args:
        robot (Robot): The robot object.
        positions (Optional[Union[np.ndarray, torch.Tensor]]): Joint positions.
        joint_indices (Optional[Union[np.ndarray, torch.Tensor]], optional): Joint indices. Defaults to None.
    """
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    robot.set_joint_positions(positions, joint_indices)

def set_initial_conf(robot: 'Robot', 
                     initial_conf: Optional[Union[np.ndarray, torch.Tensor]], 
                     joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None
                    ) -> None:
    """
    Set joint positions to initial configuration.

    Args:
        robot (Robot): The robot object.
        initial_conf (Optional[Union[np.ndarray, torch.Tensor]]): Initial configuration.
        joint_indices (Optional[Union[np.ndarray, torch.Tensor]], optional): Joint indices. Defaults to None.
    """
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    if initial_conf is None:
        initial_conf = get_joint_positions(robot, joint_indices)
    robot.set_joint_positions(initial_conf, joint_indices=joint_indices)

def apply_action(robot: 'Robot', 
                 configuration: Optional['ArticulationAction'] = None, 
                 joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None
                ) -> None:
    """
    Apply articulation action.

    Args:
        robot (Robot): The robot object.
        configuration (Optional[ArticulationAction], optional): Articulation action configuration. Defaults to None.
        joint_indices (Optional[Union[list, np.ndarray, torch.Tensor]], optional): Joint indices. Defaults to None.
    """
    art_controller = robot.get_articulation_controller()
    art_controller.apply_action(configuration)

def is_ancestor(prim: 'Usd.Prim', other_prim: 'Usd.Prim') -> bool:
    """
    Check if `prim` is an ancestor prim of `other_prim`.

    Args:
        prim (Usd.Prim): The potential ancestor prim.
        other_prim (Usd.Prim): The other prim.

    Returns:
        bool: True if `prim` is an ancestor of `other_prim`, False otherwise.
    """
    return prim in set(get_ancestor_prims(other_prim))

def is_a_movable_joint(prim: 'Usd.Prim') -> bool:
    """
    Check if the given prim is a movable joint prim.

    Args:
        prim (Usd.Prim): The prim to check.

    Returns:
        bool: True if the prim is a movable joint, False otherwise.
    """
    supported_joint_types = [UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint]
    return any(map(prim.IsA, supported_joint_types))

def is_circular(robot: 'Robot', joint: 'Usd.Prim') -> bool:
    """
    Check if the specified joint is circular.

    Args:
        robot (Robot): The robot object.
        joint (Usd.Prim): The joint to check.

    Returns:
        bool: True if the joint is circular, False otherwise.
    """
    if joint.IsA(UsdPhysics.FixedJoint):
        return False
    try:
        joint_index = robot.get_dof_index(joint.GetName())
    except ValueError as e:
        print(e)
    upper, lower = robot.dof_properties['upper'][joint_index], robot.dof_properties['lower'][joint_index]
    return upper < lower

def get_difference_fn(robot: 'Robot', joints: List['Usd.Prim']) -> Callable[[Union[list, np.ndarray], Union[list, np.ndarray]], Tuple]:
    """
    Get the difference function between joint configurations.

    Args:
        robot (Robot): The robot object.
        joints (List[Usd.Prim]): List of joint prims.

    Returns:
        Callable[[Union[list, np.ndarray], Union[list, np.ndarray]], Tuple]: Difference function.
    """
    circular_joints = [is_circular(robot, joint) for joint in joints]
    def fn(q2: Union[list, np.ndarray], q1: Union[list, np.ndarray]) -> Tuple:
        return tuple(circular_difference(v2, v1) if circular else (v2 - v1)
                     for circular, v2, v1 in zip(circular_joints, q2, q1))
    return fn

def get_refine_fn(robot: 'Robot', 
                  joints: List['Usd.Prim'], 
                  num_steps: int = 0
                 ) -> Callable[[Union[list, np.ndarray], Union[list, np.ndarray]], Iterable[Tuple]]:
    """
    Refine the given joint configuration.

    Args:
        robot (Robot): The robot object.
        joints (List[Usd.Prim]): List of joint prims.
        num_steps (int, optional): Number of steps for refinement. Defaults to 0.

    Returns:
        Callable[[Union[list, np.ndarray], Union[list, np.ndarray]], Iterable[Tuple]]: Refinement function.
    """
    difference_fn = get_difference_fn(robot, joints)
    num_steps += 1
    def fn(q1: Union[list, np.ndarray], q2: Union[list, np.ndarray]) -> Iterable[Tuple]:
        q = q1
        for i in range(num_steps):
            positions = (1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            q = tuple(positions)
            yield q
    return fn

def get_extend_fn(robot: 'Robot', 
                  joints: List['Usd.Prim'], 
                  norm: int = 2
                 ) -> Callable[[Union[list, np.ndarray], Union[list, np.ndarray]], Iterable[Tuple]]:
    """
    Extend the given joint configuration.

    Args:
        robot (Robot): The robot object.
        joints (List[Usd.Prim]): List of joint prims.
        norm (int, optional): Norm for calculating steps. Defaults to 2.

    Returns:
        Callable[[Union[list, np.ndarray], Union[list, np.ndarray]], Iterable[Tuple]]: Extension function.
    """
    resolutions = math.radians(3) * np.ones(len(joints))
    difference_fn = get_difference_fn(robot, joints)
    def fn(q1: Union[list, np.ndarray], q2: Union[list, np.ndarray]) -> Iterable[Tuple]:
        steps = int(np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=norm))
        refine_fn = get_refine_fn(robot, joints, num_steps=steps)
        return refine_fn(q1, q2)
    return fn

def get_distance_fn(robot: 'Robot', 
                    joints: List['Usd.Prim']
                   ) -> Callable[[Union[list, np.ndarray], Union[list, np.ndarray]], float]:
    """
    Get the distance function between two configurations.

    Args:
        robot (Robot): The robot object.
        joints (List[Usd.Prim]): List of joint prims.

    Returns:
        Callable[[Union[list, np.ndarray], Union[list, np.ndarray]], float]: Distance function.
    """
    weights = np.ones(len(joints))
    difference_fn = get_difference_fn(robot, joints)
    def fn(q1: Union[list, np.ndarray], q2: Union[list, np.ndarray]) -> float:
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
    return fn

def refine_path(robot: 'Robot', 
                joints: List['Usd.Prim'], 
                waypoints: Sequence, 
                num_steps: int = 0
               ) -> List[Tuple]:
    """
    Refine the given path.

    Args:
        robot (Robot): The robot object.
        joints (List[Usd.Prim]): List of joint prims.
        waypoints (Sequence): Sequence of waypoints.
        num_steps (int, optional): Number of steps for refinement. Defaults to 0.

    Returns:
        List[Tuple]: Refined path.
    """
    refine_fn = get_refine_fn(robot, joints, num_steps)
    refined_path = []
    for v1, v2 in get_pairs(waypoints):
        refined_path.extend(refine_fn(v1, v2))
    return refined_path

def get_pose_distance(pose1: Union[list, np.ndarray, torch.Tensor], 
                      pose2: Union[list, np.ndarray, torch.Tensor]
                     ) -> Tuple[float, float]:
    """
    Get the distance between two poses.

    Args:
        pose1 (Union[list, np.ndarray, torch.Tensor]): The first pose.
        pose2 (Union[list, np.ndarray, torch.Tensor]): The second pose.

    Returns:
        Tuple[float, float]: Position and orientation distances.
    """
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    pos_distance = get_distance(pos1, pos2)
    ori_distance = quat_diff_rad(quat1, quat2)
    return pos_distance, ori_distance

def interpolate_poses(pose1: Union[list, np.ndarray, torch.Tensor], 
                      pose2: Union[list, np.ndarray, torch.Tensor], 
                      pos_step_size: float = 0.01, 
                      ori_step_size: float = np.pi/16
                     ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Interpolate between two poses.

    Args:
        pose1 (Union[list, np.ndarray, torch.Tensor]): The first pose.
        pose2 (Union[list, np.ndarray, torch.Tensor]): The second pose.
        pos_step_size (float, optional): Step size for position. Defaults to 0.01.
        ori_step_size (float, optional): Step size for orientation. Defaults to π/16.

    Yields:
        Tuple[np.ndarray, np.ndarray]: Interpolated poses.
    """
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    num_steps = max(2, int(math.ceil(max(
        np.divide(get_pose_distance(pose1, pose2), [pos_step_size, ori_step_size])))))
    yield pose1
    for w in np.linspace(0, 1, num=num_steps, endpoint=True)[1:-1]:
        pos = convex_combination(pos1, pos2, w=w)
        quat = quat_combination(quat1, quat2, fraction=w)
        yield pos, quat
    yield pose2

def iterate_approach_path(robot: 'Robot', 
                          gripper: 'Usd.Prim', 
                          pose: Union[list, np.ndarray, torch.Tensor], 
                          grasp: Union[list, np.ndarray, torch.Tensor], 
                          body: Optional[Union[list, np.ndarray, torch.Tensor]] = None
                         ) -> Iterable[None]:
    """
    Interpolate approach path.

    Args:
        robot (Robot): The robot object.
        gripper (Usd.Prim): The gripper prim.
        pose (Union[list, np.ndarray, torch.Tensor]): The target pose.
        grasp (Union[list, np.ndarray, torch.Tensor]): The grasp pose.
        body (Optional[Union[list, np.ndarray, torch.Tensor]], optional): The body pose. Defaults to None.

    Yields:
        None: Interpolated approach path.
    """
    tool_from_root = get_tool_link(robot)
    grasp_pose = multiply(pose.value, invert(grasp.value))
    approach_pose = multiply(pose.value, invert(grasp.approach))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(gripper, multiply(tool_pose, tool_from_root))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.value))
        yield

# Collision Geometry API

def aabb_empty(aabb: Tuple[np.ndarray, np.ndarray]) -> bool:
    """
    Check if an axis-aligned bounding box (AABB) is empty.

    Args:
        aabb (Tuple[np.ndarray, np.ndarray]): A tuple containing the lower and upper bounds of the AABB.

    Returns:
        bool: True if the AABB is empty, False otherwise.
    """
    lower, upper = aabb
    return np.less(upper, lower).any()

def aabb2d_from_aabb(aabb: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the 2D projection of an AABB.

    Args:
        aabb (Tuple[np.ndarray, np.ndarray]): A tuple containing the lower and upper bounds of the AABB.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the lower and upper bounds of the 2D AABB.
    """
    lower, upper = aabb
    return lower[:2], upper[:2]

def sample_aabb(aabb: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Sample a point uniformly within an AABB.

    Args:
        aabb (Tuple[np.ndarray, np.ndarray]): A tuple containing the lower and upper bounds of the AABB.

    Returns:
        np.ndarray: A point sampled within the AABB.
    """
    lower, upper = aabb
    return np.random.uniform(lower, upper)

def get_aabb(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the axis-aligned bounding box (AABB) of a body.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to get the AABB for.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the lower and upper bounds of the AABB.
    """
    cache = bounds_utils.create_bbox_cache()
    body_aabb = bounds_utils.compute_aabb(cache, body.prim_path)
    lower, upper = body_aabb[:3], body_aabb[3:]
    return lower, upper

def get_center_extent(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the center and extent of a body.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to get the center and extent for.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the center and extent of the body.
    """
    lower, upper = get_aabb(body)
    diff = np.array(upper) - np.array(lower)
    center = (np.array(upper) + np.array(lower)) / 2.
    return center, diff

def aabb_contains_aabb(contained: Tuple[np.ndarray, np.ndarray], container: Tuple[np.ndarray, np.ndarray]) -> bool:
    """
    Check if one AABB is contained within another AABB.

    Args:
        contained (Tuple[np.ndarray, np.ndarray]): The contained AABB.
        container (Tuple[np.ndarray, np.ndarray]): The container AABB.

    Returns:
        bool: True if the contained AABB is within the container AABB, False otherwise.
    """
    lower1, upper1 = contained
    lower2, upper2 = container
    return np.less_equal(lower2, lower1).all() and np.less_equal(upper1, upper2).all()

def is_placed_on_aabb(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                      bottom_aabb: Tuple[np.ndarray, np.ndarray],
                      above_epsilon: float = 0.05,
                      below_epsilon: float = 0.05) -> bool:
    """
    Check if a body is placed on top of an AABB.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to check.
        bottom_aabb (Tuple[np.ndarray, np.ndarray]): The AABB to check against.
        above_epsilon (float, optional): The tolerance above the AABB.
        below_epsilon (float, optional): The tolerance below the AABB.

    Returns:
        bool: True if the body is placed on the AABB, False otherwise.
    """
    assert 0 <= above_epsilon <= below_epsilon
    top_aabb = get_aabb(body)
    top_z_min = top_aabb[0][2]
    bottom_z_max = bottom_aabb[1][2]
    return ((bottom_z_max - below_epsilon) <= top_z_min <= (bottom_z_max + above_epsilon)) and \
           aabb_contains_aabb(aabb2d_from_aabb(top_aabb), aabb2d_from_aabb(bottom_aabb))

def is_placement(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                 surface: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                ) -> bool:
    """
    Check if a body is placed on a surface.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to check.
        surface (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The surface to check against.

    Returns:
        bool: True if the body is placed on the surface, False otherwise.
    """
    if get_aabb(surface) is None:
        return False
    return is_placed_on_aabb(body, get_aabb(surface))

def is_insertion(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                 hole: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                ) -> bool:
    """
    Check if a body is inserted into a hole.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to check.
        hole (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The hole to check against.

    Returns:
        bool: True if the body is inserted into the hole, False otherwise.
    """
    if get_aabb(hole) is None:
        return False
    return is_placed_on_aabb(body, get_aabb(hole))

def check_geometry_type(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']]):
    # Is it possible to alternate this function using `get_prim_type_name`?
    if body.prim.IsA(UsdGeom.Capsule):
        return 'capsule'
    elif body.prim.IsA(UsdGeom.Cone):
        return 'cone'
    elif body.prim.IsA(UsdGeom.Cube):
        return 'cude'
    elif body.prim.IsA(UsdGeom.Cylinder):
        return 'cylinder'
    elif body.prim.IsA(UsdGeom.Sphere):
        return 'sphere'
    elif body.prim.IsA(UsdGeom.Plane):
        return 'plane'
    elif body.prim.IsA(UsdGeom.Mesh):
        return 'mesh'
    else:
        raise ValueError('prim geometry does not exist.')

def approximate_as_prism(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                         body_pose: np.ndarray = None,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate a rigid body as a prism.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to approximate.
        body_pose (np.ndarray, optional): The pose of the body.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the center and extent of the approximated prism.
    """
    if body_pose is None:
        position, rotation = get_pose(body)
        transform = tf_matrices_from_poses(position, rotation)
    else:
        position, rotation = body_pose
        transform = tf_matrices_from_poses(position, rotation)

    body_type = check_geometry_type(body)
    if body_type == 'mesh':
        mesh = get_body_geometry(body)
        mesh.apply_transform(transform)

        lower, upper = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
        diff, center = np.array(upper) - np.array(lower), (np.array(upper) + np.array(lower)) / 2.
    else:
        center, diff = get_center_extent(body)
    return center, diff

def get_prim_geometry(prim: 'Usd.Prim') -> Dict[str, np.ndarray]:
    """
    Return the geometry elements (vertices and faces) for the given prim.

    Args:
        prim (Usd.Prim): The prim to get the geometry for.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the vertices and faces of the prim.
    """
    if not prim.IsA(UsdGeom.Mesh):
        usd_geom = UsdGeom.Mesh(prim)
    else:
        usd_geom = prim
    faces = usd_geom.GetFaceVertexIndicesAttr().Get()
    faces = np.array(faces, dtype=np.int32).reshape(-1, 3)
    vertices = usd_geom.GetPointsAttr().Get()
    vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    return dict(vertices=vertices, faces=faces)

def get_body_geometry(body: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']], 
                      scale: bool = True) -> trimesh.Trimesh:
    """
    Return the geometry elements (vertices and faces) for the given body.

    Args:
        body (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The body to get the geometry for.
        scale (bool, optional): Whether to scale the geometry.

    Returns:
        trimesh.Trimesh: A trimesh object containing the geometry.
    """
    stage = omni.usd.get_context().get_stage()
    prim_path = body.prim_path + f'/{body.name}/collisions/mesh_0'  # TODO: fix
    prim = stage.GetPrimAtPath(prim_path)
    usd_geom = UsdGeom.Mesh(prim)
    faces = usd_geom.GetFaceVertexIndicesAttr().Get()
    faces = np.array(faces, dtype=np.int32).reshape(-1, 3)
    vertices = usd_geom.GetPointsAttr().Get()
    vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if scale:
        mesh.apply_scale([0.001, 0.001, 0.001])
    return mesh

def get_bounds(root_prim: 'Usd.Prim', time_code: Optional['Usd.TimeCode'] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the world axis-aligned bounds of geometry rooted at a prim.

    Args:
        root_prim (Usd.Prim): The root prim to get the bounds for.
        time_code (Optional[Usd.TimeCode], optional): The time code for which to get the bounds.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the lower and upper bounds of the geometry.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
    bbox_cache.Clear()
    prim_bbox = bbox_cache.ComputeWorldBound(root_prim)
    prim_range = prim_bbox.ComputeAlignedRange()
    lower = np.array(prim_range.GetMin())
    upper = np.array(prim_range.GetMax())
    return lower, upper

def get_closest_points(body1: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                       body2: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                       transform1: Optional[np.ndarray] = None,
                       transform2: Optional[np.ndarray] = None,
                       max_distance: float = 0.) -> Optional[float]:
    """
    Calculate the closest points or collision detection between two mesh objects.

    Args:
        body1 (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The first body.
        body2 (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The second body.
        transform1 (Optional[np.ndarray], optional): The transformation matrix for the first body.
        transform2 (Optional[np.ndarray], optional): The transformation matrix for the second body.
        max_distance (float, optional): The maximum distance to check for proximity.

    Returns:
        Optional[float]: The minimum distance between the two meshes, or None if no points are within max_distance.
    """
    if transform1 is None:
        position, rotation = get_pose(body1)
        transform1 = tf_matrices_from_poses(position, rotation)
    if transform2 is None:
        position, rotation = get_pose(body2)
        transform2 = tf_matrices_from_poses(position, rotation)

    body1_type = check_geometry_type(body1)
    body2_type = check_geometry_type(body2)

    # TODO: need to add sdf culculation function for cubes and spheres.

    if body1_type == 'mesh' and body2_type == 'mesh':
        mesh1 = get_body_geometry(body1)
        mesh2 = get_body_geometry(body2)

        mesh1.apply_transform(transform1)
        mesh2.apply_transform(transform2)

    distance = trimesh.proximity.signed_distance(mesh1, mesh2.vertices)
    if max_distance > 0:
        close_distances = distance[np.abs(distance) <= max_distance]
        if len(close_distances) > 0:
            return np.min(np.abs(close_distances))
        else:
            return None
    return np.min(np.abs(distance))

def pairwise_link_collision(body1: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                            body2: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                           ) -> bool:
    """
    Check for a collision between two links of different bodies.

    Args:
        body1 (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The first body.
        body2 (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The second body.

    Returns:
        bool: True if there is a collision, False otherwise.
    """
    return len(get_closest_points(body1, body2)) != 0

def any_link_pair_collision(body1: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                            links1: List[str],
                            body2: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                            links2: Optional[List[str]] = None,
                           ) -> bool:
    """
    Check for a collision between any pair of links from two bodies.

    Args:
        body1 (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The first body.
        links1 (List[str]): The list of links for the first body.
        body2 (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The second body.
        links2 (Optional[List[str]], optional): The list of links for the second body.

    Returns:
        bool: True if there is a collision, False otherwise.
    """
    if links1 is None:
        links1 = get_all_links(body1)
    if links2 is None:
        links2 = get_all_links(body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(link1, link2):
            return True
    return False

def body_collision(body1: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                   body2: Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']],
                  ) -> bool:
    """
    Check for a collision between two bodies.

    Args:
        body1 (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The first body.
        body2 (Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]): The second body.

    Returns:
        bool: True if there is a collision, False otherwise.
    """
    return len(get_closest_points(body1, body2)) != 0

def pairwise_collision(body1: Union[Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']], Tuple],
                       body2: Union[Optional[Union['GeometryPrim', 'RigidPrim', 'XFormPrim']], Tuple],
                      ) -> bool:
    """
    Check for a collision between two bodies or links of bodies.

    Args:
        body1 (Union[Optional[Union[GeometryPrim, RigidPrim, XFormPrim]], Tuple]): The first body or tuple of body and links.
        body2 (Union[Optional[Union[GeometryPrim, RigidPrim, XFormPrim]], Tuple]): The second body or tuple of body and links.

    Returns:
        bool: True if there is a collision, False otherwise.
    """
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        return any_link_pair_collision(body1, links1, body2, links2)
    return body_collision(body1, body2)

def all_between(lower_limits: np.ndarray, values: np.ndarray, upper_limits: np.ndarray) -> bool:
    """
    Check if all values are between their respective lower and upper limits.

    Args:
        lower_limits (np.ndarray): The lower limits.
        values (np.ndarray): The values to check.
        upper_limits (np.ndarray): The upper limits.

    Returns:
        bool: True if all values are within the limits, False otherwise.
    """
    assert len(lower_limits) == len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and np.less_equal(values, upper_limits).all()

def parse_body(robot: 'Robot', link: Optional[str] = None) -> Tuple['Robot', Union[None, str]]:
    """
    Parse a robot body and link.

    Args:
        robot (Robot): The robot object.
        link (Optional[str], optional): The link to parse.

    Returns:
        Tuple[Robot, Union[None, str]]: A tuple containing the robot and links.
    """
    collision_pair = namedtuple('Collision', ['robot', 'links'])
    return robot if isinstance(robot, tuple) else collision_pair(robot, link)

def flatten_links(robot: 'Robot', links: Optional[List[str]] = None) -> set:
    """
    Flatten a robot's links into a set of individual links.

    Args:
        robot (Robot): The robot object.
        links (Optional[List[str]], optional): The list of links.

    Returns:
        set: A set of individual links.
    """
    if links is None:
        links = [link_prim.GetName() for link_prim in robot.prim.GetChildren()]
    collision_pair = namedtuple('Collision', ['robot', 'links'])
    return {collision_pair(robot, frozenset([link])) for link in links}

def expand_links(robot: 'Robot') -> Tuple['Robot', Union[None, List[str]]]:
    """
    Expand a robot's links.

    Args:
        robot (Robot): The robot object.

    Returns:
        Tuple[Robot, Union[None, List[str]]]: A tuple containing the robot and expanded links.
    """
    body, links = parse_body(robot)
    if links is None:
        links = get_all_links(body)
    collision_pair = namedtuple('Collision', ['robot', 'links'])
    return collision_pair(body, links)

# Mathematical Utilities

def wrap_interval(value: float, interval: Tuple[float, float] = (0., 1.)) -> float:
    """
    Wrap a given value into a specified interval.

    Args:
        value (float): The value to wrap.
        interval (Tuple[float, float], optional): The interval to wrap the value into.

    Returns:
        float: The wrapped value.
    """
    lower, upper = interval
    assert lower <= upper
    return (value - lower) % (upper - lower) + lower

def circular_difference(theta2: float, theta1: float) -> float:
    """
    Calculate the circular difference between two angles, wrapped within the interval [-π, π).

    Args:
        theta2 (float): The second angle.
        theta1 (float): The first angle.

    Returns:
        float: The circular difference between the two angles.
    """
    diff_theta = theta2 - theta1
    interval = (-np.pi, np.pi)
    return wrap_interval(diff_theta, interval=interval)

def flatten(iterable_of_iterables: List[List]) -> List:
    """
    Flatten a nested iterable of iterables into a single list yielding all inner elements.

    Args:
        iterable_of_iterables (List[List]): The nested iterable to flatten.

    Returns:
        List: A list yielding all inner elements.
    """
    return [item for sublist in iterable_of_iterables for item in sublist]

def convex_combination(x: Union[np.ndarray, List[float]], 
                       y: Union[np.ndarray, List[float]], 
                       w: float = 0.5) -> np.ndarray:
    """
    Compute the convex combination of two vectors.

    Args:
        x (Union[np.ndarray, List[float]]): The first vector.
        y (Union[np.ndarray, List[float]]): The second vector.
        w (float, optional): The weight of the second vector in the combination.

    Returns:
        np.ndarray: The convex combination of the two vectors.
    """
    return (1 - w) * np.array(x) + w * np.array(y)

def unit_vector(data: np.ndarray, axis: Optional[int] = None, out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return ndarray normalized by length, i.e., Euclidean norm, along axis.

    Args:
        data (np.ndarray): The data to normalize.
        axis (Optional[int], optional): The axis along which to normalize.
        out (Optional[np.ndarray], optional): The output array.

    Returns:
        np.ndarray: The normalized array.
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= np.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def quaternion_slerp(quat0: np.ndarray, quat1: np.ndarray, fraction: float, spin: int = 0, shortestpath: bool = True) -> np.ndarray:
    """
    Return spherical linear interpolation between two quaternions.

    Args:
        quat0 (np.ndarray): The first quaternion.
        quat1 (np.ndarray): The second quaternion.
        fraction (float): The interpolation fraction.
        spin (int, optional): The number of extra spins.
        shortestpath (bool, optional): Whether to use the shortest path.

    Returns:
        np.ndarray: The interpolated quaternion.
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < np.finfo(float).eps * 4.0:
        return q0
    if shortestpath and d < 0.0:
        d = -d
        q1 *= -1.0
    angle = np.arccos(d) + spin * np.pi
    if abs(angle) < np.finfo(float).eps * 4.0:
        return q0
    isin = 1.0 / np.sin(angle)
    q0 *= np.sin((1.0 - fraction) * angle) * isin
    q1 *= np.sin(fraction * angle) * isin
    q0 += q1
    return q0

def quat_combination(quat1: np.ndarray, quat2: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    """
    Compute the combination of two quaternions using spherical linear interpolation.

    Args:
        quat1 (np.ndarray): The first quaternion.
        quat2 (np.ndarray): The second quaternion.
        fraction (float, optional): The interpolation fraction.

    Returns:
        np.ndarray: The combined quaternion.
    """
    return quaternion_slerp(quat1, quat2, fraction)

def get_pairs(sequence: List) -> zip:
    """
    Get pairs of consecutive elements in a sequence.

    Args:
        sequence (List): The sequence to get pairs from.

    Returns:
        zip: A zip object containing pairs of consecutive elements.
    """
    sequence = list(sequence)
    return zip(sequence[:-1], sequence[1:])

def get_distance(p1: Union[List[float], np.ndarray], 
                 p2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (Union[List[float], np.ndarray]): The first point.
        p2 (Union[List[float], np.ndarray]): The second point.

    Returns:
        float: The Euclidean distance between the points.
    """
    assert len(p1) == len(p2)
    diff = np.array(p2) - np.array(p1)
    return np.linalg.norm(diff, ord=2)

def multiply(pose1: Union[List, np.ndarray], 
             pose2: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multiply two poses to get the resulting transformation.

    Args:
        pose1 (Union[List, np.ndarray]): The first pose.
        pose2 (Union[List, np.ndarray]): The second pose.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the position and rotation of the resulting transformation.
    """
    transform1 = np.eye(4)
    if len(pose1[1]) == 4:
        transform1[:3, :3] = Rotation.from_quat(pose1[1]).as_matrix()
    elif len(pose1[1]) == 3:
        transform1[:3, :3] = Rotation.from_euler('xyz', pose1[1]).as_matrix()
    transform1[:3, 3] = pose1[0]

    transform2 = np.eye(4)
    if len(pose2[1]) == 4:
        transform2[:3, :3] = Rotation.from_quat(pose2[1]).as_matrix()
    elif len(pose2[1]) == 3:
        transform2[:3, :3] = Rotation.from_euler('xyz', pose2[1]).as_matrix()
    transform2[:3, 3] = pose2[0]

    result = transform1 @ transform2
    result_pos = result[:3, 3]
    result_rot = Rotation.from_matrix(result[:3, :3]).as_quat()
    return result_pos, result_rot

def multiply_array(*poses: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multiply an array of poses to get the resulting transformation.

    Args:
        *poses (Union[List, np.ndarray]): The array of poses.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the position and rotation of the resulting transformation.
    """
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = multiply(pose, next_pose)
    return pose

def invert(pose: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the inverse of a transformation.

    Args:
        pose (Union[List, np.ndarray]): The pose to invert.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the position and rotation of the inverse transformation.
    """
    transform = np.eye(4)
    if len(pose[1]) == 4:
        transform[:3, :3] = Rotation.from_quat(pose[1]).as_matrix()
    elif len(pose[1]) == 3:
        transform[:3, :3] = Rotation.from_euler('xyz', pose[1]).as_matrix()
    transform[:3, 3] = pose[0]

    result = np.linalg.inv(transform)
    result_pos = result[:3, 3]
    result_rot = Rotation.from_matrix(result[:3, :3]).as_quat()
    return result_pos, result_rot

def transform_point(affine: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Transform a point using an affine transformation.

    Args:
        affine (np.ndarray): The affine transformation matrix.
        point (np.ndarray): The point to transform.

    Returns:
        np.ndarray: The transformed point.
    """
    return multiply(affine, [point, [0, 0, 0, 1]])[0]

def apply_affine(affine: np.ndarray, points: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply an affine transformation to a list of points.

    Args:
        affine (np.ndarray): The affine transformation matrix.
        points (List[np.ndarray]): The list of points.

    Returns:
        List[np.ndarray]: A list of transformed points.
    """
    return [transform_point(affine, p) for p in points]

def tf_matrices_from_poses(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 transformation matrix from a position vector and a quaternion.

    Args:
        position (np.ndarray): Position vector [x, y, z]
        quaternion (np.ndarray): Quaternion [w, x, y, z]

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    rotation = Rotation.from_quat(quaternion).as_matrix()
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = position
    T[3, 3] = 1
    return T
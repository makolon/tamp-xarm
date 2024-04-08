import torch
import argparse
from collections import namedtuple

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
import os
import carb
import copy
import math
import time
import numpy as np
import omni.isaac.core.utils.bounds as bounds_utils
import omni.isaac.core.utils.mesh as mesh_utils
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.stage as stage_utils
from itertools import count, product
from typing import Dict, List, Tuple, Optional, Sequence, Union
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from omni.isaac.core.utils.torch.rotations import quat_diff_rad, xyzw2wxyz, wxyz2xyzw
from omni.isaac.core.utils.types import ArticulationAction


### Simulation API

def connect():
    global simulation_app
    return simulation_app

def disconnect():
    global simulation_app
    simulation_app.close()
    
def step_simulation(world):
    world.step(render=True)

def loop_simulation(world):
    sim_app = connect()
    while sim_app.is_running():
        world.step(render=True)
        print('Loop Simulation')

### Create Simulation Environment API

def create_world():
    my_world = World(stage_units_in_meters=1.0)
    return my_world

def create_floor(world, plane_cfg):
    plane = world.scene.add_default_ground_plane(
        static_friction=plane_cfg.static_friction,
        dynamic_friction=plane_cfg.dynamic_friction,
        restitution=plane_cfg.restitution,
    )
    return plane

def create_block(block_name, translation, orientation):
    block = cuboid.DynamicCuboid(
        f"/World/{block_name}",
        f"{block_name}",
        translation=translation,
        orientation=orientation,
        color=np.array([0., 0., 0.]),
        size=0.01,
    )
    return block

def create_surface(surface_name, translation, orientation):
    surface = cuboid.VisualCuboid(
        f"/World/{surface_name}",
        f"{surface_name}",
        translation=translation,
        orientation=orientation,
        color=np.array([0., 0., 0.]),
        size=0.01,
    )
    return surface

def create_hole(hole_name, translation, orientation):
    hole = cuboid.VisualCuboid(
        f"/World/{hole_name}",
        f"{hole_name}",
        translation=translation,
        orientation=orientation,
        color=np.array([0., 0., 0.]),
        size=0.01,
    )
    return hole

def create_table(table_cfg):
    table = cuboid.VisualCuboid(
        "/World/table",
        "table",
        translation=np.array(table_cfg.translation),
        orientation=np.array(table_cfg.orientation),
        color=np.array(table_cfg.color),
        size=table_cfg.size,
    )
    return table

def create_robot(robot_cfg):
    if "xarm" in robot_cfg.name:
        from xarm_tamp.tampkit.sim_tools.robots import xarm
        robot = xarm.xArm(
            prim_path="/World/xarm7",
            translation=np.array(robot_cfg.translation),
            orientation=np.array(robot_cfg.orientation),
        )
    else:
        raise ValueError("Need to give known robot_name")
    return robot

def create_fmb(fmb_cfg):
    if fmb_cfg.task == 'momo':
        from xarm_tamp.tampkit.sim_tools.objects import fmb_momo
        block = fmb_momo.Block(
            f"/World/{fmb_cfg.name}",
            translation=np.array(fmb_cfg.translation),
            orientation=np.array(fmb_cfg.orientation)
        )
    elif fmb_cfg.task == 'simo':
        from xarm_tamp.tampkit.sim_tools.objects import fmb_simo
        block = fmb_simo.Block(
            f"/World/{fmb_cfg.name}",
            translation=np.array(fmb_cfg.translation),
            orientation=np.array(fmb_cfg.orientation)
        )
    return block

### Unit API

def unit_point():
    return np.array([0.0, 0.0, 0.0])

def unit_quat():
    return np.array([1.0, 0.0, 0.0, 0.0])

def unit_pose():
    return (unit_point(), unit_quat())

def unit_from_theta(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def get_point(body):
    return get_pose(body)[0]

def get_unit_vector(vec, norm=2):
    norm = np.linalg.norm(vec, ord=norm)
    if norm == 0:
        return vec
    return np.array(vec) / norm

### Rigid Body API

def get_bodies(world, body_types=['all']) -> Optional[List[XFormPrim]]:
    all_objects = world.scene._scene_registry._all_object_dicts

    bodies = []
    if 'all' in body_types:
        for object_dict in all_objects:
            for name, usd_obj in object_dict.items():
                if 'plane' in name:  # remove plane
                    continue
                bodies.append(usd_obj)
    if 'rigid' in body_types:
        for name, usd_obj in all_objects[0]:  # index 0 is rigid_objects
            bodies.append(usd_obj)
    if 'geom' in body_types:
        for name, usd_obj in all_objects[1]:  # index 1 is geometry_objects
            if 'plane' in name:
                continue
            bodies.append(usd_obj)
    if 'robot' in body_types:
        for name, usd_obj in all_objects[3]:  # index 3 is robots
            bodies.append(usd_obj)
    if 'xform' in body_types:
        for name, usd_obj in all_objects[4]:  # index 4 is xforms
            bodies.append(usd_obj)
    return bodies

def get_body_name(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    return body.name

def get_pose(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    pos, rot = body.get_world_pose()
    return pos, rot

def get_velocity(body: Optional[RigidPrim]):
    linear_velocity = body.linear_velocity
    angular_velocity = body.angular_velocity
    return (linear_velocity, angular_velocity)

def set_pose(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
             position=np.array([0., 0., 0.]),
             orientation=np.array([1., 0., 0., 0.])) -> None:
    body.set_world_pose(position=position, orientation=orientation)

def set_velocity(body: Optional[RigidPrim],
                 translation=np.array([0., 0., 0.]),
                 rotation=np.array([0., 0., 0.])) -> None:
    body.set_linear_velocity(velocity=translation)
    body.set_angular_velocity(velocity=rotation)

### Link Utils

def get_link(robot: Robot, name: str) -> Usd.Prim:
    # get the prim of the link according to the given name in the robot.
    link_prims = [link_prim for link_prim in robot.prim.GetChildren()]
    for link_prim in link_prims:
        if link_prim.name == name:
            return link_prim
        else:
            return None

def get_tool_link(robot: Robot, tool_name: str) -> Usd.Prim:
    tool_frame = get_link(robot, tool_name)
    return tool_frame

def get_all_links(robot: Robot) -> List[Usd.Prim]:
    link_prims = [link_prim for link_prim in robot.prim.GetChildren()]
    return link_prims

def get_moving_links(robot: Robot) -> List[Usd.Prim]:
    all_links = get_all_links(robot)
    return all_links

def get_parent(prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Get the parent of prim if it exists."""
    parent_prim = prim.GetParent()
    if not parent_prim.IsValid():
        return None
    return parent_prim

def get_child(prim: Usd.Prim, child_name: str = None) -> Optional[Usd.Prim]:
    """Get the child of prim if it exists."""
    children_prim = prim.GetChildren()
    for child_prim in children_prim:
        if not child_prim.IsValid():
            return None
        if child_prim.name == child_name:
            return child_prim
        else:
            return children_prim[-1]

def get_children(prim: Usd.Prim) -> Optional[List[Usd.Prim]]:
    """Get the children of prim if it exists."""
    children_prim = prim.GetChildren()
    for child_prim in children_prim:
        if not child_prim.IsValid():
            return None
    return children_prim

def get_all_link_parents(robot: Robot) -> Dict[str, Usd.Prim]:
    """Get all parents link."""
    parents = {}
    for link in get_all_links(robot):
        parents[link.name] = get_parent(link)
    return parents

def get_all_link_children(robot: Robot) -> Dict[str, List[Usd.Prim]]:
    """Get all children link."""
    children = {}
    for link in get_all_links(robot):
        children[link.name] = get_children(link)
    return children    

def get_link_parents(robot: Robot, link: Usd.Prim) -> Optional[Usd.Prim]:
    """Get parents link."""
    parents = get_all_link_parents(robot)
    return parents.get(link.name, None)

def get_link_children(robot: Robot, link: Usd.Prim) -> Optional[List[Usd.Prim]]:
    """Get child links."""
    children = get_all_link_children(robot)
    return children.get(link.name, [])

def get_link_descendants(robot: Robot, link: Usd.Prim, test=lambda l: True):
    """Get descendants link."""
    descendants = []
    for child in get_link_children(robot, link):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(robot, child, test=test))
    return descendants

def get_link_subtree(robot: Robot, link: Usd.Prim, **kwargs):
    """Get subtree of the given link."""
    return [link] + get_link_descendants(robot, link, **kwargs)

def get_link_pose(robot: Robot, link_name: str):
    """Get specified link pose."""
    link_names = [link_prim.name for link_prim in robot.GetChildren()]
    if link_name not in link_names:
        raise ValueError("Specified link does not exist.")
    for link_prim in robot.GetChildren():
        if link_name == link_prim.name:
            return link_prim.get_world_pose()

### Joint Utils

def get_arm_joints(robot: Robot) -> Optional[Union[list, np.ndarray, torch.Tensor]]:
    """Get arm joint indices."""
    return robot.arm_joints

def get_base_joints(robot: Robot) -> Optional[Union[list, np.ndarray, torch.Tensor]]:
    """Get base joint indices."""
    return robot.base_joints

def get_gripper_joints(robot: Robot) -> Optional[Union[list, np.ndarray, torch.Tensor]]:
    """Get gripper joint indices."""
    return robot.gripper_joints

def get_movable_joints(robot: Robot, use_gripper: bool = False) -> Optional[Union[list, np.ndarray, torch.Tensor]]:
    """Get movable joint indices."""
    if use_gripper:
        movable_joints = robot.arm_joints + robot.gripper_joints
    else:
        movable_joints = robot.arm_joints
    return movable_joints

def get_joint_positions(robot: Robot,
                        joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None):
    """Get joint positions."""
    if joint_indices == None:
        joint_indices = get_movable_joints(robot)
    joint_positions = robot.get_joint_positions(joint_indices=joint_indices)
    return joint_positions

def get_joint_velocities(robot: Robot,
                         joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None):
    """Get joint velocities."""
    if joint_indices == None:
        joint_indices = get_movable_joints(robot)
    joint_velocities = robot.get_joint_velocities(joint_indices=joint_indices)
    return joint_velocities

def get_min_limit(robot: Robot,
                  joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
    """Get joint lower limit."""
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    return robot.dof_properties.lower[joint_indices]

def get_max_limit(robot: Robot,
                  joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
    """Get joint upper limit."""
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    return robot.dof_properties.upper[joint_indices]

def get_joint_limits(robot: Robot):
    """Get joint upper and lower limits."""
    return get_min_limit(robot), get_max_limit(robot)

def get_custom_limits(robot: Robot,
                      joint_names: Optional[Union[list, np.ndarray, torch.Tensor]],
                      custom_limits: dict = {}):
    """Get custom limits."""
    joint_limits = []
    for joint in joint_names:
        if joint in custom_limits:
            joint_limits.append(custom_limits[joint])
        else:
            joint_limits.append(get_joint_limits(robot))
    return zip(*joint_limits)

def get_initial_conf(robot: Robot,
                     joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None):
    """Get joint initial configuration."""
    if joint_indices == None:
        joint_indices = get_movable_joints(robot)
    state = robot.get_joints_default_state()
    if state is None:
        initial_pos = get_joint_positions(robot, joint_indices)
        initial_vel = get_joint_velocities(robot, joint_indices)
        robot.set_joints_default_state(initial_pos, initial_vel)
    else:
        initial_pos = state.positions
        initial_pos = state[joint_indices]
    return initial_pos

def get_group_conf(robot: Robot,
                   group: str = 'arm'):
    """Get joint configuration corresponding to group."""
    if group == 'arm':
        joint_indices = robot.arm_joints
    elif group == 'gripper':
        joint_indices = robot.gripper_joints
    elif group == 'base':
        joint_indices = robot.base_joints
    elif group == 'whole_body':
        joint_indices = get_movable_joints(robot, use_gripper=True)
    return robot.get_joint_positions(joint_indices=joint_indices)

def set_joint_positions(robot: Robot,
                        positions: Optional[Union[np.ndarray, torch.Tensor]],
                        joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
    """Set joint positions."""
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    robot.set_joint_positions(positions, joint_indices)

def set_initial_conf(robot: Robot,
                     initial_conf: Optional[Union[np.ndarray, torch.Tensor]],
                     joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
    """Set joint positions to initial configuration."""
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    if initial_conf is None:
        initial_conf = get_joint_positions(robot, joint_indices)
    robot.set_joint_positions(initial_conf, joint_indices=joint_indices)

def apply_action(robot: Robot,
                 joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
                 configuration: Optional[ArticulationAction] = None):
    """Apply articulation action."""
    art_controller = robot.get_articulation_controller()
    art_controller.apply_action(configuration, indices=joint_indices)

def is_circular(robot: Robot,
                joint: Usd.Prim) -> bool:
    """Check specified joint circular."""
    if joint.IsA(UsdPhysics.FixedJoint):
        return False
    joint_index = robot.get_joint_index(joint.name)
    upper, lower = robot.dof_properties['upper'][joint_index], robot.dof_properties['lower'][joint_index]
    return upper < lower

def get_difference_fn(robot: Robot,
                      joints: List[Usd.Prim]):
    """Get difference between joint configuraitons."""
    circular_joints = [is_circular(robot, joint) for joint in joints]
    def fn(q2, q1):
        return tuple(circular_difference(value2, value1) if circular else (value2 - value1)
                for circular, value2, value1 in zip(circular_joints, q2, q1))
    return fn

def get_refine_fn(robot: Robot,
                  joints: List[Usd.Prim],
                  num_steps: int = 0):
    """Refine given joint configuration."""
    difference_fn = get_difference_fn(robot, joints)
    num_steps = num_steps + 1
    def fn(q1, q2):
        q = q1
        for i in range(num_steps):
            positions = (1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            q = tuple(positions)
            yield q
    return fn

def get_extend_fn(robot: Robot,
                  joints: List[Usd.Prim],
                  norm: int = 2):
    """Extend given joint configuration."""
    resolutions = math.radians(3) * np.ones(len(joints))
    difference_fn = get_difference_fn(robot, joints)
    def fn(q1, q2):
        steps = int(np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=norm))
        refine_fn = get_refine_fn(robot, joints, num_steps=steps)
        return refine_fn(q1, q2)
    return fn

def get_distance_fn(robot: Robot,
                    joints: List[Usd.Prim]):
    """Get distance between two configurations."""
    weights = 1 * np.ones(len(joints))
    difference_fn = get_difference_fn(robot, joints)
    def fn(q1: torch.Tensor, q2: torch.Tensor):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
    return fn

def refine_path(robot: Robot,
                joints: List[Usd.Prim],
                waypoints: Sequence,
                num_steps: int = 0):
    """Refine path."""
    refine_fn = get_refine_fn(robot, joints, num_steps)
    refined_path = []
    for v1, v2 in get_pairs(waypoints):
        refined_path.extend(refine_fn(v1, v2))
    return refined_path

def get_pose_distance(pose1: Optional[Union[list, np.ndarray, torch.Tensor]],
                      pose2: Optional[Union[list, np.ndarray, torch.Tensor]]):
    """Get pose distance."""
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    pos_distance = get_distance(pos1, pos2)
    ori_distance = quat_diff_rad(quat1, quat2)
    return pos_distance, ori_distance

def interpolate_poses(pose1: Optional[Union[list, np.ndarray, torch.Tensor]],
                      pose2: Optional[Union[list, np.ndarray, torch.Tensor]],
                      pos_step_size: float = 0.01,
                      ori_step_size: float = np.pi/16):
    """Interpolate two different poses."""
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    num_steps = max(2, int(math.ceil(max(
        np.divide(get_pose_distance(pose1, pose2), [pos_step_size, ori_step_size])))))
    yield pose1
    for w in np.linspace(0, 1, num=num_steps, endpoint=True)[1:-1]:
        pos = convex_combination(pos1, pos2, w=w)
        quat = quat_combination(quat1, quat2, fraction=w)
        yield (pos, quat)
    yield pose2

def iterate_approach_path(robot: Robot,
                          gripper: Usd.Prim,
                          pose: Optional[Union[list, np.ndarray, torch.Tensor]],
                          grasp: Optional[Union[list, np.ndarray, torch.Tensor]],
                          body: Optional[Union[list, np.ndarray, torch.Tensor]] = None):
    """Interpolate approach path."""
    tool_from_root = get_tool_link(robot)
    grasp_pose = multiply(pose.value, invert(grasp.value))
    approach_pose = multiply(pose.value, invert(grasp.approach))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(gripper, multiply(tool_pose, tool_from_root))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.value))
        yield

### Collision Geomtry API

def aabb_empty(aabb):
    lower, upper = aabb
    return np.less(upper, lower).any()

def aabb2d_from_aabb(aabb):
    (lower, upper) = aabb
    return lower[:2], upper[:2]

def sample_aabb(aabb):
    lower, upper = aabb
    return np.random.uniform(lower, upper)

def get_aabb(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    cache = bounds_utils.create_bbox_cache()
    body_aabb = bounds_utils.compute_aabb(cache, body.prim_path) # [min x, min y, min z, max x, max y, max z]
    lower, upper = body_aabb[:3], body_aabb[3:]
    return lower, upper

def get_center_extent(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    lower, upper = get_aabb(body)
    diff = np.array(upper) - np.array(lower)
    center = (np.array(upper) + np.array(lower)) / 2.
    return center, diff

def aabb_contains_aabb(contained, container):
    lower1, upper1 = contained
    lower2, upper2 = container
    return np.less_equal(lower2, lower1).all() and \
           np.less_equal(upper1, upper2).all()

def is_placed_on_aabb(body, bottom_aabb, above_epsilon=5e-2, below_epsilon=5e-2):
    assert (0 <= above_epsilon) and (0 <= below_epsilon)
    top_aabb = get_aabb(body)
    top_z_min = top_aabb[0][2]
    bottom_z_max = bottom_aabb[1][2]
    return ((bottom_z_max - below_epsilon) <= top_z_min <= (bottom_z_max + above_epsilon)) and \
           (aabb_contains_aabb(aabb2d_from_aabb(top_aabb), aabb2d_from_aabb(bottom_aabb)))

def is_placement(body, surface, **kwargs):
    if get_aabb(surface) is None:
        return False
    return is_placed_on_aabb(body, get_aabb(surface), **kwargs)

def is_insertion(body, hoke, **kwargs):
    if get_aabb(hoke) is None:
        return False
    return is_placed_on_aabb(body, get_aabb(hoke), **kwargs)

def tform_point(affine, point):
    return multiply(affine, [point, unit_quat()])[0]

def apply_affine(affine, points):
    return [tform_point(affine, p) for p in points]

def vertices_from_rigid(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
                        link: Usd.Prim
    ) -> Optional[np.ndarray]:
    """Get verticies from rigid body."""
    try:
        coord_prim = stage_utils.get_current_stage().GetPrimAtPath(link.prim_path)
        vertices = mesh_utils.get_mesh_vertices_relative_to(body, coord_prim)
        return vertices
    except:
        raise NotImplementedError("Please add vertices_from_link")

def approximate_as_prism(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
                         body_pose = unit_pose(),
                         **kwargs):
    """Approximate rigid body as prism."""
    vertices = apply_affine(body_pose, vertices_from_rigid(body, **kwargs))
    lower, upper = np.min(vertices, axis=0), np.max(vertices, axis=0) 
    diff = np.array(upper) - np.array(lower)
    center = (np.array(upper) + np.array(lower)) / 2.
    return center, diff

# TODO
def get_closest_points(body1, body2, link1, link2):
    pass

def pairwise_link_collision(body1, link1, body2, link2=None, **kwargs):
    return len(get_closest_points(body1, body2, link1=link1, link2=link2, **kwargs)) != 0

def any_link_pair_collision(body1, links1, body2, links2=None, **kwargs):
    if links1 is None:
        links1 = get_all_links(body1)
    if links2 is None:
        links2 = get_all_links(body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            return True
    return False

def body_collision(body1, body2, **kwargs):
    return len(get_closest_points(body1, body2, **kwargs)) != 0

def pairwise_collision(body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        return any_link_pair_collision(body1, links1, body2, links2, **kwargs)
    return body_collision(body1, body2, **kwargs)

def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and \
           np.less_equal(values, upper_limits).all()

def parse_body(robot: Robot, link=None):
    collision_pair = namedtuple('Collision', ['robot', 'links'])
    return robot if isinstance(robot, tuple) else collision_pair(robot, link)

def flatten_links(robot: Robot, links=None):
    if links is None:
        links = [link_prim.name for link_prim in robot.GetChildren()]
    collision_pair = namedtuple('Collision', ['robot', 'links'])
    return {collision_pair(robot, frozenset([link])) for link in links}

def expand_links(robot: Robot, **kwargs):
    body, links = parse_body(robot, **kwargs)
    if links is None:
        links = get_all_links(body)
    collision_pair = namedtuple('Collision', ['robot', 'links'])
    return collision_pair(body, links)

### Mathmatic Utils

def wrap_interval(value, interval=(0., 1.)):
    lower, upper = interval
    assert lower <= upper
    return (value - lower) % (upper - lower) + lower

def circular_difference(theta2, theta1, **kwargs):
    diff_theta = theta2 - theta1
    interval = (-np.pi, -np.pi + 2 * np.pi)
    return wrap_interval(diff_theta, interval=interval)

def flatten(iterable_of_iterables):
    return (item for iterables in iterable_of_iterables for item in iterables)

def convex_combination(x, y, w=0.5):
    return (1 - w) * np.array(x) + w * np.array(y)

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis."""
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions."""
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
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < np.finfo(float).eps * 4.0:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def quat_combination(quat1, quat2, fraction=0.5):
    return quaternion_slerp(quat1, quat2, fraction)

def get_pairs(sequence):
    sequence = list(sequence)
    return zip(sequence[:-1], sequence[1:])

def get_distance(p1, p2, **kwargs):
    assert len(p1) == len(p2)
    diff = np.array(p2) - np.array(p1)
    return np.linalg.norm(diff, ord=2)

def multiply(pose1: Optional[Union[list, np.ndarray, torch.Tensor]],
             pose2: Optional[Union[list, np.ndarray, torch.Tensor]]
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Get transformation."""
    transform1 = np.zeros((4, 4))
    transform1[3, 3] = 1.0
    if len(pose1[1]) == 4:
        transform1[:3, :3] = Rotation.from_quat(pose1[1]).as_matrix()
    elif len(pose1[1]) == 3:
        transform1[:3, :3] = Rotation.from_euler('xyz', pose1[1]).as_matrix()
    transform1[:3, 3] = pose1[0]

    transform2 = np.zeros((4, 4))
    transform2[3, 3] = 1.0
    if len(pose2[1]) == 4:
        transform2[:3, :3] = Rotation.from_quat(pose2[1]).as_matrix()
    elif len(pose2[1]) == 3:
        transform2[:3, :3] = Rotation.from_euler('xyz', pose2[1]).as_matrix()
    transform2[:3, 3] = pose2[0]

    result = transform1 @ transform2
    result_pos = result[:3, 3]
    result_rot = Rotation.from_matrix(result[:3, :3]).as_matrix()
    return result_pos, result_rot

def invert(pose: Optional[Union[list, np.ndarray, torch.Tensor]]
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Get inverse transformation."""
    transform = np.zeros((4, 4))
    transform[3, 3] = 1.0
    if len(pose[1]) == 4:
        transform[:3, :3] = Rotation.from_quat(pose[1]).as_matrix()
    elif len(pose[1]) == 3:
        transform[:3, :3] = Rotation.from_euler('xyz', pose[1]).as_matrix()
    transform[:3, 3] = pose[0]

    result = np.linalg.inv(transform)
    result_pos = result[:3, 3]
    result_rot = Rotation.from_matrix(result[:3, :3]).as_matrrix()
    return result_pos, result_rot

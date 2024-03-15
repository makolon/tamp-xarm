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
import carb
import copy
import math
import time
import numpy as np
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.bounds as bounds_utils
from itertools import count, product
from typing import Union, Optional, Tuple, List
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from tampkit.sim_tools.isaacsim.robots import xarm
from tampkit.sim_tools.isaacsim.objects import fmb_momo, fmb_simo
from tampkit.sim_tools.isaacsim.geometry import Attachment
from tampkit.sim_tools.isaacsim.curobo_utils import CuroboController
from tampkit.sim_tools.isaacsim.usd_helper import *

### Simulation API

def connect():
    global simulation_app
    return simulation_app

def disconnect():
    global simulation_app
    simulation_app.close()
    
def step_simulation(world):
    world.step(render=True)

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

def create_surface(surface_name, translation, orientation):
    surface = cuboid.VisualCuboid(
        f"/World/{surface_name}",
        translation=translation,
        orientation=orientation,
        color=np.array([0., 0., 0.]),
        size=0.01,
    )
    return surface

def create_hole(hole_name, translation, orientation):
    hole = cuboid.VisualCuboid(
        f"/World/{hole_name}",
        translation=translation,
        orientation=orientation,
        color=np.array([0., 0., 0.]),
        size=0.01,
    )
    return hole

def create_table(table_cfg):
    table = cuboid.VisualCuboid(
        "/World/table",
        translation=np.array(table_cfg.translation),
        orientation=np.array(table_cfg.orientation),
        color=np.array(table_cfg.color),
        size=table_cfg.size,
    )
    return table

def create_robot(robot_cfg):
    if "xarm" in robot_cfg.name:
        robot = xarm.xArm(
            "/World/xarm7",
            translation=np.array(robot_cfg.translation),
            orientation=np.array(robot_cfg.orientation),
        )
    else:
        raise ValueError("Need to give known robot_name")
    return robot

def create_fmb(fmb_cfg):
    if fmb_cfg.task == 'momo':
        block = fmb_momo.Block(
            f"/World/{fmb_cfg.name}",
            translation=np.array(fmb_cfg.translation),
            orientation=np.array(fmb_cfg.orientation)
        )
    elif fmb_cfg.task == 'simo':
        block = fmb_simo.Block(
            f"/World/{fmb_cfg.name}",
            translation=np.array(fmb_cfg.translation),
            orientation=np.array(fmb_cfg.orientation)
        )
    return block

### Rigid Body API

def get_bodies():
    return []

def get_body_name(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    return body.name

def get_velocity(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    linear_velocity = body.linear_velocity
    angular_velocity = body.angular_velocity
    return (linear_velocity, angular_velocity)

def get_pose(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    return body.get_world_pose()

def set_pose(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
             translation=np.array([0., 0., 0.]),
             orientation=np.array([1., 0., 0., 0.])) -> None:
    body.set_world_pose(translation=translation, orientation=orientation)

def set_velocity(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
                 translation=np.array([0., 0., 0.]),
                 rotation=np.array([0., 0., 0.])) -> None:
    body.set_linear_velocity(velocity=translation)
    body.set_angular_velocity(velocity=rotation)

### Link Utils

def get_link(robot: Robot, name: str) -> Usd.Prim:
    # get the prim of the link according to the given name in the robot.
    link_prims = [link_prim for link_prim in robot.GetChildren()]
    for link_prim in link_prims:
        if link_prim.name == name:
            return link_prim
        else:
            return None

def get_tool_link(robot: Robot, tool_name: str) -> Usd.Prim:
    tool_frame = get_link(robot, tool_name)
    return tool_frame

def get_all_links(robot: Robot) -> List[Usd.Prim]:
    link_prims = [link_prim for link_prim in robot.GetChildren()]
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
    parents = get_all_link_parents(robot)
    return parents.get(link.name, None)

def get_link_children(robot: Robot, link: Usd.Prim) -> Optional[List[Usd.Prim]]:
    children = get_all_link_children(robot)
    return children.get(link.name, [])

def get_link_descendants(robot: Robot, link: Usd.Prim, test=lambda l: True):
    """Get the descendants link """
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
    link_names = [link_prim.name for link_prim in robot.GetChildren()]
    if link_name not in link_names:
        raise ValueError("Specified link does not exist.")
    for link_prim in robot.GetChildren():
        if link_name == link_prim.name:
            return link_prim.get_world_pose()

### Joint Utils

def get_arm_joints(robot: Robot) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """Get arm joint indices."""
    return robot.arm_joints

def get_base_joints(robot: Robot) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """Get base joint indices."""
    return robot.base_joints

def get_gripper_joints(robot: Robot) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """Get gripper joint indices."""
    return robot.gripper_joints

def get_movable_joints(robot: Robot, use_gripper: bool = False) -> Optional[Union[np.ndarray, torch.Tensor]]:
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
    joint_positions = robot.get_jonit_positions(joint_indices=joint_indices)
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
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    return robot.dof_properties.lower[joint_indices]

def get_max_limit(robot: Robot,
                  joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    return robot.dof_properties.upper[joint_indices]

def get_joint_limits(robot: Robot):
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
        default_pos = robot.get_joint_positions(joint_indices=joint_indices)
        return default_pos
    return state.position

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
                 conf: Optional[Union[np.ndarray, torch.Tensor]],
                 joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
    """Set joint positions to initial configuration."""
    if joint_indices is None:
        joint_indices = get_movable_joints(robot)
    robot.set_joint_positions(conf, joint_indices=joint_indices)

# TODO
def joint_controller(robot: Robot,
                     joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
                     configuration: Optional[Union[list, np.ndarray, torch.Tensor]] = None):
    pass

def is_circular(robot: Robot,
                joint: Usd.Prim) -> bool:
    if joint.IsA(UsdPhysics.FixedJoint):
        return False
    joint_index = robot.get_joint_index(joint.name)
    upper, lower = robot.dof_properties['upper'][joint_index], robot.dof_properties['lower'][joint_index]
    return upper < lower

def get_difference_fn(robot: Robot,
                      joints: List[Usd.Prim]):
    circular_joints = [is_circular(robot, joint) for joint in joints]
    def fn(q2, q1):
        return tuple(circular_difference(value2, value1) if circular else (value2 - value1)
                for circular, value2, value1 in zip(circular_joints, q2, q1))
    return fn

def get_refine_fn(robot: Robot,
                  joints: List[Usd.Prim],
                  num_steps: int = 0):
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
                  norm=2):
    resolutions = math.radians(3) * np.ones(len(joints))
    difference_fn = get_difference_fn(robot, joints)
    def fn(q1, q2):
        steps = int(np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=norm))
        refine_fn = get_refine_fn(robot, joints, num_steps=steps)
        return refine_fn(q1, q2)
    return fn

def get_distance_fn(robot: Robot,
                    joints: List[Usd.Prim]):
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
    refine_fn = get_refine_fn(robot, joints, num_steps)
    refined_path = []
    for v1, v2 in get_pairs(waypoints):
        refined_path.extend(refine_fn(v1, v2))
    return refined_path

### Grasp

def get_side_grasps():
    pass

def get_top_grasps():
    pass

def compute_grasp_width(robot, arm, body, grasp_pose, **kwargs):
    tool_link = get_tool_link(robot)
    tool_pose = get_link_pose(robot, tool_link)
    body_pose = multiply(tool_pose, grasp_pose)
    set_pose(body, body_pose)
    gripper_joints = get_gripper_joints(robot, arm)
    return close_until_collision(robot, gripper_joints, bodies=[body], **kwargs)

def create_attachment(parent, parent_link, child):
    parent_link_pose = get_link_pose(parent, parent_link)
    child_pose = get_pose(child)
    grasp_pose = multiply(invert(parent_link_pose), child_pose)
    return Attachment(parent, parent_link, grasp_pose, child)

### Collision Geomtry API

def get_aabb(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    cache = bounds_utils.create_bbox_cache()
    body_aabb = bounds_utils.compute_aabb(cache, body.prim_path)
    return body_aabb

def get_aabb_center(aabb):
    lower, upper = aabb
    return (np.array(lower) + np.array(upper)) / 2.

def get_aabb_extent(aabb):
    lower, upper = aabb
    return np.array(upper) - np.array(lower)

def get_center_extent(body, **kwargs):
    aabb = get_aabb(body, **kwargs)
    return get_aabb_center(aabb), get_aabb_extent(aabb)

def aabb_empty(aabb):
    lower, upper = aabb
    return np.less(upper, lower).any()

def sample_aabb(aabb):
    lower, upper = aabb
    return np.random.uniform(lower, upper)

def aabb2d_from_aabb(aabb):
    (lower, upper) = aabb
    AABB = namedtuple('AABB', ['lower', 'upper'])
    return AABB(lower[:2], upper[:2])

def aabb_contains_aabb(contained, container):
    lower1, upper1 = contained
    lower2, upper2 = container
    return np.less_equal(lower2, lower1).all() and \
           np.less_equal(upper1, upper2).all()

def is_placed_on_aabb(body, bottom_aabb, above_epsilon=5e-2, below_epsilon=5e-2):
    assert (0 <= above_epsilon) and (0 <= below_epsilon)
    top_aabb = get_aabb(body) # TODO: approximate_as_prism
    top_z_min = top_aabb[0][2]
    bottom_z_max = bottom_aabb[1][2]
    return ((bottom_z_max - below_epsilon) <= top_z_min <= (bottom_z_max + above_epsilon)) and \
           (aabb_contains_aabb(aabb2d_from_aabb(top_aabb), aabb2d_from_aabb(bottom_aabb)))

def is_placement(body, surface, **kwargs):
    if get_aabb(surface) is None:
        return False
    return is_placed_on_aabb(body, get_aabb(surface), **kwargs)

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

def flatten(iterable_of_iterables):
    return (item for iterables in iterable_of_iterables for item in iterables)

def get_pairs(sequence):
    sequence = list(sequence)
    return zip(sequence[:-1], sequence[1:])

def get_distance(p1, p2, **kwargs):
    assert len(p1) == len(p2)
    diff = np.array(p2) - np.array(p1)
    return np.linalg.norm(diff, ord=2)

def multiply(*poses):
    # Initialize Transform3d object at first pose
    t = Transform3d().translate(*poses[0][0]).rotate_euler(*poses[0][1])

    # Composite the remaining poses
    for next_pose in poses[1:]:
        next_t = Transform3d().translate(*next_pose[0]).rotate_euler(*next_pose[1])
        t = t.compose(next_t)

    # Obtain final position and rotation
    final_position = t.get_matrix()[:, :3, 3]
    final_rotation = t.get_euler()

    return (final_position, final_rotation)

def invert(pose):
    # Initialize Transform3d object
    position, quaternion = pose
    t = Transform3d().translate(*position).rotate_quaternion(quaternion)

    # Compute the inverse transform
    inverted_t = t.inverse()

    # Compute inverse transform, get position and quaternion of inverse transform
    inverted_position = inverted_t.get_matrix()[:, :3, 3]
    inverted_quaternion = quaternion_invert(quaternion)

    return inverted_position.numpy(), inverted_quaternion.numpy()

def base_values_from_pose(pose, tolerance=1e-3):
    (point, quat) = pose
    x, y, _ = point
    roll, pitch, yaw = euler_from_quat(quat)
    assert (abs(roll) < tolerance) and (abs(pitch) < tolerance)
    return np.array([x, y, yaw])

def wrap_interval(value, interval=(0., 1.)):
    lower, upper = interval
    assert lower <= upper
    return (value - lower) % (upper - lower) + lower

def circular_difference(theta2, theta1, **kwargs):
    diff_theta = theta2 - theta1
    interval = (-np.pi, -np.pi + 2 * np.pi)
    return wrap_interval(diff_theta, interval=interval)

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

### Executer

def apply_commands(state, commands, time_step=None, **kwargs):
    for i, command in enumerate(commands):
        print(i, command)
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            if time_step is None:
                time.sleep(1e-2)
            else:
                time.sleep(time_step)

def control_commands(commands, **kwargs):
    for i, command in enumerate(commands):
        print(i, command)
        command.control(*kwargs)

### Saver

class Saver(object):
    def save(self):
        pass

    def restore(self):
        raise NotImplementedError()

    def __enter__(self):
        self.save()

    def __exit__(self, type, value, traceback):
        self.restore()

class PoseSaver(Saver):
    def __init__(self, body, pose=None):
        self.body = body
        if pose is None:
            pose = get_pose(self.body)
        self.pose = pose
        self.velocity = get_velocity(self.body)

    def apply_mapping(self, mapping):
        self.body = mapping.get(self.body, self.body)

    def restore(self):
        set_pose(self.body, self.pose)
        set_velocity(self.body, *self.velocity)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class ConfSaver(Saver):
    def __init__(self, body, joints=None, positions=None):
        self.body = body
        if joints is None:
            joints = get_arm_joints(self.body)
        self.joints = joints
        if positions is None:
            positions = get_joint_positions(self.body, self.joints)
        self.positions = positions
        self.velocities = get_joint_velocities(self.body, self.joints)

    @property
    def conf(self):
        return self.positions

    def apply_mapping(self, mapping):
        self.body = mapping.get(self.body, self.body)

    def restore(self):
        set_joint_positions(self.body, self.joints, self.positions, self.velocities)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class BodySaver(Saver):
    def __init__(self, body, **kwargs):
        self.body = body
        self.pose_saver = PoseSaver(body)
        self.conf_saver = ConfSaver(body, **kwargs)
        self.savers = [self.pose_saver, self.conf_saver]

    def apply_mapping(self, mapping):
        for saver in self.savers:
            saver.apply_mapping(mapping)

    def restore(self):
        for saver in self.savers:
            saver.restore()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class WorldSaver(Saver):
    def __init__(self, bodies=None):
        if bodies is None:
            bodies = get_bodies()
        self.bodies = bodies
        self.body_savers = [BodySaver(body) for body in self.bodies]

    def restore(self):
        for body_saver in self.body_savers:
            body_saver.restore()

### Geometry

class Pose:
    num = count()
    def __init__(self,
                 body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim, Robot]],
                 value: Optional[Union[np.ndarray, torch.Tensor]] = None):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.index = next(self.num)

    @property
    def bodies(self):
        return flatten_links(self.body)

    def assign(self):
        set_pose(self.body, self.value)

    def iterate(self):
        yield self

    def to_base_conf(self):
        values = base_values_from_pose(self.value)
        return Conf(self.body, range(len(values)), values)

    def __repr__(self):
        index = self.index
        return '{}'.format(index)

class Grasp:
    def __init__(self,
                 body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
                 value: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 approach: Optional[Union[np.ndarray, torch.Tensor]] = None):
        self.body = body
        self.value = tuple(value)
        self.approach = tuple(approach)

    def get_attachment(self, robot, arm):
        tool_link = get_link(robot, get_tool_link(robot))
        return Attachment(robot, tool_link, self.value, self.body)

    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

class Attachment:
    def __init__(self,
                 parent,
                 parent_link,
                 grasp_pose,
                 child):
        self.parent = parent
        self.parent_link = parent_link
        self.grasp_pose = grasp_pose
        self.child = child

    @property
    def bodies(self):
        return flatten_links(self.child) | flatten_links(self.parent, get_link_subtree(
            self.parent, self.parent_link))

    def assign(self):
        parent_link_pose = get_link_pose(self.parent, self.parent_link)
        child_pose = multiply(parent_link_pose, self.grasp_pose)
        set_pose(self.child, child_pose)
        return child_pose

    def apply_mapping(self, mapping):
        self.parent = mapping.get(self.parent, self.parent)
        self.child = mapping.get(self.child, self.child)

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.parent, self.child)

class Conf:
    def __init__(self,
                 robot: Robot,
                 values: Optional[Union[np.ndarray, torch.Tensor]],
                 joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init=False):
        self.robot = robot
        self.joints = joint_indices
        if values is None:
            values = get_joint_positions(self.robot, self.joints)
        self.values = tuple(values)
        self.init = init

    @property
    def bodies(self):
        return flatten_links(self.robot, get_moving_links(self.robot))

    def assign(self):
        set_joint_positions(self.robot, self.values, self.joints)

    def iterate(self):
        yield self

    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)

class State:
    def __init__(self,
                 attachments: dict = {},
                 cleaned: set = set(),
                 cooked: set = set()):
        self.poses = {body: Pose(body, get_pose(body))
                      for body in get_bodies() if body not in attachments}
        self.grasps = {}
        self.attachments = attachments
        self.cleaned = cleaned
        self.cooked = cooked

    def assign(self):
        for attachment in self.attachments.values():
            attachment.assign()
            
class Commands:
    def __init__(self, state, savers=[], commands=[]):
        self.state = state
        self.savers = tuple(savers)
        self.commands = tuple(commands)

    def assign(self):
        for saver in self.savers:
            saver.restore()
        return copy.copy(self.state)

    def apply(self, state, **kwargs):
        for command in self.commands:
            for result in command.apply(state, **kwargs):
                yield result

    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)
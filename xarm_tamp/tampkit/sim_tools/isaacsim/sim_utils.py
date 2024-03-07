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
import numpy as np
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.bounds as bounds_utils
from typing import Union, Optional, Tuple
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from tampkit.sim_tools.isaacsim.robots import xarm
from tampkit.sim_tools.isaacsim.objects import fmb_momo, fmb_simo
from tampkit.sim_tools.isaacsim.curobo_utils import CuroboController

### Simulation Utils

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

### Robot Utils (Getter)

def get_tool_frame(robot: Robot):
    return robot.tool_frame

def get_arm_joints(robot: Robot):
    return robot.arm_joints

def get_base_joints(robot: Robot):
    return robot.base_joints

def get_gripper_joints(robot: Robot):
    return robot.gripper_joints

def get_joint_positions(robot: Robot,
                        joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None):
    if joint_indices == None:
        joint_indices = [robot.get_joint_index(name) \
            for name in robot.arm_joints]
    joint_positions = robot.get_jonit_positions(joint_indices=joint_indices)
    return joint_positions

def get_link_pose(robot: Robot, link_name: str):
    link_names = [link_prim.name for link_prim in robot.GetChildren()]
    if link_name not in link_names:
        raise ValueError("Specified link does not exist.")
    else:
        prim_path = get_prim_from_name(link_name)
        link_prim = prims_utils.get_prim_at_path(prim_path)
        return link_prim.get_local_pose()

def get_prim_from_name(name: str):
    # TODO: fix this function to recursively search prim_path
    prim_paths = prims_utils.find_matching_prim_paths(f"/World/*{name}")
    print('prim_paths:', prim_paths)
    if len(prim_paths) == 1:
        return prim_paths[0]
    else:
        raise ValueError("The specified prim path does not exist.")

def get_min_limit(robot: Robot) -> np.ndarray:
    return robot.dof_properties.lower

def get_max_limit(robot: Robot) -> np.ndarray:
    return robot.dof_properties.upper

def get_custom_limits(robot: Robot,
                      joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]],
                      cutom_limits: dict = {}):
    # TODO: fix this
    return get_min_limit(robot), get_max_limit(robot)

def get_initial_conf(robot: Robot,
                     joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]] = None):
    if joint_indices == None:
        joint_indices = [robot.get_dof_index(name) \
            for name in robot.arm_joints]
    state = robot.get_joints_default_state()
    if state is None:
        default_pos = robot.get_joint_positions(joint_indices=joint_indices)
    return default_pos

def get_group_conf(robot: Robot,
                   group: str = 'arm'):
    if group == 'arm':
        joint_indices = [robot.get_dof_index(name) \
            for name in robot.arm_joints]
    elif group == 'gripper':
        joint_indices = [robot.get_dof_index(name) \
            for name in robot.gripper_joints]
    elif group == 'base':
        joint_indices = [robot.get_dof_index(name) \
            for name in robot.base_joints]
    elif group == 'whole_body':
        joint_indices = [robot.get_dof_index(name) \
            for name in [robot.arm_joints+robot.base_joints+robot.gripper_joints]]
    return robot.get_joint_positions(joint_indices=joint_indices)

def get_moving_links(robot: Robot,
                     joint_indices: Optional[Union[list, np.ndarray, torch.Tensor]]):
    moving_links = set()
    for joint in joint_indices:
        link = child_link_from_joint(joint) # TODO: add
        if link not in moving_links:
            moving_links.update(get_link_subtree(robot, link))
    return list(moving_links)

def get_distance(p1, p2, **kwargs):
    assert len(p1) == len(p2)
    diff = np.array(p2) - np.array(p1)
    return np.linalg.norm(diff, ord=2)

def get_target_point(conf):
    robot = conf.body
    link = link_from_name(robot, 'torso_lift_link') # TODO: fix this

    with BodySaver(conf.body):
        conf.assign()
        lower, upper = get_aabb(robot, link)
        center = np.average([lower, upper], axis=0)
        point = np.array(get_group_conf(conf.body, 'base'))
        point[2] = center[2]
        return point

def get_all_link_parents(body):
    return {link: get_link_parent(body, link) for link in get_links(body)}

def get_all_link_children(body):
    children = {}
    for child, parent in get_all_link_parents(body).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children    

def get_link_children(body, link):
    children = get_all_link_children(body)
    return children.get(link, [])
    
def get_link_descendants(body, link, test=lambda l: True):
    descendants = []
    for child in get_link_children(body, link):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(body, child, test=test))
    return descendants

def get_link_subtree(body, link, **kwargs):
    return [link] + get_link_descendants(body, link, **kwargs)

### Geom/Rigid/XForm Utils (Getter)

def get_target_path(trajectory):
    return [get_target_point(conf) for conf in trajectory.path]

def get_pose(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    return body.get_local_pose()

def get_bodies():
    return []

def get_body_name(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    return body.name

def get_aabb(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]]):
    cache = bounds_utils.create_bbox_cache()
    body_aabb = bounds_utils.compute_aabb(cache, body.prim_path)
    return body_aabb

def get_extend_fn(robot: Robot, joints):
    def fn(start_conf: torch.Tensor, end_conf: torch.Tensor):
        pass
    return fn

def get_distance_fn():
    def fn(q1: torch.Tensor, q2: torch.Tensor):
        pass
    return fn

### Robot Utils (Setter)

def set_joint_positions(robot: Robot,
                        positions: Optional[Union[np.ndarray, torch.Tensor]],
                        joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
    if joint_indices is None:
        joint_indices = [robot.get_dof_index(name) \
            for name in robot.arm_joints]
    robot.set_joint_positions(positions, joint_indices)

def set_pose(body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
             translation=np.array([0., 0., 0.]),
             orientation=np.array([1., 0., 0., ])) -> None:
    body.set_local_pose(translation=translation, orientation=orientation)

def set_arm_conf(robot: Robot,
                 conf: Optional[Union[np.ndarray, torch.Tensor]],
                 joint_indices: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
    if joint_indices is None:
        joint_indices = [robot.get_dof_index(name) \
            for name in robot.arm_joints]
    robot.set_joint_positions(conf, joint_indices=joint_indices)

### Utility API

def step_simulation(world):
    world.step(render=True)

def joint_controller(world):
    return CuroboController()

def remove_redundant(path, tolerance=1e-3):
    assert path
    new_path = [path[0]]
    for conf in path[1:]:
        difference = np.array(new_path[-1]) - np.array(conf)
        if not np.allclose(np.zeros(len(difference)), difference, atol=tolerance, rtol=0):
            new_path.append(conf)
    return new_path

def get_unit_vector(vec, norm=2):
    norm = np.linalg.norm(vec, ord=norm)
    if norm == 0:
        return vec
    return np.array(vec) / norm

def waypoints_from_path(path, tolerance=1e-3):
    path = remove_redundant(path, tolerance=tolerance)
    if len(path) < 2:
        return path
    difference_fn = lambda q2, q1: np.array(q2) - np.array(q1)

    waypoints = [path[0]]
    last_conf = path[1]
    last_difference = get_unit_vector(difference_fn(last_conf, waypoints[-1]))
    for conf in path[2:]:
        difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        if not np.allclose(last_difference, difference, atol=tolerance, rtol=0):
            waypoints.append(last_conf)
            difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        last_conf = conf
        last_difference = difference
    waypoints.append(last_conf)
    return waypoints

def link_from_name(robot: Robot, name: str):
    prim_path = get_prim_from_name(name)
    link_prim = prims_utils.get_prim_at_path(prim_path)
    return link_prim

def joints_from_names(body, names):
    return tuple(joint_from_name(body, name) for name in names)

def create_attachment(parent, parent_link, child):
    parent_link_pose = get_link_pose(parent, parent_link)
    child_pose = get_pose(child)
    grasp_pose = multiply(invert(parent_link_pose), child_pose)
    return Attachment(parent, parent_link, grasp_pose, child)

def flatten_links(robot: Robot, links=None):
    if links is None:
        links = [link_prim.name for link_prim in robot.GetChildren()]
    collision_pair = namedtuple('Collision', ['robot', 'links'])
    return {collision_pair(robot, frozenset([link])) for link in links}

def base_values_from_pose(pose, tolerance=1e-3):
    (point, quat) = pose
    x, y, _ = point
    roll, pitch, yaw = euler_from_quat(quat)
    assert (abs(roll) < tolerance) and (abs(pitch) < tolerance)
    return np.array([x, y, yaw])

def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and \
           np.less_equal(values, upper_limits).all()

def parse_body(body, link=None):
    collision_pair = namedtuple('Collision', ['robot', 'links'])
    return body if isinstance(body, tuple) else collision_pair(body, link)

def get_all_links(robot: Robot):
    return [link_prim for link_prim in robot.GetChildren()]

def expand_links(body, **kwargs):
    body, links = parse_body(body, **kwargs)
    if links is None:
        links = get_all_links(body)
    collision_pair = namedtuple('Collision', ['robot', 'links'])
    return collision_pair(body, links)

def pairwise_link_collision(body1, link1, body2, link2=None, **kwargs):
    return len(get_closest_points(body1, body2, link1=link1, link2=link2, **kwargs)) != 0

def any_link_pair_collision(body1, links1, body2, links2=None, **kwargs):
    # TODO: this likely isn't needed anymore
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

def iterate_approach_path(robot, arm, gripper, pose, grasp, body=None):
    tool_from_root = get_tool_from_root(robot, arm)
    grasp_pose = multiply(pose.value, invert(grasp.value))
    approach_pose = multiply(pose.value, invert(grasp.approach))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(gripper, multiply(tool_pose, tool_from_root))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.value))
        yield

def is_placement(body, surface, **kwargs):
    if get_aabb(surface) is None:
        return False
    return is_placed_on_aabb(body, get_aabb(surface), **kwargs)

def is_placement(body, hole, **kwargs):
    if get_aabb(hole) is None:
        return False
    return is_inserted_on_aabb(body, get_aabb(hole), **kwargs)

def multiply():
    pass

def get_side_grasps():
    pass

def unit_quat():
    return np.array([1.0, 0.0, 0.0, 0.0])

def compute_grasp_width(robot, arm, body, grasp_pose, **kwargs):
    tool_link = get_tool_frame(robot)
    tool_pose = get_link_pose(robot, tool_link)
    body_pose = multiply(tool_pose, grasp_pose)
    set_pose(body, body_pose)
    gripper_joints = get_gripper_joints(robot, arm)
    return close_until_collision(robot, gripper_joints, bodies=[body], **kwargs)

def unit_pose():
    return (unit_point(), unit_quat())

def get_point(body):
    return get_pose(body)[0]

def get_center_extent(body, **kwargs):
    aabb = get_aabb(body, **kwargs)
    return get_aabb_center(aabb), get_aabb_extent(aabb)

def aabb_empty(aabb):
    lower, upper = aabb
    return np.less(upper, lower).any()

def sample_aabb(aabb):
    lower, upper = aabb
    return np.random.uniform(lower, upper)

def unit_from_theta(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def apply_commands(state, commands, time_step=None, pause=False, **kwargs):
    for i, command in enumerate(commands):
        print(i, command)
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
            else:
                wait_for_duration(time_step)
        if pause:
            wait_if_gui()

def control_commands(commands, **kwargs):
    for i, command in enumerate(commands):
        print(i, command)
        command.control(*kwargs)

### Savers

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
            joints = get_movable_joints(self.body)
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
import torch
import numpy as np
from typing import Optional, Union
from itertools import count
from tampkit.sim_tools.isaacsim.sim_utils import (
    # Getter
    get_bodies, get_pose, get_joint_positions, get_moving_links,
    get_tool_frame, get_link_pose, get_link_subtree,
    # Setter
    set_pose, set_joint_positions, 
    # Utils
    base_values_from_pose, body_from_end_effector, flatten_links, link_from_name
)

from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim


class Pose(object):
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

class Grasp(object):
    def __init__(self,
                 body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
                 value: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 approach: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 carry: Optional[Union[np.ndarray, torch.Tensor]] = None):
        self.body = body
        self.value = tuple(value)
        self.approach = tuple(approach)
        self.carry = tuple(carry)

    def get_attachment(self, robot, arm):
        tool_link = link_from_name(robot, get_tool_frame(robot))
        return Attachment(robot, tool_link, self.value, self.body)

    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

class Attachment(object):
    def __init__(self,
                 parent,
                 parent_link,
                 grasp_pose,
                 child):
        self.parent = parent # TODO: support no parent
        self.parent_link = parent_link
        self.grasp_pose = grasp_pose
        self.child = child

    @property
    def bodies(self):
        return flatten_links(self.child) | flatten_links(self.parent, get_link_subtree(
            self.parent, self.parent_link))

    def assign(self):
        parent_link_pose = get_link_pose(self.parent, self.parent_link)
        child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        set_pose(self.child, child_pose)
        return child_pose

    def apply_mapping(self, mapping):
        self.parent = mapping.get(self.parent, self.parent)
        self.child = mapping.get(self.child, self.child)

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.parent, self.child)

class Conf(object):
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
    def bodies(self): # TODO: misnomer
        return flatten_links(self.robot, get_moving_links(self.robot, self.joints))

    def assign(self):
        set_joint_positions(self.robot, self.values, self.joints)

    def iterate(self):
        yield self

    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)

class State(object):
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
import time
import torch
import numpy as np
from itertools import count
from typing import List, Optional, Union

from tampkit.sim_tools.sim_utils import (
    add_fixed_constraint, apply_action, flatten,
    get_pose, get_joint_positions, get_link_pose, get_movable_joints,
    multiply, refine_path, remove_fixed_constraint,
    set_pose, set_joint_positions
)
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Usd


##################################################

class BodyPose:
    
    num = count()
    def __init__(self,
                 body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
                 pose: Optional[Union[list, np.ndarray, torch.Tensor]]):
        if pose is None:
            pose = get_pose(body)
        self.body = body
        self.pose = pose
        self.index = next(self.num)
        
    @property
    def value(self):
        return self.pose
    
    def assign(self):
        set_pose(self.body, self.pose)
        return self.pose
    
    def __repr__(self):
        index = self.index
        return 'p{}'.format(index)
    
class BodyGrasp:
    
    num = count()
    def __init__(self,
                 robot: Robot,
                 link: Usd.Prim,
                 body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
                 grasp_pose: Optional[Union[list, np.ndarray, torch.Tensor]],
                 approach_pose: Optional[Union[list, np.ndarray, torch.Tensor]]):
        self.robot = robot
        self.link = link
        self.body = body
        self.grasp_pose = grasp_pose
        self.approach_pose = approach_pose
        self.index = next(self.num)
        
    @property
    def value(self):
        return self.grasp_pose
    
    @property
    def approach(self):
        return self.approach_pose
    
    def assign(self):
        parent_link_pose = get_link_pose(self.robot, self.link)
        child_pose = multiply(parent_link_pose, self.grasp_pose)
        set_pose(self.body, child_pose)
        return child_pose
    
    def __repr__(self):
        index = self.index
        return 'g{}'.format(index)
    
class BodyConf:
    
    num = count()
    def __init__(self,
                 robot: Robot,
                 joints: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 configuration: Optional[Union[list, np.ndarray, torch.Tensor]] = None):
        if joints is None:
            joints = get_movable_joints(robot)
        if configuration is None:
            configuration = get_joint_positions(robot, joints)
        self.robot = robot
        self.joints = joints
        self.configuration = configuration
        self.index = next(self.index)
        
    @property
    def value(self):
        return self.configuration
    
    def assign(self):
        set_joint_positions(self.robot, self.joints, self.configuration)
        return self.configuration
    
    def __repr__(self):
        index = self.index
        return 'q{}'.format(index)
    
class BodyPath:
    
    def __init__(self,
                 robot: Robot,
                 path: List[ArticulationAction],
                 joints: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 attachments: List[BodyPose] = []):
        if joints is None:
            joints = get_movable_joints(robot)
        self.robot = robot
        self.path = path
        self.joints = joints
        self.attachments = attachments
        
    def bodies(self):
        return set([self.robot] + [attachment.body for attachment in self.attachments])
    
    def iterator(self):
        for i, configuration in enumerate(self.path):
            set_joint_positions(self.robot, self.joints, configuration)
            for grasp in self.attachments:
                grasp.assign()
            yield i
            
    def control(self, dt=0):
        for values in self.path:
            apply_action(self.robot, self.joints, values)

    def refine(self, num_steps=0):
        return self.__class__(self.robot, refine_path(self.robot, self.joints, self.path, num_steps), self.joints, self.attachments)

    def reverse(self):
        return self.__class__(self.robot, self.path[::-1], self.joints, self.attachments)

    def __repr__(self):
        return '{}({},{},{},{})'.format(self.__class__.__name__, self.robot, len(self.joints), len(self.path), len(self.attachments))

class Command:
    
    num = count()
    def __init__(self, body_paths):
        self.body_paths = body_paths
        self.index = next(self.num)
        
    def bodies(self):
        return set(flatten(path.bodies() for path in self.body_paths))
    
    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = '{},{}) step>'.format(i, j)
                print(msg)
                
    def execute(self, time_step=0.05):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                time.sleep(time_step)
                
    def control(self):
        for body_path in self.body_paths:
            body_path.control()
            
    def refine(self, **kwargs):
        return self.__class__([body_path.refine(**kwargs) for body_path in self.body_paths])

    def reverse(self):
        return self.__class__([body_path.reverse() for body_path in reversed(self.body_paths)])

    def __repr__(self):
        index = self.index
        return 'c{}'.format(index)

class Attach:

    def __init__(self,
                 body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
                 robot: Robot,
                 link: Usd.Prim):
        self.body = body
        self.robot = robot
        self.link = link
        
    def bodies(self):
        return {self.body, self.robot}

    def control(self, **kwargs):
        add_fixed_constraint(self.body, self.robot, self.link)

    def iterator(self, **kwargs):
        return []

    def refine(self, **kwargs):
        return self

    def reverse(self):
        return Detach(self.body, self.robot, self.link)
    
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.robot, self.body)

class Detach:

    def __init__(self,
                 body: Optional[Union[GeometryPrim, RigidPrim, XFormPrim]],
                 robot: Robot,
                 link: Usd.Prim):
        self.body = body
        self.robot = robot
        self.link = link

    def bodies(self):
        return {self.body, self.robot}

    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)

    def iterator(self, **kwargs):
        return []

    def refine(self, **kwargs):
        return self

    def reverse(self):
        return Attach(self.body, self.robot, self.link)
    
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.robot, self.body)
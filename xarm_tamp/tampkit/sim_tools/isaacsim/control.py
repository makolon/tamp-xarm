import copy
import time
import numpy as np
from geometry import Pose
from tampkit.sim_tools.isaacsim.sim_utils import (
    # Getter
    get_distance, get_group_conf, get_target_path, get_gripper_joints,
    get_joint_positions, get_extend_fn, get_body_name, get_link_pose,
    get_min_limit, get_pose, get_tool_frame,
    # Setter
    set_joint_positions, set_pose, 
    # Utils
    step_simulation, joint_controller,
    waypoints_from_path, link_from_name, create_attachment, add_fixed_constraint,
    joints_from_names, remove_fixed_constraint
)
# TODO: fix
from omni.isaac.core.utils.torch.transformations import tf_combine

#####################################

class Command(object):
    def control(self, dt=0):
        raise NotImplementedError()

    def apply(self, state, **kwargs):
        raise NotImplementedError()

    def iterate(self):
        raise NotImplementedError()


class Commands(object):
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

#####################################

class Trajectory(Command):
    _draw = False
    def __init__(self, path):
        self.path = tuple(path)

    def apply(self, state, sample=1):
        for conf in self.path[::sample]:
            conf.assign()
            yield

        end_conf = self.path[-1]
        if isinstance(end_conf, Pose):
            state.poses[end_conf.body] = end_conf

    def control(self, dt=0, **kwargs):
        controller = joint_controller()
        for conf in self.path:
            if isinstance(conf, Pose):
                conf = conf.to_base_conf()

            for _ in controller.execute(conf.body, conf.joints, conf.values):
                step_simulation()
                time.sleep(dt)

    def to_points(self, link=None):
        points = []
        for conf in self.path:
            conf.assign()
            point = np.array(get_group_conf(conf.body, 'base'))
            point[2] = 0
            point += 1e-2*np.array([0, 0, 1])
            if not (points and np.allclose(points[-1], point, atol=1e-3, rtol=0)):
                points.append(point)

        points = get_target_path(self)
        return waypoints_from_path(points)

    def distance(self, distance_fn=get_distance):
        total = 0.
        for q1, q2 in zip(self.path, self.path[1:]):
            total += distance_fn(q1.values, q2.values)

        return total

    def iterate(self):
        for conf in self.path:
            yield conf

    def reverse(self):
        return Trajectory(reversed(self.path))

    def __repr__(self):
        d = 0
        if self.path:
            conf = self.path[0]
            d = 3 if isinstance(conf, Pose) else len(conf.joints)

        return 't({},{})'.format(d, len(self.path))


class GripperCommand(Command):
    def __init__(self, robot, arm, position, teleport=False):
        self.robot = robot
        self.arm = arm
        self.position = position
        self.teleport = teleport

    def apply(self, state, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        start_conf = get_joint_positions(self.robot, joints)
        end_conf = [self.position] * len(joints)
        if self.teleport:
            path = [start_conf, end_conf]
        else:
            extend_fn = get_extend_fn(self.robot, joints)
            path = [start_conf] + list(extend_fn(start_conf, end_conf))

        for positions in path:
            set_joint_positions(self.robot, joints, positions)
            yield positions

    def control(self, **kwargs):
        controller = joint_controller()
        for _ in controller.execute(self.robot, joints, positions):
            yield

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, self.position)


class Attach(Command):
    vacuum = True
    def __init__(self, robot, arm, grasp, body):
        self.robot = robot
        self.arm = arm
        self.grasp = grasp
        self.body = body
        self.link = link_from_name(self.robot, get_tool_frame(self.robot))

    def assign(self):
        gripper_pose = get_link_pose(self.robot, self.link)
        body_pose = tf_combine(gripper_pose, self.grasp.value)
        set_pose(self.body, body_pose)

    def apply(self, state, **kwargs):
        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        state.grasps[self.body] = self.grasp
        del state.poses[self.body]
        yield

    def control(self, dt=0, **kwargs):
        controller = joint_controller()
        controller.attach_objects_to_robot()

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_body_name(self.body))


class Detach(Command):
    def __init__(self, robot, arm, body):
        self.robot = robot
        self.arm = arm
        self.body = body
        self.link = link_from_name(self.robot,  get_tool_frame(self.robot))

    def apply(self, state, **kwargs):
        del state.attachments[self.body]
        state.poses[self.body] = Pose(self.body, get_pose(self.body))
        del state.grasps[self.body]
        yield

    def control(self, motion_gen):
        controller = joint_controller(self.robots)
        controller.detach_object_from_robot()

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_body_name(self.body))

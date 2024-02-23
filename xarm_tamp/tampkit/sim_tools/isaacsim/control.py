import time
import numpy as np
from geometry import (
    Pose, 
)
from isaacsim.sim_utils import (
    # Getter
    get_distance, get_group_conf, get_target_path, get_gripper_joints,
    get_joint_positions, get_extend_fn, get_body_name, get_link_pose,
    get_min_limit, get_name, get_pose,
    # Setter
    set_joint_positions, set_pose, 
    # Utils
    add_segments, step_simulation, remove_debug, joint_controller_hold,
    waypoints_from_path, link_from_name, create_attachment, add_fixed_constraint,
    joints_from_names, remove_fixed_constraint
)


class Command(object):
    def control(self, dt=0):
        raise NotImplementedError()

    def apply(self, state, **kwargs):
        raise NotImplementedError()

    def iterate(self):
        raise NotImplementedError()


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
        for conf in self.path:
            if isinstance(conf, Pose):
                conf = conf.to_base_conf()

            for _ in joint_controller_hold(conf.body, conf.joints, conf.values):
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
        joints = get_gripper_joints(self.robot, self.arm)
        positions = [self.position]*len(joints)
        control_mode = p.TORQUE_CONTROL
        for _ in joint_controller_hold(self.robot, joints, control_mode, positions):
            yield

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, self.position)

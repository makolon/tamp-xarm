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
        handles = add_segments(self.to_points()) if self._draw and has_gui() else []
        for conf in self.path[::sample]:
            conf.assign()
            yield
        end_conf = self.path[-1]
        if isinstance(end_conf, Pose):
            state.poses[end_conf.body] = end_conf
        for handle in handles:
            remove_debug(handle)

    def control(self, dt=0, **kwargs):
        for conf in self.path:
            if isinstance(conf, Pose):
                conf = conf.to_base_conf()
            for _ in joint_controller_hold(conf.body, conf.joints, conf.values):
                step_simulation()
                time.sleep(dt)

    def to_points(self, link=BASE_LINK):
        points = []
        for conf in self.path:
            with BodySaver(conf.body):
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


class Attach(Command):
    vacuum = True
    def __init__(self, robot, arm, grasp, body):
        self.robot = robot
        self.arm = arm
        self.grasp = grasp
        self.body = body
        self.link = link_from_name(self.robot, HSR_TOOL_FRAMES.get(self.arm, self.arm))

    def assign(self):
        gripper_pose = get_link_pose(self.robot, self.link)
        body_pose = multiply(gripper_pose, self.grasp.value)
        set_pose(self.body, body_pose)

    def apply(self, state, **kwargs):
        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        state.grasps[self.body] = self.grasp
        del state.poses[self.body]
        yield

    def control(self, dt=0, **kwargs):
        if self.vacuum:
            add_fixed_constraint(self.body, self.robot, self.link)
        else:
            gripper_name = '{}_gripper'.format(self.arm)
            joints = joints_from_names(self.robot, HSR_GROUPS[gripper_name])
            values = [get_min_limit(self.robot, joint) for joint in joints]
            for _ in joint_controller_hold(self.robot, joints, values):
                step_simulation()
                time.sleep(dt)

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))


class Detach(Command):
    def __init__(self, robot, arm, body):
        self.robot = robot
        self.arm = arm
        self.body = body
        self.link = link_from_name(self.robot, HSR_TOOL_FRAMES.get(self.arm, self.arm))

    def apply(self, state, **kwargs):
        del state.attachments[self.body]
        state.poses[self.body] = Pose(self.body, get_pose(self.body))
        del state.grasps[self.body]
        yield

    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))
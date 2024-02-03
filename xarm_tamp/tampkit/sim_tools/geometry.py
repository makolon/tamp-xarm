
class Pose(object):
    num = count()
    def __init__(self, body, value=None, support=None, extra=None, init=False):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.support = support
        self.extra= extra
        self.init = init
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
        return 'p{}'.format(index)
    

class Conf(object):
    def __init__(self, body, joints, values=None, init=False):
        self.body = body
        self.joints = joints
        if values is None:
            values = get_joint_positions(self.body, self.joints)
        self.values = tuple(values)
        self.init = init

    @property
    def bodies(self):
        return flatten_links(self.body, get_moving_links(self.body, self.joints))

    def assign(self):
        set_joint_positions(self.body, self.joints, self.values)

    def iterate(self):
        yield self

    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)


class State(object):
    def __init__(self, attachments={}, cleaned=set(), cooked=set()):
        self.poses = {body: Pose(body, get_pose(body))
                      for body in get_bodies() if body not in attachments}
        self.grasps = {}
        self.attachments = attachments
        self.cleaned = cleaned
        self.cooked = cooked

    def assign(self):
        for attachment in self.attachments.values():
            attachment.assign()
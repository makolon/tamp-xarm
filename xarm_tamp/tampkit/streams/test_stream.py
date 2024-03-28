from tampkit.sim_tools.sim_utils import (
    pairwise_collision,
    iterate_approach_path,
    is_placement,
    is_insertion,
)


def get_cfree_pose_pose_test(collisions=True, **kwargs):
    def test(b1, p1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p1.assign()
        p2.assign()
        return not pairwise_collision(b1, b2, **kwargs)
    return test

def get_cfree_approach_pose_test(problem, collisions=True):
    gripper = problem.get_gripper()
    def test(b1, p1, g1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p2.assign()
        for _ in iterate_approach_path(problem.robot, gripper, p1, g1, body=b1):
            if pairwise_collision(b1, b2) or pairwise_collision(gripper, b2):
                return False
        return True
    return test

def get_cfree_traj_pose_test(problem, collisions=True):
    def test(c, b2, p2):
        if not collisions:
            return True
        state = c.assign()
        if b2 in state.attachments:
            return True
        p2.assign()
        for _ in c.apply(state):
            state.assign()
            for b1 in state.attachments:
                if pairwise_collision(b1, b2):
                    return False
            if pairwise_collision(problem.robot, b2):
                return False
        return True
    return test

def get_supported(problem, collisions=True):
    def test(b, p1, r, p2):
        return is_placement(b, r)
    return test

def get_inserted(problem, collisions=True):
    def test(b, p1, r, p2):
        return is_insertion(b, r)
    return test

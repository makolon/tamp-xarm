import random
import numpy as np
from xarm_tamp.tampkit.sim_tools.primitives import BodyPose
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    get_center_extent,
    get_point,
    get_pose,
    multiply,
    pairwise_collision,
    set_pose,
    unit_point,
    unit_pose
)

CIRCULAR_LIMITS = (10.0, 10.0)

def sample_insertion(body, hole, max_attempts=25, **kwargs):
    body_pose = get_pose(body)
    for _ in range(max_attempts):
        # rotate body
        rotation = np.array([0., 0., np.random.uniform(*CIRCULAR_LIMITS)])
        set_pose(body, multiply([unit_point(), rotation], unit_pose()))

        # get center position and w, d, h
        center, extent = get_center_extent(body)
        hole_pose = get_point(hole)
        x, y, z = hole_pose
        point = np.array([x, y, z]) + (get_point(body) - center)
        pose = multiply([point, rotation], unit_pose())

        # reset body pose
        set_pose(body, body_pose)
        return pose
    return None


def get_insert_gen(problem, collisions=True, **kwargs):
    # Sample insert pose
    obstacles = problem.fixed if collisions else []
    def gen_fn(body, hole):
        if hole is None:
            holes = problem.holes
        else:
            holes = [hole]

        while True:
            hole = random.choice(holes)
            pose = sample_insertion(body, hole, **kwargs)
            if (pose is None) or any(pairwise_collision(body, b) for b in obstacles):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)
    return gen_fn

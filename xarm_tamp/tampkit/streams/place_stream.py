import random
import numpy as np
from xarm_tamp.tampkit.sim_tools.primitives import BodyPose
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    aabb_empty,
    get_aabb,
    get_center_extent,
    pairwise_collision,
    sample_aabb,
    unit_quat
)

CIRCULAR_LIMITS = (10.0, 10.0)

def sample_placement(top_body, bottom_body, max_attempts=25, **kwargs):
    bottom_aabb = get_aabb(bottom_body)
    for _ in range(max_attempts):
        # get center position and w, d, h
        center, extent = get_center_extent(top_body)
        lower = (np.array(bottom_aabb[0]) + extent/2)[:2]
        upper = (np.array(bottom_aabb[1]) - extent/2)[:2]
        aabb = (lower, upper)
        if aabb_empty(aabb):
            continue

        # sample place pose
        x, y = sample_aabb(aabb)
        z = (bottom_aabb[1] + extent/2.)[2] + 0.01 # TODO: fix
        point = np.array([x, y, z])
        pose = [point, unit_quat()]
        return pose
    return None

def get_place_gen(problem, collisions=True, **kwargs):
    # Sample place pose
    obstacles = problem.fixed if collisions else []
    def gen_fn(body, surface):
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]

        while True:
            surface = random.choice(surfaces)
            pose = sample_placement(body, surface, **kwargs)
            if (pose is None) or any(pairwise_collision(body, b) for b in obstacles):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)
    return gen_fn
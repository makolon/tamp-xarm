import random
import numpy as np
from tampkit.sim_tools.isaacsim.primitives import BodyPose
from tampkit.sim_tools.isaacsim.sim_utils import (
    aabb_empty,
    get_aabb,
    get_center_extent,
    get_point,
    get_pose,
    multiply,
    pairwise_collision,
    set_pose,
    sample_aabb,
    unit_point
)

CIRCULAR_LIMITS = (10.0, 10.0)

def sample_placement(top_body, bottom_body, bottom_link=None, max_attempts=25, **kwargs):
    bottom_aabb = get_aabb(bottom_body, link=bottom_link)
    top_pose = get_pose(top_body)
    for _ in range(max_attempts):
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        rotation = np.array([0., 0., theta])
        set_pose(top_body, multiply([unit_point(), rotation], top_pose))
        center, extent = get_center_extent(top_body)
        lower = (np.array(bottom_aabb[0]))[:2]
        upper = (np.array(bottom_aabb[1]))[:2]
        aabb = (lower, upper)
        if aabb_empty(aabb):
            continue
        x, y = sample_aabb(aabb)
        z = (bottom_aabb[1])[2]
        point = np.array([x, y, z]) + (get_point(top_body) - center)
        pose = multiply([point, rotation], top_pose)
        set_pose(top_body, pose)
    return pose


def get_place_gen(problem, collisions=True, **kwargs):
    # Sample place pose
    obstacles = problem.fixed if collisions else []
    def gen_fn(body, surface):
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]

        while True:
            surface = random.choise(surfaces)
            pose = sample_placement(body, surface, **kwargs)
            if (pose is None) or any(pairwise_collision(body, b) for b in obstacles):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)
    return gen_fn
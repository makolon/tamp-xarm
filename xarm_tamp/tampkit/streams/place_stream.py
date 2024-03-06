import random
import numpy as np
from tampkit.sim_tools.isaacsim.geometry import Pose
from tampkit.sim_tools.isaacsim.sim_utils import pairwise_collision


def sample_placement(top_body, bottom_body, bottom_link=None, **kwargs):
    bottom_aabb = get_aabb(bottom_body, link=bottom_link)
    for _ in range(max_attempts):
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        rotation = Euler(yaw=theta)
        set_pose(top_body, multiply(Pose(euler=rotation), top_pose))
        center, extent = get_center_extent(top_body)
        lower = (np.array(bottom_aabb[0]))[:2] # - percent*extent/2)[:2]
        upper = (np.array(bottom_aabb[1]))[:2] # + percent*extent/2)[:2]
        aabb = AABB(lower, upper)
        if aabb_empty(aabb):
            continue
        x, y = sample_aabb(aabb)
        z = (bottom_aabb[1])[2] # + extent/2.)[2] + epsilon
        point = np.array([x, y, z]) + (get_point(top_body) - center)
        pose = multiply(Pose(point, rotation), top_pose)
        set_pose(top_body, pose)
    return pose


def get_stable_gen(problem, collisions=True, **kwargs):
    # Sample place pose
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    def gen_fn(body, surface):
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]

        while True:
            surface = random.choise(surfaces)
            body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            p.assign()

            # If the obstacles is not included in the surface and body, check pairwise collision between body.
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)
    return gen_fn
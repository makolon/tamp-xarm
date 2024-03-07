import random
import numpy as np
from tampkit.sim_tools.isaacsim.geometry import Pose
from tampkit.sim_tools.isaacsim.sim_utils import (
    pairwise_collision,
    set_pose,
    multiply,
    unit_pose,
    get_point,
    get_center_extent,   
)

CIRCULAR_LIMITS = (10.0, 10.0)
Euler = None # Pose()

def sample_insertion(body, hole, max_attempts=25, percent=1.0, epsilon=1e-3, **kwargs):
    for _ in range(max_attempts):
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        rotation = Euler(yaw=theta)
        set_pose(body, multiply(Pose(euler=rotation), unit_pose()))
        center, extent = get_center_extent(body)
        hole_pose = get_point(hole)
        x, y, z = hole_pose
        point = np.array([x, y, z]) + (get_point(body) - center)
        pose = multiply(Pose(point, rotation), unit_pose())
        set_pose(body, pose)
        return pose
    return None

def get_insert_gen(problem, collisions=True, **kwargs):
    # Sample insert pose
    robot = problem.robot
    obstacles = problem.fixed if collisions else []

    def gen_fn(body, hole):
        if hole is None:
            holes = problem.holes
        else:
            holes = [hole]

        while True:
            hole = random.choise(holes)
            body_pose = sample_insertion(body, hole, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, hole)
            p.assign()

            # If the obstacles is not included in the surface and body, check pairwise collision between body.
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, hole}):
                yield (p,)
    return gen_fn

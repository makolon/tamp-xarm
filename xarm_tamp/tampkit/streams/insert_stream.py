import random
from tampkit.sim_tools.isaacsim.geometry import Pose
from tampkit.sim_tools.isaacsim.sim_utils import pairwise_collision


def sample_insertion(body, hole, **kwargs):
    pass


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

import random
from tampkit.sim_tools.isaacsim.geometry import Pose
from tampkit.sim_tools.isaacsim.sim_utils import pairwise_collision


def sample_placement(body, surface, **kwargs):
    pass


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
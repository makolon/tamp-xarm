import numpy as np
from xarm_tamp.tampkit.sim_tools.primitives import BodyGrasp
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    approximate_as_prism,
    get_pose,
    get_tool_pose,
    get_tool_link,
    multiply,
    pairwise_collision,
    unit_quat,
)


def sample_grasps(body, tool_pose, body_pose, grasp_length=0.0, max_width=0.1):
    # TODO: Modified to allow rotation of the fingertip in the z-axis direction.
    center, (width, depth, height) = approximate_as_prism(body, get_pose(body))

    if width > max_width:
        rotation = [0.0, 0.0, np.pi / 2]
    elif depth > max_width:
        rotation = unit_quat()
    else:
        rotation = unit_quat()

    pose_diff = [tool_pose[0] - body_pose[0], rotation]

    under = 0
    for _ in range(1 + under):
        grasps = multiply(tool_pose, pose_diff)
        return grasps

def get_grasp_gen(problem, collisions=True):
    robot = problem.robot
    tool_link = get_tool_link(robot, robot._end_effector_prim_name)
    obstacles = problem.fixed if collisions else []
    def gen_fn(body):
        tool_pose = get_tool_pose(robot)
        body_pose = get_pose(body)
        while True:
            grasp_pose = sample_grasps(body, tool_pose, body_pose)
            if (len(grasp_pose) == 0) or any(pairwise_collision(body, b) for b in obstacles):
                continue
            body_grasp = BodyGrasp(robot, tool_link, body, grasp_pose)
            yield (body_grasp,)
    return gen_fn
import hydra
import random
import numpy as np
from collections import namedtuple
from omegaconf import DictConfig

# Initialize isaac sim
import xarm_tamp.tampkit.sim_tools.sim_utils

from xarm_tamp.tampkit.sim_tools.primitives import BodyPose, BodyConf, Command
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    # Simulation utility
    connect, disconnect,
    # Getter
    get_pose, get_arm_joints, get_joint_positions,
)

from xarm_tamp.tampkit.problems import PROBLEMS
from xarm_tamp.tampkit.streams.inverse_kinematics_stream import get_ik_fn
from xarm_tamp.tampkit.streams.plan_motion_stream import get_motion_fn
from xarm_tamp.tampkit.streams.grasp_stream import get_grasp_gen
from xarm_tamp.tampkit.streams.place_stream import get_place_gen
from xarm_tamp.tampkit.streams.test_stream import get_cfree_pose_pose_test, get_cfree_approach_pose_test, \
    get_cfree_traj_pose_test, get_supported, get_inserted

# PDDLStream functions
from pddlstream.algorithms.meta import solve
from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.language.constants import print_solution, AND, PDDLProblem
from pddlstream.language.stream import StreamInfo
from pddlstream.utils import get_file_path, read, str_from_object, Profiler, INF

BASE_CONSTANT = 1
BASE_VELOCITY = 0.5

#######################################################

CustomValue = namedtuple('CustomValue', ['stream', 'values'])

def move_cost_fn(c):
    return 1

def opt_grasp_fn(o):
    p2 = CustomValue('p-sg', (o,))
    return p2,

def opt_place_fn(o1, o2):
    p2 = CustomValue('p-sp', (o2,))
    return p2,

def opt_insert_fn(o1, o2):
    p2 = CustomValue('p-si', (o2,))
    return p2,

def opt_motion_fn(q1, q2):
    t = CustomValue('t-pbm', (q1, q2))
    return t,

#######################################################

class TAMPPlanner(object):
    def __init__(self, task, algorithm, unit, deterministic,
                    problem, cfree, teleport, simulate):
        self._task = task
        self._algorithm = algorithm
        self._unit = unit
        self._deterministic = deterministic
        self._problem = problem
        self._cfree = cfree
        self._teleport = teleport
        self._simulate = simulate

        np.set_printoptions(precision=2)
        if deterministic:
            random.seed(0)
            np.random.seed(seed=0)

    def pddlstream_from_problem(self, problem, collisions=True, teleport=False):
        robot = problem.robot

        domain_pddl = read(get_file_path(__file__, f'problems/{self._task}/pddl/domain.pddl'))
        stream_pddl = read(get_file_path(__file__, f'problems/{self._task}/pddl/stream.pddl'))
        constant_map = {}

        # Initlaize init & goal
        init, goal = [AND], [AND]

        # Robot
        joints = get_arm_joints(robot)
        conf = BodyConf(robot, joints, get_joint_positions(robot, joints))
        init += [('CanMove',),
                 ('HandEmpty',),
                 ('Conf', conf),
                 ('AtConf', conf),]
        goal += [('AtConf', conf)]

        # Body
        for body in problem.movable:
            pose = BodyPose(body, get_pose(body))
            init += [('Graspable', body),
                     ('Pose', body, pose),
                     ('AtPose', body, pose)]

        # Surface
        for body in problem.surfaces:
            pose = BodyPose(body, get_pose(body))
            init += [('Region', body)]

        # Hole
        for body in problem.holes:
            pose = BodyPose(body, get_pose(body))
            init += [('Hole', body)]

        init += [('Placeable', b1, b2) for b1, b2 in problem.init_placeable]
        init += [('Insertable', b1, b2) for b1, b2 in problem.init_insertable]
        goal += [('Holding', a, b) for a, b in problem.goal_holding] + \
                [('On', a, b) for a, b in problem.goal_placed] + \
                [('InHole', a, b) for a, b in problem.goal_inserted] + \
                [('Cleaned', b)  for b in problem.goal_cleaned] + \
                [('Cooked', b)  for b in problem.goal_cooked]

        stream_map = {
            # Constrained sampler
            'sample-grasp': from_gen_fn(get_grasp_gen(problem, collisions=collisions)),
            'sample-place': from_gen_fn(get_place_gen(problem, collisions=collisions)),
            # Inverse kinematics
            'inverse-kinematics': from_fn(get_ik_fn(problem, collisions=collisions)),
            # Planner
            'plan-motion': from_fn(get_motion_fn(problem, collisions=collisions, teleport=teleport)),
            # Test function
            'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(problem, collisions=collisions)),
            'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(problem, collisions=collisions)),
            'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(problem, collisions=collisions)),
            'test-supported': from_test(get_supported(problem, collisions=collisions)),
            'test-inserted': from_test(get_inserted(problem, collisions=collisions)),
        }

        return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    def post_process(self, problem, plan, teleport=False):
        if plan is None:
            return None

        paths = []
        for i, (name, args) in enumerate(plan):
            if name == 'move':
                q1, q2, c = args
                new_commands = c
            elif name == 'pick':
                o, p, g, q, c = args
                new_commands = c
            elif name == 'place':
                o, p, g, q, c = args
                new_commands = c
            elif name == 'insert':
                o, p, g, q, c = args
                new_commands = c
            else:
                raise ValueError(name)
            print(i, name, args, new_commands)
            paths += [new_commands]
        return Command(paths)

    def execute(self, sim_cfg, curobo_cfg):
        simulation_app = connect()

        # Instanciate problem 
        problem_from_name = {fn.__name__: fn for fn in PROBLEMS}
        if self._problem not in problem_from_name:
            raise ValueError(self._problem)
        print('Problem:', self._problem)
        problem_fn = problem_from_name[self._problem]
        tamp_problem = problem_fn(sim_cfg, curobo_cfg)

        pddlstream_problem = self.pddlstream_from_problem(tamp_problem, collisions=not self._cfree, teleport=self._teleport)

        stream_info = {
            'sample-grasp': StreamInfo(opt_gen_fn=from_fn(opt_grasp_fn)),
            'sample-place': StreamInfo(opt_gen_fn=from_fn(opt_place_fn)),
            'sample-insert': StreamInfo(opt_gen_fn=from_fn(opt_insert_fn)),
            'plan-motion': StreamInfo(opt_gen_fn=from_fn(opt_motion_fn)),
        }

        _, _, _, stream_map, init, goal = pddlstream_problem
        print('Init:', init)
        print('Goal:', goal)
        print('Streams:', str_from_object(set(stream_map)))

        with Profiler():
            solution = solve(pddlstream_problem, algorithm=self._algorithm, unit_costs=self._unit,
                            stream_info=stream_info, success_cost=INF, verbose=True, debug=False)

        print_solution(solution)
        plan, cost, evaluations = solution

        print('#############################')
        print('plan: ', plan)
        print('#############################')

        if (plan is None) or not simulation_app.is_running():
            return

        # Post process
        command = self.post_process(tamp_problem, plan)

        # Execute commands
        if self._simulate:
            command.control()
        else:
            command.execute()

        # Close simulator
        disconnect()


config_file = input("Please input the problem name from (simple_fetch, simple_stacking, fmb_momo, fmb_simo): ")
@hydra.main(version_base=None, config_name=config_file, config_path="./configs")
def main(cfg: DictConfig):
    tamp_planer = TAMPPlanner(
        task=cfg.pddlstream.task,
        algorithm=cfg.pddlstream.algorithm,
        unit=cfg.pddlstream.unit,
        deterministic=cfg.pddlstream.deterministic,
        problem=cfg.pddlstream.problem,
        cfree=cfg.pddlstream.cfree,
        teleport=cfg.pddlstream.teleport,
        simulate=cfg.pddlstream.simulate,
    )
    tamp_planer.execute(cfg.sim, cfg.curobo)


if __name__ == '__main__':
    main()
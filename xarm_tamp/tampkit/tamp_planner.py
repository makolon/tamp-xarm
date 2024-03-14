import hydra
import random
import numpy as np
from collections import namedtuple
from omegaconf import DictConfig

# Initialize isaac sim
import tampkit.sim_tools.isaacsim.sim_utils

from tampkit.sim_tools.isaacsim.sim_utils import (
    # Simulation utility
    connect, disconnect, \
    # Getter
    get_pose, get_max_limit, get_arm_joints, get_gripper_joints, \
    get_joint_positions, \
    # Utility
    apply_commands, control_commands
)
from tampkit.sim_tools.isaacsim.geometry import (
    Pose, Conf, State
)
from tampkit.sim_tools.isaacsim.control import (
    GripperCommand, Attach, Detach
)
from tampkit.problems import PROBLEMS
from tampkit.streams.plan_base_stream import plan_base_fn
from tampkit.streams.plan_arm_stream import plan_arm_fn
from tampkit.streams.grasp_stream import get_grasp_gen
from tampkit.streams.place_stream import get_place_gen
from tampkit.streams.insert_stream import get_insert_gen
from tampkit.streams.test_stream import get_cfree_pose_pose_test, get_cfree_approach_pose_test, \
    get_cfree_traj_pose_test, get_supported, get_inserted

# PDDLStream functions
from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test
from pddlstream.language.constants import print_solution, Equal, AND, PDDLProblem
from pddlstream.language.external import defer_shared, never_defer
from pddlstream.language.function import FunctionInfo
from pddlstream.language.stream import StreamInfo
from pddlstream.language.object import SharedOptValue
from pddlstream.utils import get_file_path, read, str_from_object, Profiler, INF

BASE_CONSTANT = 1
BASE_VELOCITY = 0.5

#######################################################

CustomValue = namedtuple('CustomValue', ['stream', 'values'])

def move_cost_fn(c):
    return 1

def opt_grasp_fn(o, r):
    p2 = CustomValue('p-sg', (r,))
    return p2,

def opt_place_fn(o, r):
    p2 = CustomValue('p-sp', (r,))
    return p2,

def opt_insert_fn(o, r):
    p2 = CustomValue('p-si', (r,))
    return p2,

def opt_motion_arm_fn(a, o, p, g):
    q = CustomValue('q-ik', (p,))
    t = CustomValue('t-ik', tuple())
    return q, t

def opt_motion_base_fn(q1, q2):
    t = CustomValue('t-pbm', (q1, q2))
    return t,

#######################################################

class TAMPPlanner(object):
    def __init__(self, algorithm, unit, deterministic, problem, cfree, teleport):
        self._algorithm = algorithm
        self._unit = unit
        self._deterministic = deterministic
        self._problem = problem
        self._cfree = cfree
        self._teleport = teleport

        np.set_printoptions(precision=2)
        if deterministic:
            random.seed(0)
            np.random.seed(seed=0)

    def pddlstream_from_problem(self, problem, collisions=True, teleport=False):
        robot = problem.robot

        domain_pddl = read(get_file_path(__file__, 'problems/assemble/domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'problems/assemble/stream.pddl'))
        constant_map = {}

        # Initlaize init & goal
        init, goal = [AND], [AND]

        init += [
            ('CanMove',),
            Equal(('PickCost',), 1),
            Equal(('PlaceCost',), 1),
            Equal(('InsertCost',), 1),
        ]

        joints = get_arm_joints(robot)
        conf = Conf(robot, joints, get_joint_positions(robot, joints))
        init += [('Arm', 'arm'), ('Conf', 'arm', conf), ('HandEmpty', 'arm'), ('AtConf', 'arm', conf)]
        init += [('Controllable', 'arm')]

        for body in problem.movable:
            pose = Pose(body, get_pose(body))
            init += [('Graspable', body),
                     ('Pose', body, pose),
                     ('AtPose', body, pose)]

        # Surface pose
        for body in problem.surfaces:
            pose = Pose(body, get_pose(body))
            init += [('RegionPose', body, pose)]

        # Hole pose
        for body in problem.holes:
            pose = Pose(body, get_pose(body))
            init += [('HolePose', body, pose)]

        init += [('Inserted', b1) for b1 in problem.holes]
        init += [('Placeable', b1, b2) for b1, b2 in problem.init_placeable]
        init += [('Insertable', b1, b2) for b1, b2 in problem.init_insertable]
        goal += [('Holding', a, b) for a, b in problem.goal_holding] + \
                [('On', a, b) for a, b in problem.goal_on] + \
                [('InHole', a, b) for a, b in problem.goal_inserted] + \
                [('Cleaned', b)  for b in problem.goal_cleaned] + \
                [('Cooked', b)  for b in problem.goal_cooked]

        stream_map = {
            # Constrained sampler
            'sample-grasp': from_gen_fn(get_grasp_gen(problem, collisions=False)),
            'sample-place': from_gen_fn(get_place_gen(problem, collisions=collisions)),
            'sample-insert': from_gen_fn(get_insert_gen(problem, collisions=collisions)),
            # Planner
            'plan-base-motion': from_fn(plan_base_fn(problem, collisions=True, teleport=teleport)),
            'plan-arm-motion': from_fn(plan_arm_fn(problem, collisions=collisions, teleport=teleport)),
            # Test function
            'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(collisions=collisions)),
            'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(problem, collisions=collisions)),
            'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(problem, collisions=collisions)),
            'test-supported': from_test(get_supported(problem, collisions=collisions)),
            'test-inserted': from_test(get_inserted(problem, collisions=collisions)),
        }

        return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    def post_process(self, problem, plan, teleport=False):
        if plan is None:
            return None
        commands = []
        for i, (name, args) in enumerate(plan):
            if name == 'move':
                q1, q2, c = args
                new_commands = c.commands
            elif name == 'pick':
                a, b, p, g, _, c = args
                [traj_pick] = c.commands
                close_gripper = GripperCommand(problem.robot, a, g.grasp_width, teleport=teleport)
                attach = Attach(problem.robot, a, g, b)
                new_commands = [traj_pick, close_gripper, attach, traj_pick.reverse()]
            elif name == 'place':
                a, b1, b2, p, g, _, c = args
                [traj_place] = c.commands
                gripper_joint = get_gripper_joints(problem.robot, a)[0]
                position = get_max_limit(problem.robot, gripper_joint)
                new_commands = [traj_place,]
            elif name == 'insert':
                a, b1, b2, p1, p2, g, _, _, c = args
                [traj_insert, traj_depart, traj_return] = c.commands
                gripper_joint = get_gripper_joints(problem.robot, a)[0]
                position = get_max_limit(problem.robot, gripper_joint)
                open_gripper = GripperCommand(problem.robot, a, position, teleport=teleport)
                detach = Detach(problem.robot, a, b1)
                new_commands = [traj_insert, detach, open_gripper, traj_depart, traj_return.reverse()]
            else:
                raise ValueError(name)
            print(i, name, args, new_commands)
            commands += new_commands
        return commands

    def execute(self, sim_cfg):
        simulation_app = connect()

        # Instanciate problem 
        problem_from_name = {fn.__name__: fn for fn in PROBLEMS}
        if self._problem not in problem_from_name:
            raise ValueError(self._problem)
        print('Problem:', self._problem)
        problem_fn = problem_from_name[self._problem]
        tamp_problem = problem_fn(sim_cfg)

        pddlstream_problem = self.pddlstream_from_problem(tamp_problem, collisions=not self._cfree, teleport=self._teleport)

        stream_info = {
            'sample-grasp': StreamInfo(opt_gen_fn=from_fn(opt_grasp_fn)),
            'sample-place': StreamInfo(opt_gen_fn=from_fn(opt_place_fn)),
            'sample-insert': StreamInfo(opt_gen_fn=from_fn(opt_insert_fn)),
            'plan-arm_motion': StreamInfo(opt_gen_fn=from_fn(opt_motion_arm_fn)),
            'plan-base-motion': StreamInfo(opt_gen_fn=from_fn(opt_motion_base_fn)),
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
        commands = self.post_process(tamp_problem, plan)

        # Execute commands
        if sim_cfg.simulate:
            control_commands(State(), commands, time_step=0.03)
        else:
            apply_commands(State(), commands, time_step=0.03)

        # Close simulator
        disconnect()


@hydra.main(version_base=None, config_name="config", config_path="./configs")
def main(cfg: DictConfig):
    tamp_planer = TAMPPlanner(
        algorithm=cfg.pddlstream.algorithm,
        unit=cfg.pddlstream.unit,
        deterministic=cfg.pddlstream.deterministic,
        problem=cfg.pddlstream.problem,
        cfree=cfg.pddlstream.cfree,
        teleport=cfg.pddlstream.teleport,
    )
    tamp_planer.execute(cfg.sim, cfg.curobo)


if __name__ == '__main__':
    main()
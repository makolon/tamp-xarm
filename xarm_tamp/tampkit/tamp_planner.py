import hydra
import random
import numpy as np
from collections import namedtuple
from omegaconf import DictConfig

# Initialize isaac sim
import xarm_tamp.tampkit.sim_tools.sim_utils

from xarm_tamp.tampkit.sim_tools.primitives import (
    BodyPose, BodyConf, ArmCommand, GripperCommand
)
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    # Simulation utility
    connect, disconnect, step_simulation, apply_action,
    create_grasp_action, create_trajectory, target_reached,
    # Getter
    get_pose, get_arm_joints, get_joint_positions, get_gripper_joints
)

from xarm_tamp.tampkit.problems import PROBLEMS
from xarm_tamp.tampkit.streams.inverse_kinematics_stream import get_ik_fn
from xarm_tamp.tampkit.streams.plan_motion_stream import get_free_motion_fn, get_holding_motion_fn
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

def opt_free_motion_fn(q1, q2):
    t = CustomValue('t-pbm', (q1, q2))
    return t,

def opt_holding_motion_fn(q1, q2, o, g):
    t = CustomValue('t-hm', (q1, q2, o, g))
    return t,

#######################################################

class TAMPPlanner(object):
    def __init__(self, task, algorithm, unit, deterministic,
                    problem, cfree, teleport, simulate, attach):
        self._task = task
        self._algorithm = algorithm
        self._unit = unit
        self._deterministic = deterministic
        self._problem = problem
        self._cfree = cfree
        self._teleport = teleport
        self._simulate = simulate
        self._attach = attach

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
            'plan-free-motion': from_fn(get_free_motion_fn(problem, collisions=collisions)),
            'plan-holding-motion': from_fn(get_holding_motion_fn(problem, collisions=collisions)),
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

        commands = []
        for i, (name, args) in enumerate(plan):
            new_commands = []
            if name == 'move_free':
                q1, q2, c = args
                move_trajectory = create_trajectory(c.path, c.joints)
                new_commands += [ArmCommand(name, c.robot, move_trajectory)]
            elif name == 'move_holding':
                q1, q2, o, g, c = args
                move_trajectory = create_trajectory(c.path, c.joints)
                new_commands += [ArmCommand(name, c.robot, move_trajectory)]
            elif name == 'pick':
                o, p, g, q, c = args
                grasp_action = create_grasp_action([35.0 * np.pi / 180, -35.0 * np.pi / 180], get_gripper_joints(c.robot)) # create_grasp_action([10.0, 10.0], get_gripper_joints(c.robot))
                new_commands += [GripperCommand('grasp', o, c.robot, grasp_action)]
                return_trajectory = create_trajectory(c.reverse().path, c.joints)
                new_commands += [ArmCommand(name, c.robot, return_trajectory)]
            elif name == 'place':
                o, p, g, q, c = args
                release_action = create_grasp_action([0.0, 0.0], get_gripper_joints(c.robot))
                new_commands += [GripperCommand('release', o, c.robot, release_action)]
                return_trajectory = create_trajectory(c.reverse().path, c.joints)
                new_commands += [ArmCommand(name, c.robot, return_trajectory)]
            elif name == 'insert':
                o, p, g, q, c = args
                release_action = create_grasp_action([0.0, 0.0], get_gripper_joints(c.robot))
                new_commands += [GripperCommand('release', c.robot, release_action)]
                return_trajectory = create_trajectory(c.reverse().path, c.joints)
                new_commands += [ArmCommand(name, c.robot, return_trajectory)]
            else:
                raise ValueError(name)
            print(i, name, args, new_commands)
            commands += new_commands
        return commands

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
            'plan-free-motion': StreamInfo(opt_gen_fn=from_fn(opt_free_motion_fn)),
            'plan-holding-motion': StreamInfo(opt_gen_fn=from_fn(opt_holding_motion_fn))
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
        print('commands:', commands)
        input('wait_for_user')

        # Execute commands
        for step, command in enumerate(commands):
            print('step.{}: {} action'.format(step, command.name))
            for path in command.path:
                if command.name in ['grasp', 'release']:
                    for _ in range(50):
                        apply_action(command.robot, path)
                        # command.robot.set_joint_positions(torch.tensor(path.joint_positions, device='cuda'), joint_indices=get_gripper_joints(command.robot))
                        step_simulation(tamp_problem.world)
                    continue

                while not target_reached(command.robot, path):
                    apply_action(command.robot, path)
                    step_simulation(tamp_problem.world)

        # Close simulator
        disconnect()


config_file = input("Please input the problem name from (simple_fetch, simple_stacking, fmb_momo, siemense_gearbox, peg_in_hole, block_world): ")
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
        attach=cfg.pddlstream.attach,
    )
    tamp_planer.execute(cfg.sim, cfg.curobo)


if __name__ == '__main__':
    main()
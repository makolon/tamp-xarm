import hydra
from xarm_tamp.tampkit.sim_tools.curobo_utils import *


@hydra.main(version_base=None, config_name="assembly_config", config_path="../configs")
def main():
    tensor_args = get_tensor_device_type()
    print('tensor_args')

    ### Config Test
    robot_cfg = get_robot_cfg(curobo_cfg)
    print('robot_cfg:', robot_cfg)

    world_cfg = get_world_cfg(curobo_cfg)
    print('world_cfg:', world_cfg)

    plan_cfg = get_motion_gen_plan_cfg(curobo_cfg)
    print('plan_cfg:', plan_cfg)

    robot_world_cfg = get_robot_world_cfg(curobo_cfg, world_cfg)
    print('robot_world_cfg:', robot_world_cfg)

    world_collision_cfg = get_world_collision_cfg(curobo_cfg, world_cfg, tensor_args)
    print('world_collision_cfg:', world_collision_cfg)

    ik_cfg = get_ik_solver_cfg(curobo_cfg, robot_cfg, world_cfg, tensor_args)
    print('ik_cfg:', ik_cfg)

    motion_gen_cfg = get_motion_gen_cfg(curobo_cfg, robot_cfg, world_cfg, tensor_args)
    print('motion_gen_cfg:', motion_gen_cfg)

    mpc_cfg = get_mpc_solver_cfg(curobo_cfg, robot_cfg, world_cfg)
    print('mpc_cfg:', mpc_cfg)

    ### cuRobo Modules Test
    robot_world = get_robot_world(robot_world_cfg)
    print('robot_world:', robot_world)

    collision_checker = get_collision_checker(world_collision_cfg)
    print('collision_checker:', collision_checker)

    ik_solver = get_ik_solver(ik_cfg)
    print('ik_solver:', ik_solver)

    motion_gen = get_motion_gen(motion_gen_cfg)
    print('motion_gen:', motion_gen)

    mpc_solver = get_mpc_solver(mpc_cfg)
    print('mpc_solver:', mpc_solver)
    


if __name__ == "__main__":
    main()
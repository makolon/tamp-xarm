from xarm_tamp.tampkit.sim_tools.sim_utils import (
    # Creater
    create_world, create_floor, create_robot,
    create_table, create_block, create_distant_light,
    # Setter
    set_initial_conf,
)
from xarm_tamp.tampkit.sim_tools.curobo_utils import (
    # Config
    get_robot_cfg,
    get_world_cfg,
    get_robot_world_cfg,
    get_world_collision_cfg,
    get_motion_gen_plan_cfg,
    get_ik_solver_cfg,
    get_motion_gen_cfg,
    get_mpc_solver_cfg,
    get_tensor_device_type,
    get_usd_helper,
    get_motion_gen,
    get_robot_world,
    get_collision_checker,
    get_ik_solver,
    get_mpc_solver,
)
from xarm_tamp.tampkit.problems.base_problem import Problem


def stacking_problem(sim_cfg, curobo_cfg):
    world = create_world(sim_params=sim_cfg)
    stage = world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    ########################

    # create light
    create_distant_light()

    # create plane
    create_floor(world, sim_cfg.floor)

    # create robot
    xarm = create_robot(sim_cfg.robot)
    world.scene.add(xarm)

    # create table
    table = create_table(sim_cfg.table)
    world.scene.add(table)

    # set momo parts
    block1 = create_block(sim_cfg.block1.name,
                          sim_cfg.block1.translation,
                          sim_cfg.block1.orientation)
    world.scene.add(block1)

    block2 = create_block(sim_cfg.block2.name,
                          sim_cfg.block2.translation,
                          sim_cfg.block2.orientation)
    world.scene.add(block2)

    block3 = create_block(sim_cfg.block3.name,
                          sim_cfg.block3.translation,
                          sim_cfg.block3.orientation)
    world.scene.add(block3)

    block4 = create_block(sim_cfg.block4.name,
                          sim_cfg.block4.translation,
                          sim_cfg.block4.orientation)
    world.scene.add(block4)

    # reset world
    world.reset()

    # initialize world
    initial_conf = sim_cfg.robot.initial_configuration
    set_initial_conf(xarm, initial_conf, use_gripper=True)
    world.step(render=True)

    # play world
    world.play()

    ########################

    # define robot_cfg
    robot_cfg = get_robot_cfg(curobo_cfg.robot_cfg)

    # define world_cfg
    world_cfg = get_world_cfg(curobo_cfg.world_cfg)

    # define plan_cfg
    plan_cfg = get_motion_gen_plan_cfg(curobo_cfg.motion_generation_plan_cfg)

    ########################
    
    # define tensor_args
    tensor_args = get_tensor_device_type()

    ########################

    # usd helper
    usd_helper = get_usd_helper()
    usd_helper.load_stage(stage)

    ########################

    # define world model
    robot_world_cfg = get_robot_world_cfg(
        cfg=curobo_cfg,
        world_cfg=world_cfg,
    )
    robot_world = get_robot_world(robot_world_cfg)

    ########################

    # define collision checker
    world_collision_cfg = get_world_collision_cfg(
        cfg=curobo_cfg,
        world_cfg=world_cfg,
        tensor_args=tensor_args,
    )
    world_collision = get_collision_checker(world_collision_cfg)

    ########################

    # define inverse kinematics
    ik_solver_cfg = get_ik_solver_cfg(
        cfg=curobo_cfg,
        robot_cfg=robot_cfg,
        world_cfg=world_cfg,
        tensor_args=tensor_args,
    )
    ik_solver = get_ik_solver(ik_solver_cfg)

    # define model predictive controller
    mpc_cfg = get_mpc_solver_cfg(
        cfg=curobo_cfg,
        robot_cfg=robot_cfg,
        world_cfg=world_cfg,
    )
    mpc = get_mpc_solver(mpc_cfg)

    # define motion planner
    motion_gen_cfg = get_motion_gen_cfg(
        cfg=curobo_cfg,
        robot_cfg=robot_cfg,
        world_cfg=world_cfg,
        tensor_args=tensor_args,
    )
    motion_gen = get_motion_gen(motion_gen_cfg)
    print('warming up...')
    motion_gen.warmup(enable_graph=curobo_cfg.motion_gen_cfg.enable_graph,
                      warmup_js_trajopt=curobo_cfg.motion_gen_cfg.warmup_js_trajopt,
                      parallel_finetune=curobo_cfg.motion_gen_cfg.parallel_finetune)
    print('cuRobo is Ready!')

    ########################

    return Problem(
        # PDDL
        world=world,
        robot=xarm,
        movable=[block1, block2, block3, block4],
        fixed=[table],
        surfaces=[table, block1, block2, block3],
        bodies=[table, block1, block2, block3, block4],
        init_placeable=[(block2, block1), (block4, block3)],
        goal_placed=[(block2, block1), (block4, block3)],
        # Config
        robot_cfg=robot_cfg,
        world_cfg=world_cfg,
        plan_cfg=plan_cfg,
        # Tensor args
        tensor_args=tensor_args,
        # Usd helper
        usd_helper=usd_helper,
        # World
        robot_world=robot_world,
        # Collision
        world_collision=world_collision,
        # Planner
        ik_solver=ik_solver,
        motion_planner=motion_gen,
        mpc=mpc,
    )
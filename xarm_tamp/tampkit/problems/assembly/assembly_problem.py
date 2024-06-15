from xarm_tamp.tampkit.sim_tools.sim_utils import (
    # Creater
    create_world, create_floor, create_robot,
    create_table, create_fmb, create_surface,
    create_shaft, create_gearbox_base, create_distant_light,
    create_fixed_block, create_block,
    # Utils
    set_initial_conf, apply_physics_settings
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


def fmb_momo_problem(sim_cfg, curobo_cfg):
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

    # create base plate
    base_block = create_fmb(sim_cfg.base_block)

    # set momo parts
    block1 = create_fmb(sim_cfg.block1)
    block2 = create_fmb(sim_cfg.block2)
    block3 = create_fmb(sim_cfg.block3)
    block4 = create_fmb(sim_cfg.block4)
    apply_physics_settings(block1, sim_cfg.block1)
    apply_physics_settings(block2, sim_cfg.block2)
    apply_physics_settings(block3, sim_cfg.block3)
    apply_physics_settings(block4, sim_cfg.block4)

    # define surfaces
    surf1 = create_surface(sim_cfg.surface1.name,
                           sim_cfg.surface1.translation,
                           sim_cfg.surface1.orientation)
    world.scene.add(surf1)

    surf2 = create_surface(sim_cfg.surface2.name,
                           sim_cfg.surface2.translation,
                           sim_cfg.surface2.orientation)
    world.scene.add(surf2)

    surf3 = create_surface(sim_cfg.surface3.name,
                           sim_cfg.surface3.translation,
                           sim_cfg.surface3.orientation)
    world.scene.add(surf3)

    surf4 = create_surface(sim_cfg.surface4.name,
                           sim_cfg.surface4.translation,
                           sim_cfg.surface4.orientation)
    world.scene.add(surf4)

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
        surfaces=[surf1, surf2, surf3, surf4],
        bodies=[table, base_block, block1, block2, block3, block4],
        init_placeable=[(block1, surf1), (block2, surf2)],
        goal_placed=[(block1, surf1), (block2, surf2)],
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

def siemense_gearbox_problem(sim_cfg, curobo_cfg):
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

    # create base plate
    base_block = create_gearbox_base(sim_cfg.base_block)
    world.scene.add(base_block)

    # set momo parts
    shaft1 = create_shaft(sim_cfg.shaft1)
    shaft2 = create_shaft(sim_cfg.shaft2)
    apply_physics_settings(shaft1, sim_cfg.shaft1)
    apply_physics_settings(shaft2, sim_cfg.shaft2)

    # define surfaces
    surf1 = create_surface(
        sim_cfg.surface1.name,
        sim_cfg.surface1.translation,
        sim_cfg.surface1.orientation
    )
    world.scene.add(surf1)

    surf2 = create_surface(
        sim_cfg.surface2.name,
        sim_cfg.surface2.translation,
        sim_cfg.surface2.orientation
    )
    world.scene.add(surf2)

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
        movable=[shaft1, shaft2],
        fixed=[table],
        surfaces=[surf1, surf2],
        bodies=[table, base_block, shaft1, shaft2],
        init_placeable=[(shaft1, surf1), (shaft2, surf2)],
        goal_placed=[(shaft1, surf1), (shaft2, surf2)],
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

def peg_in_hole_problem(sim_cfg, curobo_cfg):
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

    # create base plate
    fixed_block1 = create_fixed_block(sim_cfg.fixed_block1.name,
                                      sim_cfg.fixed_block1.scale,
                                      sim_cfg.fixed_block1.translation,
                                      sim_cfg.fixed_block1.orientation)
    world.scene.add(fixed_block1)

    fixed_block2 = create_fixed_block(sim_cfg.fixed_block2.name,
                                      sim_cfg.fixed_block2.scale,
                                      sim_cfg.fixed_block2.translation,
                                      sim_cfg.fixed_block2.orientation)
    world.scene.add(fixed_block2)

    fixed_block3 = create_fixed_block(sim_cfg.fixed_block3.name,
                                      sim_cfg.fixed_block3.scale,
                                      sim_cfg.fixed_block3.translation,
                                      sim_cfg.fixed_block3.orientation)
    world.scene.add(fixed_block3)

    fixed_block4 = create_fixed_block(sim_cfg.fixed_block4.name,
                                      sim_cfg.fixed_block4.scale,
                                      sim_cfg.fixed_block4.translation,
                                      sim_cfg.fixed_block4.orientation)
    world.scene.add(fixed_block4)

    # set peg parts
    block = create_block(sim_cfg.block.name,
                         sim_cfg.block.translation,
                         sim_cfg.block.orientation)
    world.scene.add(block)

    # define surfaces
    surface = create_surface(
        sim_cfg.surface.name,
        sim_cfg.surface.translation,
        sim_cfg.surface.orientation
    )
    world.scene.add(surface)

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
        movable=[block],
        fixed=[table],
        surfaces=[surface],
        bodies=[table, block],
        init_placeable=[(block, surface)],
        goal_placed=[(block, surface)],
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

def block_world_problem(sim_cfg, curobo_cfg):
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

    # create base plate
    fixed_block1 = create_fixed_block(sim_cfg.fixed_block1.name,
                                      sim_cfg.fixed_block1.scale,
                                      sim_cfg.fixed_block1.translation,
                                      sim_cfg.fixed_block1.orientation)
    world.scene.add(fixed_block1)

    fixed_block2 = create_fixed_block(sim_cfg.fixed_block2.name,
                                      sim_cfg.fixed_block2.scale,
                                      sim_cfg.fixed_block2.translation,
                                      sim_cfg.fixed_block2.orientation)
    world.scene.add(fixed_block2)

    fixed_block3 = create_fixed_block(sim_cfg.fixed_block3.name,
                                      sim_cfg.fixed_block3.scale,
                                      sim_cfg.fixed_block3.translation,
                                      sim_cfg.fixed_block3.orientation)
    world.scene.add(fixed_block3)

    fixed_block4 = create_fixed_block(sim_cfg.fixed_block4.name,
                                      sim_cfg.fixed_block4.scale,
                                      sim_cfg.fixed_block4.translation,
                                      sim_cfg.fixed_block4.orientation)
    world.scene.add(fixed_block4)

    fixed_block5 = create_fixed_block(sim_cfg.fixed_block5.name,
                                      sim_cfg.fixed_block5.scale,
                                      sim_cfg.fixed_block5.translation,
                                      sim_cfg.fixed_block5.orientation)
    world.scene.add(fixed_block5)

    fixed_block6 = create_fixed_block(sim_cfg.fixed_block6.name,
                                      sim_cfg.fixed_block6.scale,
                                      sim_cfg.fixed_block6.translation,
                                      sim_cfg.fixed_block6.orientation)
    world.scene.add(fixed_block6)

    # set peg parts
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

    # define surfaces
    surface1 = create_surface(
        sim_cfg.surface1.name,
        sim_cfg.surface1.translation,
        sim_cfg.surface1.orientation
    )
    world.scene.add(surface1)

    surface2 = create_surface(
        sim_cfg.surface2.name,
        sim_cfg.surface2.translation,
        sim_cfg.surface2.orientation
    )
    world.scene.add(surface2)

    surface3 = create_surface(
        sim_cfg.surface3.name,
        sim_cfg.surface3.translation,
        sim_cfg.surface3.orientation
    )
    world.scene.add(surface3)

    surface4 = create_surface(
        sim_cfg.surface4.name,
        sim_cfg.surface4.translation,
        sim_cfg.surface4.orientation
    )
    world.scene.add(surface4)

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
        surfaces=[surface1, surface2, surface3, surface4],
        bodies=[table, block1, block2, block3, block4],
        init_placeable=[(block1, surface1), (block2, surface2),
            (block3, surface3), (block4, surface4)],
        goal_placed=[(block1, surface1), (block2, surface2),
            (block3, surface3), (block4, surface4)],
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
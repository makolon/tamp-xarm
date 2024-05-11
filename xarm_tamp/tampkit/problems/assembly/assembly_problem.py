from xarm_tamp.tampkit.sim_tools.sim_utils import (
    # Creater
    create_world, create_floor, create_robot,
    create_table, create_fmb, create_surface,
    create_hole,
    # Getter
    get_initial_conf, get_pose,
    # Setter
    set_pose, set_initial_conf,
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
    get_motion_gen,
    get_robot_world,
    get_collision_checker,
    get_ik_solver,
    get_mpc_solver,
)
from xarm_tamp.tampkit.problems.base_problem import Problem


def fmb_momo_problem(sim_cfg, curobo_cfg):
    world = create_world()
    stage = world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    ########################

    # create plane
    plane = create_floor(world, sim_cfg.floor)

    # create robot
    xarm = create_robot(sim_cfg.robot)
    initial_conf = get_initial_conf(xarm)
    set_initial_conf(xarm, initial_conf)
    world.scene.add(xarm)

    # create table
    table = create_table(sim_cfg.table)
    set_pose(table, (sim_cfg.table.translation, sim_cfg.table.orientation))
    world.scene.add(table)

    # create base plate
    base_block = create_fmb(sim_cfg.base_block)
    set_pose(base_block, (sim_cfg.base_block.translation, sim_cfg.base_block.orientation))
    world.scene.add(base_block)

    # set momo parts
    block1 = create_fmb(sim_cfg.block1)
    set_pose(block1, (sim_cfg.block1.translation, sim_cfg.block1.orientation))
    world.scene.add(block1)

    block2 = create_fmb(sim_cfg.block2)
    set_pose(block2, (sim_cfg.block2.translation, sim_cfg.block2.orientation))
    world.scene.add(block2)

    block3 = create_fmb(sim_cfg.block3)
    set_pose(block3, (sim_cfg.block3.translation, sim_cfg.block3.orientation))
    world.scene.add(block3)

    block4 = create_fmb(sim_cfg.block4)
    set_pose(block4, (sim_cfg.block4.translation, sim_cfg.block4.orientation))
    world.scene.add(block4)

    # define surfaces
    block1_pose = get_pose(block1)
    surf1 = create_surface(sim_cfg.surface1.name, *block1_pose)
    surf1_pose = calc_surf_pose(block1_pose, "surface1")
    set_pose(surf1, surf1_pose)
    world.scene.add(surf1)

    block2_pose = get_pose(block2)
    surf2 = create_surface(sim_cfg.surface2.name, *block2_pose)
    surf2_pose = calc_surf_pose(block2_pose, "surface2")
    set_pose(surf2, surf2_pose)
    world.scene.add(surf2)

    block3_pose = get_pose(block3)
    surf3 = create_surface(sim_cfg.surface3.name, *block3_pose)
    surf3_pose = calc_surf_pose(block3_pose, "surface3")
    set_pose(surf3, surf3_pose)
    world.scene.add(surf3)

    block4_pose = get_pose(block4)
    surf4 = create_surface(sim_cfg.surface4.name, *block4_pose)
    surf4_pose = calc_surf_pose(block4_pose, "surface4")
    set_pose(surf4, surf4_pose)
    world.scene.add(surf4)

    # define holes
    hole1 = create_hole(sim_cfg.hole1.name, *block1_pose)
    hole1_pose = calc_hole_pose(block1_pose, "hole1")
    set_pose(hole1, hole1_pose)
    world.scene.add(hole1)

    hole2 = create_hole(sim_cfg.hole2.name, *block2_pose)
    hole2_pose = calc_hole_pose(block2_pose, "hole2")
    set_pose(hole2, hole2_pose)
    world.scene.add(hole2)

    hole3 = create_hole(sim_cfg.hole3.name, *block3_pose)
    hole3_pose = calc_hole_pose(block3_pose, "hole3")
    set_pose(hole3, hole3_pose)
    world.scene.add(hole3)

    hole4 = create_hole(sim_cfg.hole4.name, *block4_pose)
    hole4_pose = calc_hole_pose(block4_pose, "hole4")
    set_pose(hole4, hole4_pose)
    world.scene.add(hole4)

    # reset world
    world.reset()

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
        robot=xarm,
        movable=[block1, block2, block3, block4],
        fixed=[table, base_block],
        surfaces=[surf1, surf2, surf3, surf4],
        holes=[hole1, hole2, hole3, hole4],
        bodies=[table, base_block, block1, block2, block3, block4],
        init_placeable=[
            (block1, surf1), (block2, surf2),
            (block3, surf3), (block4, surf4)],
        init_insertable=[
            (block1, hole1), (block2, hole2),
            (block3, hole3), (block4, hole4)],
        goal_inserted=[
            (block1, hole1), (block2, hole2),
            (block3, hole3), (block4, hole4)],
        # Config
        robot_cfg=robot_cfg,
        world_cfg=world_cfg,
        plan_cfg=plan_cfg,
        # Tensor args
        tensor_args=tensor_args,
        # World
        robot_world=robot_world,
        # Collision
        world_collision=world_collision,
        # Planner
        ik_solver=ik_solver,
        motion_planner=motion_gen,
        mpc=mpc,
    )


def fmb_simo_problem(sim_cfg):
    world = create_world()
    stage = world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    ########################

    # create plane
    plane = create_floor(world, sim_cfg.floor)

    # create robot
    xarm = create_robot(sim_cfg.robot)
    initial_conf = get_initial_conf(xarm)
    set_initial_conf(xarm, initial_conf)

    # create table
    table = create_table(sim_cfg.table)
    set_pose(table, (sim_cfg.table.translation, sim_cfg.table.orientation))

    # create base plate
    base_block = create_fmb(sim_cfg.base_block)
    set_pose(base_block, (sim_cfg.base_block.translation, sim_cfg.base_block.orientation))

    # set sim parts
    blocks = []
    for i in range(sim_cfg.num_blocks):
        block_cfg = sim_cfg[f"block{i}"]
        block = create_fmb(block_cfg)
        set_pose(block, (block_cfg.translation, block_cfg.orientation))
        blocks.append(block)

    return Problem(
        robot=xarm,
        movable=blocks,
        surfaces=[table, base_block],
        init_insertable=[],
        init_placeable=[],
        goal_on=[],
        goal_inserted=[],
    )

# TODO: add function   
def calc_surf_pose(block_pose, name):
    return block_pose

# TODO: add function
def calc_hole_pose(block_pose, name):
    return block_pose
from tampkit.sim_tools.isaacsim.sim_utils import (
    # Creater
    create_world, create_floor, create_robot,
    create_table, create_fmb, create_surface,
    create_hole,
    # Getter
    get_initial_conf, get_pose,
    # Setter
    set_pose, set_arm_conf,
)
from tampkit.sim_tools.isaacsim.curobo_utils import (
    get_robot_cfg,
    get_world_cfg,
    get_tensor_device_type,
    get_motion_gen,
    get_robot_world,
    get_ik_solver,
    get_mpc_solver,
)
from base_problem import Problem


def fmb_momo_problem(sim_cfg):
    world = create_world()
    stage = world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/fmb_momo", "Xform")

    ########################

    # create plane
    plane = create_floor(world, sim_cfg.floor)

    # create robot
    xarm = create_robot(sim_cfg.robot)
    initial_conf = get_initial_conf(xarm)
    set_arm_conf(xarm, initial_conf)

    # create table
    table = create_table(sim_cfg.table)
    set_pose(table, sim_cfg.table.translation, sim_cfg.table.orientation)

    # create base plate
    base_block = create_fmb(sim_cfg.base_block)
    set_pose(base_block, sim_cfg.base_block.translation, sim_cfg.base_block.orientation)

    # set momo parts
    block1 = create_fmb(sim_cfg.block1)
    set_pose(block1, sim_cfg.block1.translation, sim_cfg.block1.orientation)

    block2 = create_fmb(sim_cfg.block2)
    set_pose(block2, sim_cfg.block2.translation, sim_cfg.block2.orientation)

    block3 = create_fmb(sim_cfg.block3)
    set_pose(block3, sim_cfg.block3.translation, sim_cfg.block3.orientation)

    block4 = create_fmb(sim_cfg.block4)
    set_pose(block4, sim_cfg.block4.translation, sim_cfg.block4.orientation)
    
    # define surfaces
    # TODO: add function to calculate surface position
    block1_pose = get_pose(block1)
    surf1 = create_surface(sim_cfg.surface1.name, *block1_pose)
    surf1_pose = calc_surf_pose(block1_pose,"surface1")
    set_pose(surf1, surf1_pose)
    
    block2_pose = get_pose(block2)
    surf2 = create_surface(sim_cfg.surface2.name, *block2_pose)
    surf2_pose = calc_surf_pose(block2_pose, "surface2")
    set_pose(surf2, surf2_pose)
    
    block3_pose = get_pose(block3)
    surf3 = create_surface(sim_cfg.surface3.name, *block3_pose)
    surf3_pose = calc_surf_pose(block3_pose, "surface3")
    set_pose(surf3, surf3_pose)
    
    block4_pose = get_pose(block4)
    surf4 = create_surface(sim_cfg.surface4.name, *block4_pose)
    surf4_pose = calc_surf_pose(block4_pose, "surface4")
    set_pose(surf4, surf4_pose)
 
    # define holes
    # TODO: add function to calculate hole position
    hole1 = create_hole(sim_cfg.hole1.name, *block1_pose)
    hole1_pose = calc_hole_pose(block1_pose, "hole1")
    set_pose(hole1, hole1_pose)

    hole2 = create_hole(sim_cfg.hole2.name, *block2_pose)
    hole2_pose = calc_hole_pose(block2_pose, "hole2")
    set_pose(hole2, hole2_pose)

    hole3 = create_hole(sim_cfg.hole3.name, *block3_pose)
    hole3_pose = calc_hole_pose(block3_pose, "hole3")
    set_pose(hole3, hole3_pose)

    hole4 = create_hole(sim_cfg.hole4.name, *block4_pose)
    hole4_pose = calc_hole_pose(block4_pose, "hole4")
    set_pose(hole4, hole4_pose)

    ########################

    # define robot_cfg
    robot_cfg = get_robot_cfg(sim_cfg)

    # define world_config
    world_cfg = get_world_cfg(world_cfg)

    ########################
    
    # define tensor_args
    tensor_args = get_tensor_device_type()
    
    ########################

    # define world model
    robot_world = get_robot_world()
    
    # define inverse kinematics
    ik_solver = get_ik_solver()

    # define model predictive controller
    mpc = get_mpc_solver()

    # define motion planner
    motion_gen = get_motion_gen()
    print('warming up...')
    motion_gen.warmup(enable_graph=sim_cfg.motion_generation.enable_graph,
                      warmup_js_trajopt=sim_cfg.motion_generation.warmup_js_trajopt,
                      parallel_finetune=sim_cfg.motion_generation.parallel_finetune)
    print('cuRobo is Ready!')

    ########################

    return Problem(
        # Instance
        robot=xarm,
        arms=['arm'],
        movable=[block1, block2, block3, block4],
        fixed=[table, base_block],
        surfaces=[surf1, surf2, surf3, surf4], # surfaces=[table, base_block],
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
        # Tensor args
        tensor_args=tensor_args,
        # World
        robot_world=robot_world,
        # Planner
        ik_solver=ik_solver,
        motion_planner=motion_gen,
        mpc=mpc
    )

def fmb_simo_problem(sim_cfg):
    world = create_world()
    stage = world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/fmb_momo", "Xform")

    # create plane
    plane = create_floor(world, sim_cfg.floor)

    # create robot
    xarm = create_robot(sim_cfg.robot)
    initial_conf = get_initial_conf(xarm)
    set_arm_conf(xarm, initial_conf)

    # create table
    table = create_table(sim_cfg.table)
    set_pose(table, sim_cfg.table.translation, sim_cfg.table.orientation)

    # create base plate
    base_block = create_fmb(sim_cfg.base_block)
    set_pose(base_block, sim_cfg.base_block.translation, sim_cfg.base_block.orientation)

    # set sim parts
    blocks = []
    for i in range(sim_cfg.num_blocks):
        block_cfg = sim_cfg[f"block{i}"]
        block = create_fmb(block_cfg)
        set_pose(block, block_cfg.translation, block_cfg.orientation)
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
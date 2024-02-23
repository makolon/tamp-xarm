import dataclasses
from sim_tools.isaacsim.sim_utils import (
    # Creater
    create_world, create_floor, create_robot,
    create_table, create_fmb,
    # Getter
    get_initial_conf,
    # Setter
    set_point, set_arm_conf,
)
from base_problem import Problem


def fmb_momo_problem(sim_cfg):
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
    set_point(table, sim_cfg.table.translation, sim_cfg.table.orientation)

    # create base plate
    base_block = create_fmb(sim_cfg.base_block)
    set_point(base_block, sim_cfg.base_block.translation, sim_cfg.base_block.orientation)

    # set momo parts
    block1 = create_fmb(sim_cfg.block1)
    set_point(block1, sim_cfg.block1.translation, sim_cfg.block1.orientation)

    block2 = create_fmb(sim_cfg.block2)
    set_point(block2, sim_cfg.block2.translation, sim_cfg.block2.orientation)

    block3 = create_fmb(sim_cfg.block3)
    set_point(block3, sim_cfg.block3.translation, sim_cfg.block3.orientation)

    block4 = create_fmb(sim_cfg.block4)
    set_point(block4, sim_cfg.block4.translation, sim_cfg.block4.orientation)

    return Problem(
        robot=xarm,
        movable=[block1, block2, block3, block4],
        surfaces=[table, base_block],
        init_insertable=[],
        init_placeable=[],
        goal_on=[],
        goal_inserted=[],
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
    set_point(table, sim_cfg.table.translation, sim_cfg.table.orientation)

    # create base plate
    base_block = create_fmb(sim_cfg.base_block)
    set_point(base_block, sim_cfg.base_block.translation, sim_cfg.base_block.orientation)

    # set sim parts
    blocks = []
    for i in range(sim_cfg.num_blocks):
        block_cfg = sim_cfg[f"block{i}"]
        block = create_fmb(block_cfg)
        set_point(block, block_cfg.translation, block_cfg.orientation)
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
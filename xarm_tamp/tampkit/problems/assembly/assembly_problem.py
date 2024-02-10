import dataclasses
from sim_tools.isaacsim import (
    # Creater
    create_world, create_floor, create_xarm,
    create_table, create_momo_block,
    # Getter
    get_initial_conf,
    # Setter
    set_point, set_arm_conf,
)
from base_problem import Problem


def create_fmb_momo_problem():
    world = create_world()
    stage = world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/fmb_momo", "Xform")

    # create plane
    plane = create_floor()

    # create robot
    xarm = create_xarm()
    initial_conf = get_initial_conf(xarm)
    set_arm_conf(xarm, initial_conf)

    # create table
    table = create_table(table_width, table_depth, table_height)
    set_point(table, table_translation, table_orientation)

    # set momo parts
    block1 = create_momo_block('block1')
    set_point(block1, )

    return Problem(

    )
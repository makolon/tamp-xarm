

def create_fmb_momo_problem():
    plane = create_floor(fixed_base=True)

    # set robot
    xarm = create_xarm()
    initial_conf = get_initial_conf(xarm)
    set_arm_conf(xarm, initial_conf)

    # set table
    table = create_table(table_width, table_depth, table_height)
    set_point(table, table_translation, table_orientation)

    # set momo parts
    block1 = create_momo_block('block1')
    set_point(block1, )

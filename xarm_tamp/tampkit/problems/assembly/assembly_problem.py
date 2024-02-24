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
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)


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

    # define robot_cfg
    robot_cfg = load_yaml(join_path(sim_cfg.robot_cfg_path, sim_cfg.robot)["robot_cfg"])

    # create table
    table = create_table(sim_cfg.table)
    set_point(table, sim_cfg.table.translation, sim_cfg.table.orientation)
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )

    # create base plate
    base_block = create_fmb(sim_cfg.base_block)
    set_point(base_block, sim_cfg.base_block.translation, sim_cfg.base_block.orientation)
    world_cfg_base = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_base.yml"))
    )

    # set momo parts
    block1 = create_fmb(sim_cfg.block1)
    set_point(block1, sim_cfg.block1.translation, sim_cfg.block1.orientation)
    world_cfg_block1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_block1.yml"))
    )

    block2 = create_fmb(sim_cfg.block2)
    set_point(block2, sim_cfg.block2.translation, sim_cfg.block2.orientation)
    world_cfg_block2 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_block2.yml"))
    )

    block3 = create_fmb(sim_cfg.block3)
    set_point(block3, sim_cfg.block3.translation, sim_cfg.block3.orientation)
    world_cfg_block3 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_block3.yml"))
    )

    block4 = create_fmb(sim_cfg.block4)
    set_point(block4, sim_cfg.block4.translation, sim_cfg.block4.orientation)
    world_cfg_block4= WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_block4.yml"))
    )

    # define world_config
    world_cfg = WorldConfig(
        cuboid=world_cfg_table.cuboid,
        mesh=[
            world_cfg_base.mesh,
            world_cfg_block1.mesh,
            world_cfg_block2.mesh,
            world_cfg_block3.mesh,
            world_cfg_block4.mesh,
        ],
    )

    # define tensor_args
    tensor_args = TensorDeviceType()

    # define motion plan config
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=sim_cfg.motion_generation.num_trajopt_seeds,
        num_graph_seeds=sim_cfg.motion_generation.num_graph_seeds,
        interpolation_dt=sim_cfg.motion_generation.interpolation_dt,
        collision_cache={
            "obb": sim_cfg.motion_generation.n_obstacle_cuboids,
            "mesh": sim_cfg.motion_generation.n_obstacle_mesh},
        optimize_dt=sim_cfg.motion_generation.optimize_dt,
        trajopt_dt=sim_cfg.motion_generation.trajopt_dt,
        trajopt_tsteps=sim_cfg.motion_generation.trajopt_tsteps,
        trim_steps=sim_cfg.motion_generation.trim_steps,
    )

    # define motion planner
    motion_gen = MotionGen(motion_gen_cfg)
    print('warming up...')
    motion_gen.warmup(enable_graph=sim_cfg.motion_generation.enable_graph,
                      warmup_js_trajopt=sim_cfg.motion_generation.warmup_js_trajopt,
                      parallel_finetune=sim_cfg.motion_generation.parallel_finetune)
    print('cuRobo is Ready!')

    # define plan config
    plan_cfg = MotionGenPlanConfig(
        enable_graph=sim_cfg.motion_generation_plan.enable_graph,
        enable_graph_attempt=sim_cfg.motion_generation_plan.enable_graph_attempt,
        max_attempts=sim_cfg.motion_generation_plan.max_attempts,
        enable_finetune_trajopt=sim_cfg.motion_generation_plan.enable_finetune_trajopt,
        parallel_finetune=sim_cfg.motion_generation_plan.parallel_finetune,
    )

    return Problem(
        # Instance
        robot=xarm,
        movable=[block1, block2, block3, block4],
        surfaces=[table, base_block],
        init_insertable=[],
        init_placeable=[],
        goal_on=[],
        goal_inserted=[],
        # Config
        robot_cfg=robot_cfg,
        world_cfg=world_cfg,
        plan_cfg=plan_cfg,
        # Planner
        motion_planner=motion_gen,
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
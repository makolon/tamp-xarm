import dataclasses
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
from curobo.wrap.reacher.ik_solver import (
    IKSolver,
    IKSolverConfig,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from curobo.wrap.reacher.mpc import (
    MpcSolver,
    MpcSolverConfig,
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
    set_pose(table, sim_cfg.table.translation, sim_cfg.table.orientation)
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )

    # create base plate
    base_block = create_fmb(sim_cfg.base_block)
    set_pose(base_block, sim_cfg.base_block.translation, sim_cfg.base_block.orientation)
    world_cfg_base = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_base.yml"))
    )

    # set momo parts
    block1 = create_fmb(sim_cfg.block1)
    set_pose(block1, sim_cfg.block1.translation, sim_cfg.block1.orientation)
    world_cfg_block1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_block1.yml"))
    )

    block2 = create_fmb(sim_cfg.block2)
    set_pose(block2, sim_cfg.block2.translation, sim_cfg.block2.orientation)
    world_cfg_block2 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_block2.yml"))
    )

    block3 = create_fmb(sim_cfg.block3)
    set_pose(block3, sim_cfg.block3.translation, sim_cfg.block3.orientation)
    world_cfg_block3 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_block3.yml"))
    )

    block4 = create_fmb(sim_cfg.block4)
    set_pose(block4, sim_cfg.block4.translation, sim_cfg.block4.orientation)
    world_cfg_block4= WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_block4.yml"))
    )
    
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

    # define plan config
    plan_cfg = MotionGenPlanConfig(
        enable_graph=sim_cfg.motion_generation_plan.enable_graph,
        enable_graph_attempt=sim_cfg.motion_generation_plan.enable_graph_attempt,
        max_attempts=sim_cfg.motion_generation_plan.max_attempts,
        enable_finetune_trajopt=sim_cfg.motion_generation_plan.enable_finetune_trajopt,
        parallel_finetune=sim_cfg.motion_generation_plan.parallel_finetune,
    )

    # define tensor_args
    tensor_args = TensorDeviceType()

    # define inverse kinematics config
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        position_threshold=sim_cfg.inverse_kinematics.position_threshold,
        rotation_threshold=sim_cfg.inverse_kinematics.rotation_threshold,
        num_seeds=sim_cfg.inverse_kinematics.num_seeds,
        self_collision_check=sim_cfg.inverse_kinematics.self_collision_check,
        self_collision_opt=sim_cfg.inverse_kinematics.self_collision_opt,
        use_cuda_graph=sim_cfg.inverse_kinematics.use_cuda_graph,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={
            "obb": sim_cfg.inverse_kinematics.n_obstacle_cuboids,
            "mesh": sim_cfg.inverse_kinematics.n_obstacle_mesh},
    )
    
    # defailt inverse kinematics
    ik_solver = IKSolver(ik_config)

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

    # define model predictive controller config
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=sim_cfg.mpc.use_cuda_graph,
        use_cuda_graph_metrics=sim_cfg.mpc.use_cuda_graph_metrics,
        use_cuda_graph_full_step=sim_cfg.mpc.use_cuda_graph_full_step,
        self_collision_check=sim_cfg.mpc.self_collision_check,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={
            "obb": sim_cfg.mpc.n_obstackle_cuboids,
            "mesh": sim_cfg.mpc.n_obstackle_mesh,
        },
        use_mppi=sim_cfg.mpc.use_mppi,
        use_lbfgs=sim_cfg.mpc.use_lbfgs,
        store_rollouts=sim_cfg.mpc.store_rollouts,
        step_dt=sim_cfg.mpc.step_dt
    )
    
    # define model predictive controller
    mpc = MpcSolver(mpc_config)

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
        plan_cfg=plan_cfg,
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
    
    
def calc_surf_pose(block_pose, name):
    return block_pose
    
def calc_hole_pose(block_pose, name):
    return block_pose
import hydra
import torch
from omegaconf import DictConfig
from xarm_tamp.tampkit.sim_tools.sim_utils import *
from xarm_tamp.tampkit.sim_tools.curobo_utils import *
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.objects import sphere
from omni.isaac.debug_draw import _debug_draw
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.utils.logger import setup_curobo_logger
from curobo.utils.usd_helper import UsdHelper


### cuRobo Utils Test
def curobo_utils_example(curobo_cfg):
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


### Collision Checker Test
def collision_check_example(sim_cfg, curobo_cfg):
    # connect
    sim_app = connect()

    # create world
    world = create_world()
    floor = create_floor(world, sim_cfg.floor)

    # add usd_helper
    usd_help = UsdHelper()

    radius = 0.1
    target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))
    target = sphere.VisualSphere(
        "/World/target",
        position=np.array([0.5, 0.0, 1.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        radius=radius,
        visual_material=target_material
    )

    # set up logger
    setup_curobo_logger("warn")

    tensor_args = get_tensor_device_type()
    world_cfg = get_world_cfg(curobo_cfg.world_cfg)  # TODO: add

    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    robot_world = get_robot_world(curobo_cfg.robot_cfg)  # TODO: add
    ignore_list = ["/World/target", "/World/defaultGroundPlane"]

    i = 0
    x_sph = torch.zeros((1, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    x_sph[..., 3] = radius

    while sim_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation **** ")

        step_index = world.current_tim_step_index
        if step_index == 0:
            world.reset()

        if step_index % 1000 == 0:
            obstacles = usd_help.get_obstacles_from_stage(
                reference_prim_path="/World",
                ignore_substring=ignore_list,
            ).get_collision_check_world()
            robot_world.update_world(obstacles)

        sph_position, _ = target.get_local_pose()
        x_sph[..., :3] = tensor_args.to_device(sph_position).view(1, 1, 1, 3)

        d, d_vec = robot_world.get_collision_vector(x_sph)
        d = d.view(-1).cpu()

        p = d.item()
        p = max(1, p * 5)
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()

        if d.item() == 0.0:
            target_material.set_color(np.ravel([0, 1, 0]))
        elif d.item() <= robot_world.contact_distance:
            target_material.set_color(np.array([0, 0, p]))
        elif d.item() >= robot_world.contact_distance:
            target_material.set_color(np.array([p, 0, 0]))


### IK Test
def ik_example(sim_cfg, curobo_cfg):
    # connect
    sim_app = connect()

    # create world
    world = create_world()
    floor = create_floor(world, sim_cfg.floor)
    robot = create_robot(sim_cfg.robot)

    # add usd_helper
    usd_help = UsdHelper()

    # create target
    radius = 0.1
    target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))
    target = sphere.VisualSphere(
        "/World/target",
        position=np.array([0.5, 0.0, 1.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        radius=radius,
        visual_material=target_material
    )

    # set up logger
    setup_curobo_logger("warn")

    tensor_args = get_tensor_device_type()
    world_cfg = get_world_cfg(curobo_cfg.world_cfg)
    robot_cfg = get_robot_cfg()
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    # ik solver
    ik_solver = get_ik_solver()

    def get_pose_grid(n_x, n_y, n_z, max_x, max_y, max_z):
        x = np.linspace(-max_x, max_x, n_x)
        y = np.linspace(-max_y, max_y, n_y)
        z = np.linspace(0, max_z, n_z)
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        position_arr = np.zeros((n_x * n_y * n_z, 3))
        position_arr[:, 0] = x.flatten()
        position_arr[:, 1] = y.flatten()
        position_arr[:, 2] = z.flatten()
        return position_arr

    # warm up ik solver
    position_grid_offset = tensor_args.to_device(get_pose_grid(10, 10, 5, 0.5, 0.5, 0.5))
    fk_state = ik_solver.fk(ik_solver.get_regract_config().view(1, -1))
    goal_pose = fk_state.ee_pose
    goal_pose = goal_pose.repeat(position_grid_offset.shape[0])
    goal_pose.position += position_grid_offset
    result = ik_solver.solve_batch(goal_pose)

    print("Curobo is Ready")

    cmd_plan = None
    cmd_idx = 0
    i = 0
    spheres = None
    while sim_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation ****")
            i += 1
            continue

        step_index = world.current_time_step_index
        if step_index == 0:
            world.reset()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index % 500 == 0:
            print("Updating world, reading w.r.t.", robot.prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                reference_prim_path=robot.prim_path,
                ignore_substring=[
                    robot.prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            ik_solver.update_world(obstacles)
            print("Updated World")

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position

        sim_js = robot.get_joints_state()
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.veloities) * 0.0,
            joint_names=robot.dof_names,
        )

        cu_js = cu_js.get_ordered_joint_state(ik_solver.kinematics.joint_names)

        if step_index % 2 == 0:
            sph_list = ik_solver.kinematics.get_robot_as_spheres(cu_js.position)
            if spheres is None:
                spheres = []
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0.0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    spheres[si].set_world_pose(position=np.ravel(s.position))
                    spheres[si].set_radius(float(s.radius))

        if (
            np.linalg.norm(cube_position - target_pose) > 1e-3
            and np.linalg.norm(past_pose - cube_position) == 0.0
            and np.linalg.norm(sim_js.velocities) < 0.2
        ):
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation

            # compute curobo solution
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            goal_pose.position[:] = ik_goal.position[:] + position_grid_offset
            goal_pose.quaternion[:] = ik_goal.quaternion[:]
            result = ik_solver.solve_batch(goal_pose)
            
            succ = torch.any(result.success)
            print(
                "IK completed: Poses: "
                + str(goal_pose.batch)
                + " Time(s): "
                + str(result.solve_time)
            )

            if succ:
                cmd_plan = result.js_solution[result.success]
                idx_list = []
                common_js_names = []
                for x in robot.dof_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                cmd_idx = 0
            else:
                carb.log_warn("Plan did not converge to a solution. No action is being taken.")

        past_pose = cube_position
        if cmd_plan is not None and step_index % 20 == 0:
            cmd_state = cmd_plan[cmd_idx]
            robot.set_joint_positions(cmd_state.position.cpu().numpy(), idx_list)
            cmd_idx += 1
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                world.step(render=True)
                robot.set_joint_positions(default_config, idx_list)

    sim_app.close()


### Kinematics Test
def kinematics_example():
    tensor_args = get_tensor_device_type()

    robot_cfg = get_robot_cfg()
    kin_model = CudaRobotModel(robot_cfg.kinematics)

    # compute forward kinematics
    q = torch.rand((10, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
    out = kin_model.get_state(q)
    print('out:', out)


### Motion Planner Test
def motion_gen_example():
    # create a curobo motion gen instance:
    num_targets = 0
    # assuming obstacles are in objects_path:
    sim_app = connect()

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot = create_robot(sim_cfg.robot_cfg)

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.05
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)

    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=True,
        parallel_finetune=True,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    cmd_plan = None
    cmd_idx = 0
    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    past_cmd = None
    target_orientation = None
    past_orientation = None
    pose_metric = None
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index
        # print(step_index)
        if articulation_controller is None:
            # robot.initialize()
            articulation_controller = robot.get_articulation_controller()
        if step_index < 2:
            my_world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(len(obstacles.objects))

            motion_gen.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

        sim_js = robot.get_joints_state()
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )

        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0

        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # create spheres:

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))

        robot_static = False
        if (np.max(np.abs(sim_js.velocities)) < 0.2) or args.reactive:
            robot_static = True
        if (
            (
                np.linalg.norm(cube_position - target_pose) > 1e-3
                or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
            )
            and np.linalg.norm(past_pose - cube_position) == 0.0
            and np.linalg.norm(past_orientation - cube_orientation) == 0.0
            and robot_static
        ):
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation

            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            plan_config.pose_cost_metric = pose_metric
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()  # ik_result.success.item()
            if num_targets == 1:
                if args.constrain_grasp_approach:
                    pose_metric = PoseCostMetric.create_grasp_approach_metric()
                if args.reach_partial_pose is not None:
                    reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
                    pose_metric = PoseCostMetric(
                        reach_partial_pose=True, reach_vec_weight=reach_vec
                    )
                if args.hold_partial_pose is not None:
                    hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
                    pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
            if succ:
                num_targets += 1
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                # get only joint names that are in both:
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                cmd_idx = 0

            else:
                carb.log_warn("Plan did not converge to a solution.  No action is being taken.")
            target_pose = cube_position
            target_orientation = cube_orientation
        past_pose = cube_position
        past_orientation = cube_orientation
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            # set desired joint angles obtained from IK:
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
    simulation_app.close()


### MPC Test
def mpc_example():
    pass


### TrajOpt Test
def trajopt_example():
    pass


### World Representation Test
def world_representation_example():
    pass


@hydra.main(version_base=None, config_name="assembly_config", config_path="../configs")
def main(cfg: DictConfig):
    curobo_cfg = cfg.curobo

    curobo_utils_example(curobo_cfg)
    collision_check_example()
    ik_example()
    kinematics_example()
    motion_gen_example()
    mpc_example()
    trajopt_example()
    world_representation_example()


if __name__ == "__main__":
    main()
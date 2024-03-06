

def get_ir_sampler(problem, custom_limits={}, max_attempts=25, collisions=True, learned=False, resolution=0.05):
    # Sample move_base pose
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    gripper = problem.get_gripper()

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
            if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
                return

        gripper_pose = pose.value # multiply(pose.value, invert(grasp.value))
        default_conf = arm_conf(arm, grasp.carry)
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_group_joints(robot, 'base')

        if learned:
            base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)

        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            for base_conf in islice(base_generator, max_attempts):
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                yield (bq,)
                break
            else:
                yield None
    return gen_fn

def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=False, resolution=0.05):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    if is_ik_compiled():
        print('Using ikfast for inverse kinematics')
    else:
        print('Using pybullet for inverse kinematics')

    saved_place_conf = {}
    saved_default_conf = {}
    def fn(arm, obj, pose, grasp, base_conf):
        # Obstacles
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}

        # HSR joints
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_base_joints(robot, arm)
        base_arm_joints = get_base_arm_joints(robot, arm)

        # Select grasp_type
        if 'shaft' in str(problem.body_names[obj]):
            grasp_type = 'side'
        elif 'gear' in str(problem.body_names[obj]):
            grasp_type = 'bottom'

        # Set planning parameters
        resolutions = resolution * np.ones(len(base_arm_joints))
        weights = [10, 10, 10, 50, 50, 50, 50, 50]

        # Default confs
        default_arm_conf = arm_conf(arm, grasp.carry)
        default_base_conf = get_joint_positions(robot, base_joints)
        default_base_arm_conf = [*default_base_conf, *default_arm_conf]

        pose.assign()
        base_conf.assign()

        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {attachment.child: attachment}

        # If pick action, return grasp pose (pose is obtained from PyBullet at initialization)
        pick_pose, approach_pose = deterministic_pick(obj, pose)

        # Set position to default configuration for grasp action
        set_joint_positions(robot, arm_joints, default_arm_conf)

        pick_conf = hsr_inverse_kinematics(robot, arm, pick_pose, custom_limits=custom_limits)
        if (pick_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):
            return None

        approach_conf = hsr_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            return None

        approach_conf = get_joint_positions(robot, base_arm_joints)

        # Plan joint motion for grasp
        grasp_path = plan_joint_motion(robot,
                                    base_arm_joints,
                                    pick_conf,
                                    attachments=attachments.values(),
                                    obstacles=approach_obstacles,
                                    self_collisions=SELF_COLLISIONS,
                                    custom_limits=custom_limits,
                                    resolutions=resolutions/2.,
                                    weights=weights,
                                    restarts=2,
                                    iterations=25,
                                    smooth=25)
        if grasp_path is None:
            print('Grasp path failure')
            return None

        set_joint_positions(robot, base_arm_joints, default_base_arm_conf)

        # Plan joint motion for approach
        approach_path = plan_direct_joint_motion(robot,
                                                base_arm_joints,
                                                approach_conf,
                                                attachments=attachments.values(),
                                                obstacles=obstacles,
                                                self_collisions=SELF_COLLISIONS,
                                                custom_limits=custom_limits,
                                                resolutions=resolutions)
        if approach_path is None:
            print('Approach path failure')
            return None

        grasp_arm_conf = get_carry_conf(arm, grasp_type)
        grasp_base_conf = default_base_conf
        grasp_base_arm_conf = [*grasp_base_conf, *grasp_arm_conf]

        set_joint_positions(robot, base_arm_joints, approach_conf)

        # Plan joint motion for return
        return_path = plan_direct_joint_motion(robot,
                                            base_arm_joints,
                                            grasp_base_arm_conf,
                                            attachments=attachments.values(),
                                            obstacles=obstacles,
                                            self_collisions=SELF_COLLISIONS,
                                            custom_limits=custom_limits,
                                            resolutions=resolutions)
        if return_path is None:
            print('Return path failure')
            return None

        path1 = approach_path
        mt1 = create_trajectory(robot, base_arm_joints, path1)
        path2 = grasp_path
        mt2 = create_trajectory(robot, base_arm_joints, path2)
        path3 = return_path
        mt3 = create_trajectory(robot, base_arm_joints, path3)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt1, mt2, mt3])
        return (cmd,)
    return fn

def get_ik_gen(problem, max_attempts=25, learned=False, teleport=False, **kwargs):
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=1, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)

    def gen_fn(*inputs):
        b, a, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
    return gen_fn
import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation
from xarm_tamp.tampkit.sim_tools.sim_utils import (
    # utils
    connect, step_simulation, apply_action, is_circular,
    sample_aabb, aabb2d_from_aabb, aabb_empty, aabb_contains_aabb,
    is_placement, is_insertion, check_geometry_type,
    approximate_as_prism, wrap_interval, circular_difference,
    flatten, convex_combination, unit_vector,
    quaternion_slerp, quat_combination, multiply, invert,
    # sim env api
    create_world, create_floor, create_table, create_robot,
    create_hole, create_surface,
    # setter
    set_pose, set_initial_conf, set_velocity, set_transform_world,
    set_transform_local, set_joint_positions,
    # getter
    get_initial_conf, get_bodies, get_body_name, get_pose,
    get_velocity, get_transform_local, get_transform_world,
    get_all_links, get_link, get_tool_link, get_parent,
    get_children, get_link_parents, get_link_children, get_link_descendants,
    get_link_pose, get_all_link_parents, get_all_link_children,
    get_link_subtree, get_arm_joints, get_gripper_joints, get_movable_joints,
    get_joint_positions, get_joint_velocities, get_min_limit,
    get_max_limit, get_joint_limits, get_custom_limits,
    get_group_conf, get_joints, get_difference_fn, get_refine_fn,
    get_extend_fn, get_distance_fn, get_aabb, get_center_extent,
    get_body_geometry, get_pairs, get_distance,
)


### Simulation API
def sim_api_test():
    # connect to isaac sim
    connect()

    # disconnect isaac sim app; disconnect()

    # step simulation
    world = create_world()
    for _ in range(10):
        print('step simulation')
        step_simulation(world)

    return world


### Test Environment API
def env_api_test(cfg, world):
    # create floor
    floor = create_floor(world, cfg.sim.floor)
    print('floor:', floor)

    # create table
    table = create_table(cfg.sim.table)
    set_pose(table, (cfg.sim.table.translation, cfg.sim.table.orientation))
    print('table:', table.name)
    world.scene.add(table)

    # create robot
    robot = create_robot(cfg.sim.robot)
    initial_conf = get_initial_conf(robot)
    set_initial_conf(robot, initial_conf)
    print('robot:', robot.name)
    world.scene.add(robot)

    # create hole
    pose = [np.array([0.0, 0.1, 0.5]), np.array([1., 0., 0., 0.])]
    hole = create_hole(cfg.sim.hole1.name, *pose) # TODO: fix, remove collision & visual
    print('hole:', hole.name)
    world.scene.add(hole)

    # create surface
    pose = [np.array([0.0, -0.1, 0.5]), np.array([1., 0., 0., 0.])]
    surf = create_surface(cfg.sim.surface1.name, *pose) # TODO: fix, remove collision & visual
    print('surf:', surf.name)
    world.scene.add(surf)
    
    world.reset()

    return floor, table, robot, hole, surf


### Rigid Body API
def rigid_body_api_test(world):
    # get bodies
    bodies = get_bodies(world)
    print('all bodies:', bodies)

    # get names
    for body in bodies:
        name = get_body_name(body)
        print('body name:', name)
        
    # get pose
    for body in bodies:
        pos, rot = get_pose(body)
        print('position:', pos)
        print('rotation:', rot)

    # set pose
    rigid_bodies = get_bodies(world, body_types=['rigid'])
    for body in rigid_bodies:
        position = np.array([0.1, 0., 0.])  # TODO: add position randomization from utils
        orientation = np.array([0., 0., 0., 1.])  # TODO: add orientation randomization from utils
        set_pose(body, (position, orientation))

    # get velocity
    for body in rigid_bodies:
        lin_vel, ang_vel = get_velocity(body)
        print('linear velocity:', lin_vel)
        print('angular velocity:', ang_vel)

    # set velocity
    for body in rigid_bodies:
        translation = np.array([0.1, 0., 0.])  # TODO: add translation velocity randomization from utils
        rotation = np.array([0., 0.5, 0.])  # TODO: add rotation velocity randomization from utils
        set_velocity(body, translation, rotation)

    # set local transform
    for body in rigid_bodies:
        translation = np.array([0.0, 0.1, 0.0])
        rotation = np.array([0.0, 0.0, 0.0, 1.0])
        
        transform = np.zeros(4, 4)
        rot_mat = Rotation.from_quat(rotation).as_matrix()
        transform[:3, :3] = rot_mat
        transform[:3, 3] = translation
        set_transform_local(body.prim, transform)

    # set world transform
    for body in rigid_bodies:
        translation = np.array([0.0, 0.1, 0.0])
        rotation = np.array([0.0, 0.0, 0.0, 1.0])

        transform = np.zeros(4, 4)
        rot_mat = Rotation.from_quat(rotation).as_matrix()
        transform[:3, :3] = rot_mat
        transform[:3, 3] = translation
        set_transform_world(body.prim, transform)

    # get local transform
    for body in rigid_bodies:
        pos, rot = get_transform_local(body.prim)
        print('position:', pos)
        print('rotation:', rot)

    # get world transform
    for body in rigid_bodies:
        pos, rot = get_transform_world(body.prim)
        print('position:', pos)
        print('rotation:', rot)


### Link Utils
def link_api_test(robot):
    # get all links
    link_prims = get_all_links(robot)
    print('link_prims:', link_prims)

    # get specified link
    link_names = [link_prim.GetName() for link_prim in link_prims]
    print('link_names:', link_names)
    for link_name in link_names:
        link_prim = get_link(robot, link_name)
        print('link_prim:', link_prim)

    # get tool link
    tools = ["xarm_gripper_base_link", "fingertip_centered"]
    for tool in tools:
        tool_frame = get_tool_link(robot, tool)
        print('tool_frame:', tool_frame)

    # get parent
    for link_prim in link_prims:
        parent_link = get_parent(link_prim)
        print('parent_link:', parent_link)

    # get children
    for link_prim in link_prims:
        children_link = get_children(link_prim)
        print('children_link:', children_link)

    # get link parents
    for link_prim in link_prims:
        parents = get_link_parents(robot, link_prim)
        print('parents:', parents)

    # get link children
    for link_prim in link_prims:
        children = get_link_children(robot, link_prim)
        print('children:', children)

    # get link descendants
    for link_prim in link_prims:
        descendants = get_link_descendants(robot, link_prim)
        print('descendants:', descendants)

    # get link subtree
    for link_prim in link_prims:
        subtree = get_link_subtree(robot, link_prim)
        print('subtree:', subtree)

    # get link pose
    link_pose = get_link_pose(robot, link_name)
    print('link_pose:', link_pose)

    # get all link parents
    all_parents = get_all_link_parents(robot)
    print('all_parents:', all_parents)

    # get all link children
    all_children = get_all_link_children(robot)
    print('all_children:', all_children)


### Joint Utils
def joint_api_test(robot):
    # get base joints
    # try:
    #     base_joints = get_base_joints(robot)
    #     print('base_joints:', base_joints)
    # except ValueError as e:
    #     print(f'{e}: Base joint does not exist.')

    # get arm joints
    arm_joints = get_arm_joints(robot)
    print('arm_joints:', arm_joints)

    # get gripper joints
    gripper_joints = get_gripper_joints(robot)
    print('gripper_joints:', gripper_joints)

    # get movable joints
    movable_joints = get_movable_joints(robot)
    print('movable_joints:', movable_joints)
    
    # get joint positions
    joint_positions = get_joint_positions(robot)
    print('joint_positions:', joint_positions)
    
    # get joint velocities
    joint_velocities = get_joint_velocities(robot)
    print('joint_velocities:', joint_velocities)

    # get joint minimum limits
    joint_min_limit = get_min_limit(robot)
    print('joint_min_limit:', joint_min_limit)
    
    # get joint maximum limits
    joint_max_limit = get_max_limit(robot)
    print('joint_max_limit:', joint_max_limit)
    
    # get min/max joint limits
    min_limit, max_limit = get_joint_limits(robot)
    print('min_limit:', min_limit)
    print('max_limit:', max_limit)
    
    # get custom limits
    joint_names = robot._arm_dof_names
    custom_limits = get_custom_limits(robot, joint_names)
    print('custom_limits:', custom_limits)
    
    # get initial conf
    initial_conf = get_initial_conf(robot)
    print('initial_conf:', initial_conf)
    
    # get group conf
    arm_conf = get_group_conf(robot, 'arm')
    print('arm_conf:', arm_conf)
    
    # set joint positions
    positions = np.random.randn(7)
    set_joint_positions(robot, positions)

    # set initial conf
    positions = np.random.randn(7)
    set_initial_conf(robot, positions)
    
    # apply action
    arm_pos, arm_vel = np.random.randn(7), np.random.randn(7)
    from omni.isaac.core.utils.types import ArticulationAction
    art_action = ArticulationAction(
        arm_pos, arm_vel, joint_indices=get_movable_joints(robot),
    )
    apply_action(robot, art_action)

    # check is_circular
    joint_prims = get_joints(robot)
    for joint_prim in joint_prims:
        result = is_circular(robot, joint_prim)
        print('is_circular:', result)

    # check get_difference_fn    
    fn = get_difference_fn(robot, joint_prims)
    diff = fn(np.random.randn(7), np.random.randn(7))
    print('difference_fn:', diff)

    # check get_refine_fn
    fn = get_refine_fn(robot, joint_prims)
    refine = fn(np.random.randn(7), np.random.randn(7))
    print('refine_fn:', refine)

    # check get_extend_fn
    fn = get_extend_fn(robot, joint_prims)
    extend = fn(np.random.randn(7), np.random.randn(7))
    print('extend_fn:', extend)

    # check get_distance_fn
    fn = get_distance_fn(robot, joint_prims)
    distance = fn(np.random.randn(7), np.random.randn(7))
    print('distance_fn:', distance)


### Collision Utils
def collision_api_test(world):
    rigid_bodies = get_bodies(world, body_types=['rigid'])

    # get aabb
    for body in rigid_bodies:
        lower, upper = get_aabb(body)
        print('lower:', lower)
        print('upper:', upper)
        
    # get center extent
    for body in rigid_bodies:
        center, diff = get_center_extent(body)
        print('center:', center)
        print('diff:', diff)
        
    # sample aabb
    for body in rigid_bodies:
        aabb = get_aabb(body)
        value = sample_aabb(aabb)
        print('value:', value)
    
    # get aabb2d_from_aabb
    for body in rigid_bodies:
        aabb = get_aabb(body)
        aabb2d = aabb2d_from_aabb(aabb)
        print('aabb2d:', aabb2d)
        
    # get aabb_empty
    for body in rigid_bodies:
        aabb = get_aabb(body)
        empty = aabb_empty(aabb)
        print('empty:', empty)

    # check one aabb contains ohter aabb
    body1, body2 = rigid_bodies[0], rigid_bodies[1]
    aabb1 = get_aabb(body1)
    aabb2 = get_aabb(body2)
    print('body1 contains body2:', aabb_contains_aabb(aabb1, aabb2))

    # check is_placement
    print('is placed on body2:', is_placement(body1, body2))

    # check is_insertion
    print('is inserted into body2:', is_insertion(body1, body2))

    # check geometry types
    for body in rigid_bodies:
        geom_type = check_geometry_type(body)
        print('geometry_type:', geom_type)

    # get mesh-based center extent
    for body in rigid_bodies:
        center, diff = approximate_as_prism(body)
        print('center:', center)
        print('diff:', diff)

    # get mesh
    for body in rigid_bodies:
        mesh = get_body_geometry(body)
        vertices = mesh.vertices
        faces = mesh.faces
        print('vertices:', vertices)
        print('faces:', faces)

    # # get bounds (deprecated)
    # for body in rigid_bodies:
    #     lower, upper = get_bounds(body.prim)


### Math Utils
def math_api_test():
    # check wrap_interval
    result = wrap_interval(value=0.1)
    print('wrap_interval:', result)
    
    # check circular_difference
    result = circular_difference(theta2=np.pi/2, theta1=-np.pi/2)
    print('circular_difference:', result)

    # check flatten
    result = flatten([range(10)])
    print('flatten:', result)

    # check convex_combination
    result = convex_combination(x=0.1, y=0.5)
    print('convex_combination:', result)

    # check unit_vector
    result = unit_vector(data=[0.1, 0.3, 0.5])
    print('unit_vector:', result)

    # check quaternion_slerp
    result = quaternion_slerp(quat0=np.array([0., 0., 0., 1.]),
                              quat1=np.array([1., 0., 0., 0.]),
                              fraction=0.0)
    print('quaternion_slerp:', result)

    # check quat_combination
    result = quat_combination(quat1=np.array([0., 0., 0., 1.]),
                              quat2=np.array([1., 0., 0., 0.]))
    print('quat_combination:', result)

    # check get_pairs
    result = get_pairs(sequence=[1, 2, 3])
    print('get_pairs:', result)

    # check get_distance
    result = get_distance(p1=np.array([1, 2, 3]),
                          p2=np.array([2, 3, 4]))
    print('get_distance:', result)

    # check multiply
    result = multiply(pose1=[np.array([0.0, 0.0, 0.1]), np.array([0., 0., 0., 1.])],
                      pose2=[np.array([0.1, 0.0, 0.0]), np.array([0.5, 0.5, 0.5, 0.5])])
    print('multiply:', result)

    # check invert
    result = invert(pose=[np.array([0.0, 0.0, 0.1]), np.array([1.0, 0.0, 0.0, 0.0])])
    print('invert:', invert)


@hydra.main(version_base=None, config_name="fmb_momo", config_path="../configs")
def main(cfg: DictConfig):
    print('###########################')
    world = sim_api_test()
    print('Simulation API Ready!')
    print('###########################')

    print('###########################')
    floor, table, robot, hole, surf = env_api_test(cfg, world)
    print('Environment API Ready!')
    print('###########################')

    print('###########################')
    rigid_body_api_test(world)
    print('Rigid Body API Ready!')
    print('###########################')

    print('###########################')
    link_api_test(robot)
    print('Link API Ready!')
    print('###########################')

    print('###########################')
    joint_api_test(robot)
    print('Joint API Ready!')
    print('###########################')

    print('###########################')
    collision_api_test(world)
    print('Collision API Ready!')
    print('###########################')

    print('###########################')
    math_api_test()
    print('Math API Ready!')
    print('###########################')

    input('Finish!')


if __name__ == "__main__":
    main()
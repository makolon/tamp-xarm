import hydra
from omegaconf import DictConfig
from xarm_tamp.tampkit.sim_tools.sim_utils import *


### Simulation API
def sim_api_test():
    # connect to isaac sim
    sim_app = connect()
    
    # disconnect isaac sim app; disconnect()
    
    # step simulation
    world = create_world()
    world.reset()
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
    set_pose(table, cfg.sim.table.translation, cfg.sim.table.orientation)
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
    for body in bodies:
        position = np.array([0.1, 0., 0.]) # TODO: add position randomization from utils
        orientation = np.array([0., 0., 0., 1.]) # TODO: add orientation randomization from utils
        set_pose(body, position, orientation)

    # get velocity
    rigid_bodies = get_bodies(world, body_types=['rigid'])
    for body in rigid_bodies:
        lin_vel, ang_vel = get_velocity(body)
        print('linear velocity:', lin_vel)
        print('angular velocity:', ang_vel)

    # set velocity
    for body in rigid_bodies:
        translation = np.array([0.1, 0., 0.]) # TODO: add translation velocity randomization from utils
        rotation = np.array([0., 0.5, 0.]) # TODO: add rotation velocity randomization from utils
        set_velocity(body, translation, rotation)


### Link Utils
def link_api_test(robot):
    robot._articulation_view.initialize()
    print('Robot Initialized!')

    # get all links
    link_prims = get_all_links(robot)
    print('link_prims:', link_prims)

    # get specified link
    link_names = [link_prim.name for link_prim in link_prims]
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
    for link_name in link_names:
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
    # get arm joints
    arm_joints = get_arm_joints(robot)
    print('arm_joints:', arm_joints)
    
    # get base joints
    try:
        base_joints = get_base_joints(robot)
        print('base_joints:', base_joints)
    except:
        print('Base joint does not exist.')

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
    custom_limits = get_custom_limits(robot)
    print('custom_limits:', custom_limits)
    
    # get initial conf
    initial_conf = get_initial_conf(robot)
    print('initial_conf:', initial_conf)
    
    # get group conf
    arm_conf = get_group_conf(robot, 'arm')
    print('arm_conf:', arm_conf)
    
    # set joint positions
    positions = np.array()  # TODO
    set_joint_positions(robot, positions)

    # set initial conf
    positions = np.array()  # TODO
    set_initial_conf(robot, positions)
    
    # apply action
    configuration = np.array()  # TODO
    apply_action(robot, configuration)

    # check is_circular
    result = is_circular(robot, joint)
    print('is_circular:', is_circular)
    
    get_difference_fn(robot, joints)
    get_refine_fn(robot, joints)
    get_extend_fn(robot, joints)
    get_distance_fn(robot, joints)
    refine_path(robot)


### Collision Utils
def collision_api_test(world):
    bodies = get_bodies(world)

    # get aabb
    for body in bodies:
        lower, upper = get_aabb(body)
        print('lower:', lower)
        print('upper:', upper)
        
    # get center extent
    for body in bodies:
        center, diff = get_center_extent(body)
        print('center:', center)
        print('diff:', diff)
        
    # sample aabb
    for body in bodies:
        aabb = get_aabb(body)
        value = sample_aabb(aabb)
        print('value:', value)
    
    # get aabb2d_from_aabb
    for body in bodies:
        aabb = get_aabb(body)
        aabb2d = aabb2d_from_aabb(aabb)
        print('aabb2d:', aabb2d)
        
    # get aabb_empty
    for body in bodies:
        aabb = get_aabb(body)
        empty = aabb_empty(aabb)
        print('empty:', empty)

    # check one aabb contains ohter aabb
    for body in bodies:
        pass


### Math Utils
def math_api_test():
    # check wrap_interval
    result = wrap_interval(value=0.1)
    print('wrap_interval:', result)
    
    # check circular_difference
    result = circular_difference(theta2=np.pi/2, theta1=-np.pi/2)
    print('circular_difference:', result)

    # check flatten
    result = flatten(range(10))
    print('flatten:', result)

    # check convex_combination
    result = convex_combination(x=0.1, y=0.5)
    print('convex_combination:', result)

    # check unit_vector
    result = unit_vector(data=[0.1, 0.3, 0.5])
    print('unit_vector:', result)

    # check quaternion_slerp
    result = quaternion_slerp(quat0=np.array([0., 0., 0., 1.]),
                              quat1=np.array([1., 0., 0., 0.]))
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


@hydra.main(version_base=None, config_name="config", config_path="../configs")
def main(cfg: DictConfig):
    world = sim_api_test()
    print('Simulation API Ready!')

    floor, table, robot, hole, surf = env_api_test(cfg, world)
    print('Environment API Ready!')

    rigid_body_api_test(world)
    print('Rigid Body API Ready!')
    
    link_api_test(robot)
    print('Link API Ready!')

    joint_api_test(robot)
    print('Joint API Ready!')

    collision_api_test()
    print('Collision API Ready!')

    math_api_test()
    print('Math API Ready!')


if __name__ == "__main__":
    main()
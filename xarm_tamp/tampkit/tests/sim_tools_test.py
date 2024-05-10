import xarm_tamp.tampkit.sim_tools.sim_utils
import unittest
import numpy as np
from unittest.mock import MagicMock

from xarm_tamp.tampkit.sim_tools.sim_utils import (
    connect, disconnect, step_simulation, loop_simulation,
    create_world, create_floor, create_block, create_surface, create_hole, create_table, create_robot, create_fmb,
    unit_point, unit_quat, unit_pose, get_point, get_quat,
    get_bodies, get_body_name, get_pose, get_velocity, get_transform_local, get_transform_world, get_transform_relative,
    set_pose, set_velocity, set_transform_local, set_transform_world, set_transform_relative,
    get_ancestor_prims, get_links_for_joint, get_joints_for_articulated_root, get_joints, get_arm_joints, get_base_joints,
    get_gripper_joints, get_movable_joints, get_joint_positions, get_joint_velocities, get_min_limit, get_max_limit,
    get_joint_limits, get_custom_limits, get_initial_conf, get_group_conf, set_joint_positions, set_initial_conf,
    apply_action, is_ancestor, is_a_movable_joint, is_circular, get_difference_fn, get_refine_fn, get_extend_fn,
    get_distance_fn, refine_path, get_pose_distance, interpolate_poses, iterate_approach_path,
    aabb_empty, aabb2d_from_aabb, sample_aabb, get_aabb, get_center_extent, aabb_contains_aabb, is_placed_on_aabb,
    is_placement, is_insertion, approximate_as_prism, get_prim_geometry, get_body_geometry, get_bounds, get_closest_points,
    pairwise_link_collision, any_link_pair_collision, body_collision, pairwise_collision, all_between, parse_body,
    flatten_links, expand_links, wrap_interval, circular_difference, flatten, convex_combination, unit_vector,
    quaternion_slerp, quat_combination, get_pairs, get_distance, multiply, multiply_array, invert, transform_point,
    apply_affine, tf_matrices_from_poses
)

class TestSimulationAPI(unittest.TestCase):

    def test_connect(self):
        # Mock the global simulation_app object
        global simulation_app
        simulation_app = MagicMock()
        self.assertEqual(connect(), simulation_app)

    def test_disconnect(self):
        global simulation_app
        simulation_app = MagicMock()
        disconnect()
        simulation_app.close.assert_called_once()

    def test_step_simulation(self):
        world = MagicMock()
        step_simulation(world)
        world.step.assert_called_once_with(render=True)

    def test_loop_simulation(self):
        world = MagicMock()
        global simulation_app
        simulation_app = MagicMock()
        simulation_app.is_running.side_effect = [True, False]
        loop_simulation(world)
        self.assertEqual(world.step.call_count, 1)
        world.step.assert_called_with(render=True)

class TestCreateSimulationEnvironmentAPI(unittest.TestCase):

    def test_create_world(self):
        world = create_world()
        self.assertIsInstance(world, World)

    def test_create_floor(self):
        world = MagicMock()
        plane_cfg = MagicMock()
        floor = create_floor(world, plane_cfg)
        world.scene.add_default_ground_plane.assert_called_once_with(
            static_friction=plane_cfg.static_friction,
            dynamic_friction=plane_cfg.dynamic_friction,
            restitution=plane_cfg.restitution
        )

    def test_create_block(self):
        block_name = "test_block"
        translation = np.array([0.0, 0.0, 0.0])
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        block = create_block(block_name, translation, orientation)
        self.assertEqual(block.name, block_name)

    def test_create_surface(self):
        surface_name = "test_surface"
        translation = np.array([0.0, 0.0, 0.0])
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        surface = create_surface(surface_name, translation, orientation)
        self.assertEqual(surface.name, surface_name)

    def test_create_hole(self):
        hole_name = "test_hole"
        translation = np.array([0.0, 0.0, 0.0])
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        hole = create_hole(hole_name, translation, orientation)
        self.assertEqual(hole.name, hole_name)

    def test_create_table(self):
        table_cfg = MagicMock()
        table = create_table(table_cfg)
        self.assertEqual(table.name, table_cfg.table_name)

    def test_create_robot(self):
        robot_cfg = MagicMock(name="xarm")
        robot = create_robot(robot_cfg)
        self.assertIsNotNone(robot)

    def test_create_fmb(self):
        fmb_cfg = MagicMock(task='momo')
        block = create_fmb(fmb_cfg)
        self.assertIsNotNone(block)

class TestUnitAPI(unittest.TestCase):

    def test_unit_point(self):
        point = unit_point()
        self.assertTrue(np.array_equal(point, np.array([0.0, 0.0, 0.0])))

    def test_unit_quat(self):
        quat = unit_quat()
        self.assertTrue(np.array_equal(quat, np.array([1.0, 0.0, 0.0, 0.0])))

    def test_unit_pose(self):
        pose = unit_pose()
        self.assertEqual(pose, (unit_point(), unit_quat()))

    def test_get_point(self):
        body = MagicMock()
        body.get_world_pose.return_value = (np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        point = get_point(body)
        self.assertTrue(np.array_equal(point, np.array([1.0, 2.0, 3.0])))

    def test_get_quat(self):
        body = MagicMock()
        body.get_world_pose.return_value = (np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        quat = get_quat(body)
        self.assertTrue(np.array_equal(quat, np.array([0.0, 0.0, 0.0, 1.0])))

class TestRigidBodyAPI(unittest.TestCase):

    def test_get_bodies(self):
        world = MagicMock()
        world.scene._scene_registry._all_object_dicts = [{}, {}, {}, {}, {}]
        bodies = get_bodies(world)
        self.assertIsInstance(bodies, list)

    def test_get_body_name(self):
        body = MagicMock(name="test_body")
        name = get_body_name(body)
        self.assertEqual(name, "test_body")

    def test_get_pose(self):
        body = MagicMock()
        body.get_world_pose.return_value = (np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        pos, rot = get_pose(body)
        self.assertTrue(np.array_equal(pos, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.array_equal(rot, np.array([0.0, 0.0, 0.0, 1.0])))

    def test_get_velocity(self):
        body = MagicMock()
        body.linear_velocity = np.array([1.0, 2.0, 3.0])
        body.angular_velocity = np.array([0.1, 0.2, 0.3])
        linear, angular = get_velocity(body)
        self.assertTrue(np.array_equal(linear, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.array_equal(angular, np.array([0.1, 0.2, 0.3])))

    def test_get_transform_local(self):
        prim = MagicMock()
        prim.GetAttribute.return_value = True
        local_transform = np.eye(4)
        prim.GetLocalTransformation.return_value = local_transform
        local_pos, local_quat = get_transform_local(prim)
        self.assertTrue(np.array_equal(local_pos, np.array([0.0, 0.0, 0.0])))
        self.assertTrue(np.array_equal(local_quat, np.array([1.0, 0.0, 0.0, 0.0])))

    def test_get_transform_world(self):
        prim = MagicMock()
        prim.GetAttribute.return_value = True
        world_transform = np.eye(4)
        prim.ComputeLocalToWorldTransform.return_value = world_transform
        world_pos, world_quat = get_transform_world(prim)
        self.assertTrue(np.array_equal(world_pos, np.array([0.0, 0.0, 0.0])))
        self.assertTrue(np.array_equal(world_quat, np.array([1.0, 0.0, 0.0, 0.0])))

    def test_get_transform_relative(self):
        prim = MagicMock()
        other_prim = MagicMock()
        prim.GetAttribute.return_value = True
        other_prim.GetAttribute.return_value = True
        relative_transform = np.eye(4)
        prim.ComputeLocalToWorldTransform.return_value = relative_transform
        other_prim.ComputeLocalToWorldTransform.return_value = np.eye(4)
        relative_pos = get_transform_relative(prim, other_prim)
        self.assertTrue(np.array_equal(relative_pos, np.eye(4)))

    def test_set_pose(self):
        body = MagicMock()
        set_pose(body, np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        body.set_world_pose.assert_called_once_with(position=np.array([1.0, 2.0, 3.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))

    def test_set_velocity(self):
        body = MagicMock()
        set_velocity(body, np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        body.set_linear_velocity.assert_called_once_with(velocity=np.array([1.0, 2.0, 3.0]))
        body.set_angular_velocity.assert_called_once_with(velocity=np.array([0.1, 0.2, 0.3]))

    def test_set_transform_local(self):
        prim = MagicMock()
        transform = np.eye(4)
        set_transform_local(prim, transform)
        prim.GetAttribute("xformOp:transform").Set.assert_called_once_with(Gf.Matrix4d(transform.T), Usd.TimeCode.Default())

    def test_set_transform_world(self):
        prim = MagicMock()
        transform = np.eye(4)
        set_transform_world(prim, transform)
        prim.GetAttribute("xformOp:transform").Set.assert_called_once()

    def test_set_transform_relative(self):
        prim = MagicMock()
        other_prim = MagicMock()
        transform = np.eye(4)
        set_transform_relative(prim, other_prim, transform)
        prim.GetAttribute("xformOp:transform").Set.assert_called_once()

class TestUnitAPI(unittest.TestCase):

    def test_unit_point(self):
        result = unit_point()
        self.assertTrue(np.array_equal(result, np.array([0.0, 0.0, 0.0])))

    def test_unit_quat(self):
        result = unit_quat()
        self.assertTrue(np.array_equal(result, np.array([1.0, 0.0, 0.0, 0.0])))

    def test_unit_pose(self):
        result = unit_pose()
        self.assertEqual(result, (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])))

    def test_get_point(self):
        body = MagicMock()
        body.get_world_pose.return_value = (np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        result = get_point(body)
        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0, 3.0])))

    def test_get_quat(self):
        body = MagicMock()
        body.get_world_pose.return_value = (np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        result = get_quat(body)
        self.assertTrue(np.array_equal(result, np.array([0.0, 0.0, 0.0, 1.0])))

class TestCollisionGeometryAPI(unittest.TestCase):

    def test_aabb_empty(self):
        aabb = (np.array([0, 0, 0]), np.array([1, 1, 1]))
        self.assertFalse(aabb_empty(aabb))

    def test_aabb2d_from_aabb(self):
        aabb = (np.array([0, 0, 0]), np.array([1, 1, 1]))
        result = aabb2d_from_aabb(aabb)
        self.assertEqual(result, (np.array([0, 0]), np.array([1, 1])))

    def test_sample_aabb(self):
        aabb = (np.array([0, 0, 0]), np.array([1, 1, 1]))
        result = sample_aabb(aabb)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))

    def test_get_aabb(self):
        body = MagicMock()
        body.prim_path = "test_path"
        result = get_aabb(body)
        self.assertIsInstance(result, tuple)

    def test_get_center_extent(self):
        body = MagicMock()
        result = get_center_extent(body)
        self.assertIsInstance(result, tuple)

    def test_aabb_contains_aabb(self):
        contained = (np.array([0, 0, 0]), np.array([1, 1, 1]))
        container = (np.array([0, 0, 0]), np.array([2, 2, 2]))
        self.assertTrue(aabb_contains_aabb(contained, container))

    def test_is_placed_on_aabb(self):
        body = MagicMock()
        bottom_aabb = (np.array([0, 0, 0]), np.array([1, 1, 1]))
        self.assertFalse(is_placed_on_aabb(body, bottom_aabb))

    def test_is_placement(self):
        body = MagicMock()
        surface = MagicMock()
        self.assertFalse(is_placement(body, surface))

    def test_is_insertion(self):
        body = MagicMock()
        hole = MagicMock()
        self.assertFalse(is_insertion(body, hole))

    def test_approximate_as_prism(self):
        body = MagicMock()
        result = approximate_as_prism(body)
        self.assertIsInstance(result, tuple)

    def test_get_prim_geometry(self):
        prim = MagicMock()
        result = get_prim_geometry(prim)
        self.assertIsInstance(result, dict)

    def test_get_body_geometry(self):
        body = MagicMock()
        result = get_body_geometry(body)
        self.assertIsInstance(result, trimesh.Trimesh)

    def test_get_bounds(self):
        root_prim = MagicMock()
        result = get_bounds(root_prim)
        self.assertIsInstance(result, tuple)

    def test_get_closest_points(self):
        body1 = MagicMock()
        body2 = MagicMock()
        result = get_closest_points(body1, body2)
        self.assertIsInstance(result, float)

    def test_pairwise_link_collision(self):
        body1 = MagicMock()
        body2 = MagicMock()
        self.assertFalse(pairwise_link_collision(body1, body2))

    def test_any_link_pair_collision(self):
        body1 = MagicMock()
        links1 = ["link1"]
        body2 = MagicMock()
        links2 = ["link2"]
        self.assertFalse(any_link_pair_collision(body1, links1, body2, links2))

    def test_body_collision(self):
        body1 = MagicMock()
        body2 = MagicMock()
        self.assertFalse(body_collision(body1, body2))

    def test_pairwise_collision(self):
        body1 = MagicMock()
        body2 = MagicMock()
        self.assertFalse(pairwise_collision(body1, body2))

class TestMathematicalUtilitiesAPI(unittest.TestCase):

    def test_wrap_interval(self):
        self.assertEqual(wrap_interval(2.5, (0, 2)), 0.5)

    def test_circular_difference(self):
        self.assertEqual(circular_difference(np.pi, -np.pi), 0)

    def test_flatten(self):
        self.assertEqual(flatten([[1, 2], [3, 4]]), [1, 2, 3, 4])

    def test_convex_combination(self):
        result = convex_combination([0, 0], [1, 1], 0.5)
        self.assertTrue(np.array_equal(result, [0.5, 0.5]))


if __name__ == '__main__':
    unittest.main()

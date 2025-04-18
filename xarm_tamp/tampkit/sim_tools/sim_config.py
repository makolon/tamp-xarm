# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import copy

import omni.usd
from xarm_rl.utils.config_utils.default_scene_params import default_actor_options


class SimConfig:
    def __init__(self, config: dict = None):
        if config is None:
            config = dict()

        self._config = config
        self._cfg = config.get("task", dict())

    def parse_actor_config(self, actor_name):
        actor_params = copy.deepcopy(default_actor_options)
        if "sim" in self._cfg and actor_name in self._cfg["sim"]:
            actor_cfg = self._cfg["sim"][actor_name]
            for opt in actor_cfg.keys():
                if actor_cfg[opt] != -1 and opt in actor_params:
                    actor_params[opt] = actor_cfg[opt]
                elif opt not in actor_params:
                    print("Actor params does not have attribute: ", opt)

        return actor_params

    def _get_actor_config_value(self, actor_name, attribute_name, attribute=None):
        actor_params = self.parse_actor_config(actor_name)

        if attribute is not None:
            if attribute_name not in actor_params:
                return attribute.Get()

            if actor_params[attribute_name] != -1:
                return actor_params[attribute_name]
            elif actor_params["override_usd_defaults"] and not attribute.IsAuthored():
                return self._physx_params[attribute_name]
        else:
            if actor_params[attribute_name] != -1:
                return actor_params[attribute_name]

    @property
    def config(self):
        return self._config

    @property
    def task_config(self):
        return self._cfg

    @property
    def physx_params(self):
        return self._physx_params

    def _get_physx_collision_api(self, prim):
        from pxr import PhysxSchema

        physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
        if not physx_collision_api:
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        return physx_collision_api

    def _get_physx_rigid_body_api(self, prim):
        from pxr import PhysxSchema

        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_rb_api:
            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        return physx_rb_api

    def _get_physx_articulation_api(self, prim):
        from pxr import PhysxSchema

        arti_api = PhysxSchema.PhysxArticulationAPI(prim)
        if not arti_api:
            arti_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        return arti_api

    def set_contact_offset(self, name, prim, value=None):
        physx_collision_api = self._get_physx_collision_api(prim)
        contact_offset = physx_collision_api.GetContactOffsetAttr()
        # if not contact_offset:
        #     contact_offset = physx_collision_api.CreateContactOffsetAttr()
        if value is None:
            value = self._get_actor_config_value(name, "contact_offset", contact_offset)
        if value != -1:
            contact_offset.Set(value)

    def set_rest_offset(self, name, prim, value=None):
        physx_collision_api = self._get_physx_collision_api(prim)
        rest_offset = physx_collision_api.GetRestOffsetAttr()
        # if not rest_offset:
        #     rest_offset = physx_collision_api.CreateRestOffsetAttr()
        if value is None:
            value = self._get_actor_config_value(name, "rest_offset", rest_offset)
        if value != -1:
            rest_offset.Set(value)

    def set_position_iteration(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        solver_position_iteration_count = physx_rb_api.GetSolverPositionIterationCountAttr()
        if value is None:
            value = self._get_actor_config_value(
                name, "solver_position_iteration_count", solver_position_iteration_count
            )
        if value != -1:
            solver_position_iteration_count.Set(value)

    def set_velocity_iteration(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        solver_velocity_iteration_count = physx_rb_api.GetSolverVelocityIterationCountAttr()
        if value is None:
            value = self._get_actor_config_value(
                name, "solver_velocity_iteration_count", solver_velocity_iteration_count
            )
        if value != -1:
            solver_velocity_iteration_count.Set(value)

    def set_max_depenetration_velocity(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        max_depenetration_velocity = physx_rb_api.GetMaxDepenetrationVelocityAttr()
        if value is None:
            value = self._get_actor_config_value(name, "max_depenetration_velocity", max_depenetration_velocity)
        if value != -1:
            max_depenetration_velocity.Set(value)

    def set_sleep_threshold(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        sleep_threshold = physx_rb_api.GetSleepThresholdAttr()
        if value is None:
            value = self._get_actor_config_value(name, "sleep_threshold", sleep_threshold)
        if value != -1:
            sleep_threshold.Set(value)

    def set_stabilization_threshold(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        stabilization_threshold = physx_rb_api.GetStabilizationThresholdAttr()
        if value is None:
            value = self._get_actor_config_value(name, "stabilization_threshold", stabilization_threshold)
        if value != -1:
            stabilization_threshold.Set(value)

    def set_gyroscopic_forces(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        enable_gyroscopic_forces = physx_rb_api.GetEnableGyroscopicForcesAttr()
        if value is None:
            value = self._get_actor_config_value(name, "enable_gyroscopic_forces", enable_gyroscopic_forces)
        if value != -1:
            enable_gyroscopic_forces.Set(value)

    def set_density(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        density = physx_rb_api.GetDensityAttr()
        if value is None:
            value = self._get_actor_config_value(name, "density", density)
        if value != -1:
            density.Set(value)
            # auto-compute mass
            self.set_mass(prim, 0.0)

    def set_mass(self, name, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        mass = physx_rb_api.GetMassAttr()
        if value is None:
            value = self._get_actor_config_value(name, "mass", mass)
        if value != -1:
            mass.Set(value)

    def make_kinematic(self, name, prim, cfg, value=None):
        # make rigid body kinematic (fixed base and no collision)
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if value is None:
            value = self._get_actor_config_value(name, "make_kinematic")
        if value:
            # parse through all children prims
            prims = [prim]
            while len(prims) > 0:
                cur_prim = prims.pop(0)
                rb = UsdPhysics.RigidBodyAPI.Get(stage, cur_prim.GetPath())

                if rb:
                    rb.CreateKinematicEnabledAttr().Set(True)

                children_prims = cur_prim.GetPrim().GetChildren()
                prims = prims + children_prims

    def set_articulation_position_iteration(self, name, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        solver_position_iteration_count = arti_api.GetSolverPositionIterationCountAttr()
        if value is None:
            value = self._get_actor_config_value(
                name, "solver_position_iteration_count", solver_position_iteration_count
            )
        if value != -1:
            solver_position_iteration_count.Set(value)

    def set_articulation_velocity_iteration(self, name, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        solver_velocity_iteration_count = arti_api.GetSolverVelocityIterationCountAttr()
        if value is None:
            value = self._get_actor_config_value(
                name, "solver_velocity_iteration_count", solver_velocity_iteration_count
            )
        if value != -1:
            solver_velocity_iteration_count.Set(value)

    def set_articulation_sleep_threshold(self, name, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        sleep_threshold = arti_api.GetSleepThresholdAttr()
        if value is None:
            value = self._get_actor_config_value(name, "sleep_threshold", sleep_threshold)
        if value != -1:
            sleep_threshold.Set(value)

    def set_articulation_stabilization_threshold(self, name, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        stabilization_threshold = arti_api.GetStabilizationThresholdAttr()
        if value is None:
            value = self._get_actor_config_value(name, "stabilization_threshold", stabilization_threshold)
        if value != -1:
            stabilization_threshold.Set(value)

    def apply_rigid_body_settings(self, name, prim, cfg, is_articulation):
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPath())
        if not physx_rb_api:
            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

        # if it's a body in an articulation, it's handled at articulation root
        if not is_articulation:
            self.make_kinematic(name, prim, cfg, cfg["make_kinematic"])
        self.set_position_iteration(name, prim, cfg["solver_position_iteration_count"])
        self.set_velocity_iteration(name, prim, cfg["solver_velocity_iteration_count"])
        self.set_max_depenetration_velocity(name, prim, cfg["max_depenetration_velocity"])
        self.set_sleep_threshold(name, prim, cfg["sleep_threshold"])
        self.set_stabilization_threshold(name, prim, cfg["stabilization_threshold"])
        self.set_gyroscopic_forces(name, prim, cfg["enable_gyroscopic_forces"])

        # density and mass
        mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
        if mass_api is None:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_attr = mass_api.GetMassAttr()
        density_attr = mass_api.GetDensityAttr()
        if not mass_attr:
            mass_attr = mass_api.CreateMassAttr()
        if not density_attr:
            density_attr = mass_api.CreateDensityAttr()

        if cfg["density"] != -1:
            density_attr.Set(cfg["density"])
            mass_attr.Set(0.0)  # mass is to be computed
        elif cfg["override_usd_defaults"] and not density_attr.IsAuthored() and not mass_attr.IsAuthored():
            density_attr.Set(self._physx_params["density"])

    def apply_rigid_shape_settings(self, name, prim, cfg):
        from pxr import PhysxSchema, UsdPhysics

        # collision APIs
        collision_api = UsdPhysics.CollisionAPI(prim)
        if not collision_api:
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
        if not physx_collision_api:
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)

        self.set_contact_offset(name, prim, cfg["contact_offset"])
        self.set_rest_offset(name, prim, cfg["rest_offset"])

    def apply_articulation_settings(self, name, prim, cfg):
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()

        is_articulation = False
        # check if is articulation
        prims = [prim]
        while len(prims) > 0:
            prim_tmp = prims.pop(0)
            articulation_api = UsdPhysics.ArticulationRootAPI.Get(stage, prim_tmp.GetPath())
            physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Get(stage, prim_tmp.GetPath())

            if articulation_api or physx_articulation_api:
                is_articulation = True

            children_prims = prim_tmp.GetPrim().GetChildren()
            prims = prims + children_prims

        # parse through all children prims
        prims = [prim]
        while len(prims) > 0:
            cur_prim = prims.pop(0)
            rb = UsdPhysics.RigidBodyAPI.Get(stage, cur_prim.GetPath())
            collision_body = UsdPhysics.CollisionAPI.Get(stage, cur_prim.GetPath())
            articulation = UsdPhysics.ArticulationRootAPI.Get(stage, cur_prim.GetPath())
            if rb:
                self.apply_rigid_body_settings(name, cur_prim, cfg, is_articulation)
            if collision_body:
                self.apply_rigid_shape_settings(name, cur_prim, cfg)

            if articulation:
                articulation_api = UsdPhysics.ArticulationRootAPI.Get(stage, cur_prim.GetPath())
                physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Get(stage, cur_prim.GetPath())

                # enable self collisions
                enable_self_collisions = physx_articulation_api.GetEnabledSelfCollisionsAttr()
                if cfg["enable_self_collisions"] != -1:
                    enable_self_collisions.Set(cfg["enable_self_collisions"])

                self.set_articulation_position_iteration(name, cur_prim, cfg["solver_position_iteration_count"])
                self.set_articulation_velocity_iteration(name, cur_prim, cfg["solver_velocity_iteration_count"])
                self.set_articulation_sleep_threshold(name, cur_prim, cfg["sleep_threshold"])
                self.set_articulation_stabilization_threshold(name, cur_prim, cfg["stabilization_threshold"])

            children_prims = cur_prim.GetPrim().GetChildren()
            prims = prims + children_prims
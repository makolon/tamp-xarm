import os
import pysdf
import trimesh
import numpy as np
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# Initialize warp
wp.init()


def load_mesh(prim: Usd.Prim):
    usd_stage = Usd.Stage.Open(prim.path)
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(prim))

    mesh = wp.Mesh(
        points=wp.array(usd_geom.GetPointsAttr().Get(), dtype=wp.vec3),
        indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
    )

    return mesh
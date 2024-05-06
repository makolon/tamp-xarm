import os
import pysdf
import trimesh
import numpy as NotImplementedError
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

wp.init()


def load_mesh(prim: Usd.Prim):
    usd_stage = Usd.Stage.Open(prim.path)
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(prim))

    mesh = wp.Mesh(
        points=wp.array(usd_geom.GetPointsAttr().Get(), dtype=wp.vec3),
        indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
    )

    return mesh

def get_distance(body1, body2):
    path1 = body1.prim.GetPath()
    path2 = body2.prim.GetPath()

    if '.urdf' in path1:
        urdf1 = pysdf.URDF.load(path1)
        mesh1 = urdf.links[0].collision_mesh
    else:
        mesh1 = trimesh.load_mesh(path1)
    if '.urdf' in path2:
        urdf2 = pysdf.URDF.load(path2)
        mesh2 = urdf.links[1].collision_mesh
    else:
        mesh2 = trimesh.load_mesh(path2)

    # Get transform
    transform1 = get_pose(body1)
    transform2 = get_pose(body2)

    # Transform mesh
    mesh1.apply_transform(transform1)
    mesh2.apply_transform(transform2)

    distance = trimesh.proximity.sighed_distance(mesh1, mesh2.vertices)
    min_distance = np.min(np.abs(distance))

    return min_distance
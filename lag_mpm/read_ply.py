import trimesh
import numpy as np

def read_ply(ply_path_no_ext, start=1, stop=100):
    pts=[]
    for i in range(start, stop):
        ply_path = ply_path_no_ext + f"{i:}.ply"
        mesh = trimesh.load(ply_path)
        v = mesh.vertices
        pts.append(np.array(v))
    return pts
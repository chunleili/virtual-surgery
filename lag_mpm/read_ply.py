import trimesh
import numpy as np
import time

def read_ply(ply_path_no_ext, start=1, stop=100):
    pts=[]
    for i in range(start, stop):
        ply_path = ply_path_no_ext + f"{i:}.ply"
        # print("Reading ", ply_path)
        mesh = trimesh.load(ply_path)
        v = mesh.vertices
        # mesh.show()
        # print(np.array(v))
        pts.append(np.array(v))
    return pts



if __name__ == "__main__":
    plys=[]
    ply_path = "D:/Dev/virtual-surgery/models/control_points/CP1/CP1_"
    plys = read_ply(ply_path, start=1, stop=201)

    import taichi as ti
    ti.init()

    N=6121
    plys_ti = ti.Vector.field(3, dtype=ti.f32, shape = N)

    window = ti.ui.Window("Test for reading ply", (800, 800))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(1,1,2)
    camera.lookat(0, 0, 0)
    frame_num = 0
    while window.running:
        frame_num += 1
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        plys_ti.from_numpy(plys[frame_num])

        scene.particles(plys_ti, color = (0.6, 0.26, 0.19), radius = 0.01)
        canvas.scene(scene)

        time.sleep(1.0/24)
        print(frame_num)
        if(frame_num==199):
            frame_num=0
        window.show()
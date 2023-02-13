import os
import taichi as ti
import meshtaichi_patcher as Patcher
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default="./results/",
                        help='Output Path')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--arch', default='cuda')
    args = parser.parse_args()
    return args


args = parse_args()

# os.makedirs(args.output + "/armadillo", exist_ok=True)
# os.makedirs(args.output + "/particles", exist_ok=True)

ti.init(arch=getattr(ti, args.arch), random_seed=0, device_memory_GB=4)
from fem import *


fems, models = [], []

def transform(verts, scale, offset): return verts / max(verts.max(0) - verts.min(0)) * scale + offset
def init(x, y, i):
    # #armadillo
    # model = Patcher.load_mesh_rawdata("./models/armadillo0/armadillo0.1.node")
    # model[0] = transform(model[0], model_size, [x, y, 0.05 + (model_size / 2 + 0.012) * i])

    model = Patcher.load_mesh_rawdata("./models/skin.1.node")

    models.append(model)

# #armadillo TODO:
# model_size = 0.1 
# init(0.5, 0.5, 0)
# armadillo = Patcher.load_mesh(models, relations=["CV"])
# fems.append(FEM(armadillo))

model_size = 0.5
init(0.5, 0.0, 1.0)
skin = Patcher.load_mesh(models, relations=["CV"])
fems.append(FEM(skin))


@ti.kernel
def init_indices_surf(mesh: ti.template(), indices: ti.template()):
    for f in mesh.faces:
        for j in ti.static(range(3)):
           indices[f.id * 3 + j] = f.verts[j].id

ground_model = "models/ground.obj"
ground = Patcher.load_mesh(ground_model, relations=["FV"])
ground_indices = ti.field(ti.i32, shape = len(ground.faces) * 3)
ground.verts.place({'x' : ti.math.vec3})
ground.verts.x.from_numpy(ground.get_position_as_numpy())
init_indices_surf(ground, ground_indices)

coord_model = "models/coord.obj"
coord = Patcher.load_mesh(coord_model, relations=["FV"])
coord.verts.place({'x' : ti.math.vec3})
coord.verts.x.from_numpy(coord.get_position_as_numpy())
coord_indices = ti.field(ti.i32, shape = len(coord.faces) * 3)
init_indices_surf(coord, coord_indices)


@ti.kernel
def init_indices_tet(mesh: ti.template(), indices: ti.template()):
    for c in mesh.cells:
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id

# armadillo_indices = ti.field(ti.u32, shape = len(armadillo.cells) * 4 * 3)
# init_indices_tet(armadillo, armadillo_indices)

skin_indices = ti.field(ti.u32, shape = len(skin.cells) * 4 * 3)
init_indices_tet(skin, skin_indices)


window = ti.ui.Window("virtual surgery", (1920, 1080))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.up(0, 1, 0)
camera.fov(75)
camera.position(0.7,0.7,0.1)
camera.lookat(0, 0, 0)
camera.fov(75)


frame = 0
paused = ti.field(int, shape=())
paused[None] = 1
while window.running:

    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.SPACE:
            paused[None] = not paused[None]
            print("paused:", paused[None])
    if not paused[None]:
        solve(1, fems)
        print(f"frame: {frame}")
        frame += 1
    # print("camera.curr_position",camera.curr_position)
    # print("camera.curr_lookat",camera.curr_lookat)

    camera.track_user_inputs(window, movement_speed=0.0005, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    
    scene.particles(fems[0].x, 1e-4, color = (0.5, 0.5, 0.5))
    scene.mesh(ground.verts.x, ground_indices, color = (0.5,0.5,0.5))
    scene.mesh(coord.verts.x, coord_indices, color = (0.5,0.5,0.5))

    scene.mesh(fems[0].x, skin_indices, color = (232/255, 190/255, 172/255))
    
    # scene.mesh(fems[0].x, armadillo_indices, color = (232/255, 190/255, 172/255))#armadillo

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    scene.ambient_light((0.2,0.2,0.2))

    canvas.scene(scene)

    window.show()

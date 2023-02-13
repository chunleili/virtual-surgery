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

os.makedirs(args.output + "/armadillo", exist_ok=True)
os.makedirs(args.output + "/particles", exist_ok=True)
ti.init(arch=getattr(ti, args.arch), random_seed=0)

from fem import *

model_size = 0.1

fems, models = [], []

def transform(verts, scale, offset): return verts / max(verts.max(0) - verts.min(0)) * scale + offset
def init(x, y, i):
    model = Patcher.load_mesh_rawdata("./models/armadillo0/armadillo0.1.node")
    model[0] = transform(model[0], model_size, [x, y, 0.05 + (model_size / 2 + 0.012) * i])
    models.append(model)

n_armadillo = 1

init(0.5, 0.5, 0)


mesh = Patcher.load_mesh(models, relations=["CV"])
fems.append(FEM(mesh))


ground_model = "models/ground.obj"
ground = Patcher.load_mesh(ground_model, relations=["FV"])
ground.verts.place({'x' : ti.math.vec3})
ground.verts.x.from_numpy(ground.get_position_as_numpy())


ground_indices = ti.field(ti.i32, shape = len(ground.faces) * 3)
@ti.kernel
def init_indices_surf(mesh: ti.template(), indices: ti.template()):
    for f in mesh.faces:
        for j in ti.static(range(3)):
           indices[f.id * 3 + j] = f.verts[j].id

init_indices_surf(ground, ground_indices)


window = ti.ui.Window("virtual surgery", (1920, 1080))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.7,0.7,0.1)
camera.up(0, 1, 0)
camera.lookat(0, 0, 0)
camera.fov(75)


frame = 0
paused = ti.field(int, shape=())
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

    camera.track_user_inputs(window, movement_speed=0.0005, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    
    scene.particles(fems[0].x, 1e-3, color = (0.5, 0.5, 0.5))
    scene.mesh(ground.verts.x, ground_indices, color = (0.5,0.5,0.5))

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    scene.ambient_light((0.2,0.2,0.2))

    canvas.scene(scene)

    window.show()

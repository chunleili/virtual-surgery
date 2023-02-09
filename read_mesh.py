import taichi as ti
import meshtaichi_patcher as Patcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="models/skin2_surf.1.node")
parser.add_argument('--arch', default='gpu')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch), dynamic_index=True, random_seed=0)

mesh = Patcher.load_mesh(args.model, relations=["CE", "CV", "EV"])
mesh.verts.place({'x' : ti.math.vec3})

mesh.verts.x.from_numpy(mesh.get_position_as_numpy())

x = mesh.verts.x

indices = ti.field(ti.u32, shape = len(mesh.cells) * 4 * 3)
@ti.kernel
def init():
    for c in mesh.cells:
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id


init()

window = ti.ui.Window("virtual surgery", (1024, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1, 1.5, 0)
camera.up(0, 1, 0)
camera.lookat(0, 0, 0)
camera.fov(75)

while window.running:
    camera.track_user_inputs(window, movement_speed=0.003, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.mesh(mesh.verts.x, indices, color = (0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    scene.ambient_light((1, 1, 1))

    canvas.scene(scene)

    window.show()
    for event in window.get_events(ti.ui.PRESS):
        if event.key in [ti.ui.ESCAPE]:
            window.running = False

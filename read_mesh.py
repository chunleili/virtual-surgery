import taichi as ti
import meshtaichi_patcher as Patcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="models/skin2_surf.1.node")
parser.add_argument('--arch', default='gpu')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch), dynamic_index=True, random_seed=0)



mesh = Patcher.load_mesh(args.model, relations=["CE", "CV", "EV"])
mesh.verts.place({'x' :         ti.math.vec3, 
                  'v' :         ti.math.vec3,
                  'mul_ans' :   ti.math.vec3,
                  'f' :         ti.math.vec3,
                  'hessian':    ti.f32,
                  'm' :         ti.f32})

mesh.edges.place({'hessian':    ti.f32})
mesh.cells.place({'B' :         ti.math.mat3, 
                  'W' :         ti.f32})
mesh.verts.x.from_numpy(mesh.get_position_as_numpy())

x = mesh.verts.x
m = mesh.verts.m



b = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.verts))
r0 = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.verts))
p0 = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.verts))
y = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.verts))
x0 = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.verts))



indices = ti.field(ti.u32, shape = len(mesh.cells) * 4 * 3)
@ti.kernel
def init():
    for c in mesh.cells:
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id


init()


if args.test:
    # for frame in range(100):
        # newton()
    arr = x.to_numpy()
    print(arr.mean(), (arr**2).mean())
    assert abs(arr.mean() - 0.50) < 2e-2
    assert abs((arr**2).mean() - 0.287) < 2e-2
    exit(0)

window = ti.ui.Window("Projective Dynamics", (1024, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1, 1.5, 0)
camera.up(0, 1, 0)
camera.lookat(0, 0, 0)
camera.fov(75)

while window.running:
    # newton()
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
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

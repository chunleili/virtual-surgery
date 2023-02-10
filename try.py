import taichi as ti
import meshtaichi_patcher as Patcher

ti.init(arch=ti.cpu, random_seed=0)

ground_model = "models/ground.obj"
ground = Patcher.load_mesh(ground_model, relations=["FV"])
ground.verts.place({'x' : ti.math.vec3})
ground.verts.x.from_numpy(ground.get_position_as_numpy())


skin_model = "models/skin2_surf.1.node"
skin = Patcher.load_mesh(skin_model, relations=["CE", "CV", "EV","CF","FV"])
skin.verts.place({'x' : ti.math.vec3})
skin.verts.x.from_numpy(skin.get_position_as_numpy())


ground_indices = ti.field(ti.i32, shape = len(ground.faces) * 3)
@ti.kernel
def init_indices_surf(mesh: ti.template(), indices: ti.template()):
    for f in mesh.faces:
        for j in ti.static(range(3)):
           indices[f.id * 3 + j] = f.verts[j].id

skin_indices = ti.field(ti.u32, shape = len(skin.cells) * 4 * 3)
@ti.kernel
def init_indices_tet(mesh: ti.template(), indices: ti.template()):
    for c in mesh.cells:
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id


init_indices_surf(ground, ground_indices)
init_indices_tet(skin, skin_indices)


window = ti.ui.Window("virtual surgery", (1024, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1, 2, 0)
camera.up(0, 1, 0)
camera.lookat(0, 0, 0)
camera.fov(75)

while window.running:
    camera.track_user_inputs(window, movement_speed=0.005, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.mesh(ground.verts.x, ground_indices, color = (0.5,0.5,0.5))
    scene.mesh(skin.verts.x, skin_indices, color = (232/255, 190/255, 172/255))

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    scene.ambient_light((0.2,0.2,0.2))

    canvas.scene(scene)

    window.show()
    for event in window.get_events(ti.ui.PRESS):
        if event.key in [ti.ui.ESCAPE]:
            window.running = False
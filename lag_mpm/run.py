import os
import taichi as ti
import meshtaichi_patcher as Patcher
import argparse
import read_ply

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

is_aramdillo = False
is_skin = True

fems, models = [], []

def transform(verts, scale, offset): return verts / max(verts.max(0) - verts.min(0)) * scale + offset
# def scale_model_to_01(verts): 
#     bbox_min, bbox_max = find_bbox(verts)
#     print("bbox_min: ", bbox_min)
#     print("bbox_max: ", bbox_max)
#     # 缩放至最大bbox边长为0.8
#     max_scale = max(bbox_max - bbox_min)
#     print("max_scale: ", max_scale)
#     sclae_factor = 0.8 / max_scale 
#     verts *= sclae_factor
#     # 平移至每一个bbox在0.1-0.9范围内
#     bbox_min_new, bbox_max_new = find_bbox(verts)
#     print("bbox_min_new: ", bbox_min_new)
#     print("bbox_max_new: ", bbox_max_new)
#     verts += (0.1 - bbox_min_new)
#     return verts

# def find_bbox(verts):
#     min_x, min_y, min_z = np.min(verts, axis=0)
#     max_x, max_y, max_z = np.max(verts, axis=0)
#     return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])

def init(x, y, i):
    # #armadillo
    if(is_aramdillo):
        model = Patcher.load_mesh_rawdata("./models/armadillo0/armadillo0.1.node")
        model[0] = transform(model[0], model_size, [x, y, 0.05 + (model_size / 2 + 0.012) * i])
    if(is_skin):
        model = Patcher.load_mesh_rawdata("./models/skin/skin3.1.node")
        # model[0] = scale_modtiel_to_01(model[0])
    models.append(model)

# #armadillo TODO:
if(is_aramdillo):
    model_size = 0.1 
    init(0.5, 0.5, 0)
    armadillo = Patcher.load_mesh(models, relations=["CV"])
    fems.append(FEM(armadillo))

if(is_skin):
    init(0,0,0)
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

if(is_aramdillo):
    armadillo_indices = ti.field(ti.u32, shape = len(armadillo.cells) * 4 * 3)
    init_indices_tet(armadillo, armadillo_indices)

if(is_skin):
    skin_indices = ti.field(ti.u32, shape = len(skin.cells) * 4 * 3)
    init_indices_tet(skin, skin_indices)


cp1 = 20187
cp2 = 15948
#AD-HOC: 现在先直接通过tetview手动看出来控制点的编号，然后update它
# @ti.kernel
def init_cp_pos():
    fems[0].cp_on_skin[0] = fems[0].x[cp1]
    fems[0].cp_on_skin[1] = fems[0].x[cp2]
    fems[0].cp_attractor[None] = fems[0].x[cp1]

ply_path = "D:/Dev/virtual-surgery/models/control_points/CP12_"
plys = read_ply.read_ply(ply_path, start=1, stop=201)

def copy_cp(frame):
    # print(f"{frame} frame, {plys[frame].shape}")
    if(frame < len(plys)):
        fems[0].cp_user.from_numpy(plys[frame])


@ti.kernel
def update_cp_pos(frame:ti.i32):
    fems[0].cp_on_skin[0] = fems[0].x[cp1]
    fems[0].cp_on_skin[1] = fems[0].x[cp2]
    fems[0].cp_attractor[None] += fems[0].keyboard_move[None] * 0.001 #user controlling



window = ti.ui.Window("virtual surgery", (1920, 1080))
gui = window.get_gui()
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.up(0, 1, 0)
camera.fov(75)
# camera.position(0.7,0.7,0.1)
camera.position(0.621,0.667,0.915)
# camera.lookat(0, 0, 0)
camera.lookat(0.443, -0.055,  0.247)
camera.fov(75)


frame = 1
paused = ti.field(int, shape=())
paused[None] = 0
init_cp_pos()
while window.running:
    # user controlling of control points
    fems[0].keyboard_move[None] = ti.Vector([0.0, 0.0, 0.0])
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.SPACE:
            paused[None] = not paused[None]
            print("paused:", paused[None])
        if e.key == ti.ui.ESCAPE:
            exit()
    
    cp_move_speed = 5
    if window.is_pressed("j"):#left x
        fems[0].keyboard_move[None][0] = -cp_move_speed
    elif window.is_pressed("l"):#right x
        fems[0].keyboard_move[None][0] = cp_move_speed
    elif window.is_pressed("i"):#up y
        fems[0].keyboard_move[None][1] = cp_move_speed
    elif window.is_pressed("k"):#down y
        fems[0].keyboard_move[None][1] = -cp_move_speed
    elif window.is_pressed("u"):#forward z
        fems[0].keyboard_move[None][2] = cp_move_speed
    elif window.is_pressed("o"):#backward z
        fems[0].keyboard_move[None][2] = -cp_move_speed

    if window.is_pressed("+"):#backward z
        fems[0].force_strength[None] += 10
    if window.is_pressed("-"):#backward z
        fems[0].force_strength[None] -= 10

    if not paused[None]:
        copy_cp(frame)
        solve(1, fems)
        update_cp_pos(frame)
        frame += 1

    camera.track_user_inputs(window, movement_speed=0.005, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    
    scene.particles(fems[0].x, 1e-4, color = (0.5, 0.5, 0.5))
    scene.particles(fems[0].cp_on_skin, 1e-2, color = (1, 0, 0)) #在皮肤上的
    scene.particles(fems[0].cp_user, 1e-2, color = (1, 1, 0)) # 导入的动画路径
    scene.mesh(ground.verts.x, ground_indices, color = (0.5,0.5,0.5))
    scene.mesh(coord.verts.x, coord_indices, color = (0.5,0.5,0.5))

    if(is_skin):
        scene.mesh(fems[0].x, skin_indices, color = (232/255, 190/255, 172/255))
    if(is_aramdillo):
        scene.mesh(fems[0].x, armadillo_indices, color = (232/255, 190/255, 172/255))#armadillo

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    scene.ambient_light((0.2,0.2,0.2))

    canvas.scene(scene)

    # 加一个选项窗口
    with gui.sub_window("Options", 0, 0, 0.25, 0.3) as w:
        gui.text("w/a/s/d/q/e to move camera")
        gui.text("press j/l/i/k/u/o to move control point")
        gui.text("press +/- to change force strength")
        gui.text("press space to pause/continue")
        fems[0].force_strength[None] = gui.slider_float("force_strength", fems[0].force_strength[None], 0, 1e4)

        gui.text("frame: " + str(frame))
        gui.text("camera.curr_position: " + str(camera.curr_position))
        gui.text("camera.curr_lookat: " + str(camera.curr_lookat))
        gui.text("control point id: " + str(fems[0].cp_id))
        gui.text("cp attractor pos: " + str(fems[0].cp_attractor[None]))
        if paused[None]:
            gui.text("paused")
        switch = gui.button("switch control point")
        if switch and fems[0].cp_id == cp1:
            fems[0].cp_id = cp2
        elif switch and fems[0].cp_id == cp2:
            fems[0].cp_id = cp1

    if frame == len(plys)-1:
        paused[None] = 1
    window.show()

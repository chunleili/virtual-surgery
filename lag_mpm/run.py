import taichi as ti
import meshtaichi_patcher as Patcher
import read_ply

ti.init(arch=ti.cuda, random_seed=0, device_memory_GB=4,kernel_profiler=True)

from fem import *

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

is_aramdillo = False
is_skin = True
armadillo, skin = None, None

def initialize():
    global armadillo, skin
    def init_mesh(x, y, i):
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
        init_mesh(0.5, 0.5, 0)
        armadillo = Patcher.load_mesh(models, relations=["CV"])
        fems.append(FEM(armadillo))
        
    if(is_skin):
        init_mesh(0,0,0)
        skin = Patcher.load_mesh(models, relations=["CV"])
        fems.append(FEM(skin))

initialize()

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


# cp1 = 15948
# cp2 = 20187
cp1 = 15941
cp2 = 20212
#AD-HOC: 现在先直接通过tetview手动看出来控制点的编号，然后update它
# @ti.kernel
def init_cp_pos():
    fems[0].cp_on_skin[0] = fems[0].x[cp1]
    fems[0].cp_on_skin[1] = fems[0].x[cp2]

anime_start_frame, anime_end_frame = 23, 200
def read_animation():
    ply_path = "D:/Dev/virtual-surgery/models/control_points/CP12_"
    plys = read_ply.read_ply(ply_path, start=anime_start_frame, stop=anime_end_frame+1)
    return plys
    

def copy_cp(frame,plys):
    if(frame < len(plys)):
        fems[0].cp_user.from_numpy(plys[frame])
        fems[0].cp_attractor[0] = fems[0].cp_user[0] #用导入动画的点控制attractor
        fems[0].cp_attractor[1] = fems[0].cp_user[1] 


@ti.kernel
def update_cp_pos(frame:ti.i32):
    #cp_on_skin是用来显示的在皮上的红点, cp_attractor是实际计算中的引力中心，cp_user是用户控制或者导入动画的点
    fems[0].cp_on_skin[0] = fems[0].x[cp1] 
    fems[0].cp_on_skin[1] = fems[0].x[cp2] 
    fems[0].cp_attractor[fems[0].cp_id[None]] += fems[0].keyboard_move[None]  #user controlling

    # 只是为了visualize 实际被吸引的一坨点
    for show_num in (show_be_attracted1):
        show_be_attracted1_x[show_num] = fems[0].x[show_be_attracted1[show_num]]
    for show_num in (show_be_attracted2):
        show_be_attracted2_x[show_num] = fems[0].x[show_be_attracted2[show_num]]



show_be_attracted1_x = ti.Vector.field(3, dtype=ti.f32, shape=(7))
show_be_attracted2_x = ti.Vector.field(3, dtype=ti.f32, shape=(7))
show_be_attracted1 = ti.field(ti.i32, shape=(7))
show_be_attracted2 = ti.field(ti.i32, shape=(7))

@ti.kernel
def mark_skin_attracted_particles():
    show_num1 = 0
    show_num2 = 0
    for p in fems[0].x:
        for ii in ti.static(range(2)):
            dist_around_skin = fems[0].x[p] - fems[0].cp_on_skin[ii] #dist from cp_on_skin to skin
            if(dist_around_skin.norm()<0.01):
                fems[0].skin_be_attracted[p] = ii+1 #0是非被吸引的粒子，1是被cp1吸引的粒子，2是被cp2吸引的粒子
        if(fems[0].skin_be_attracted[p] == 1):
            print("p ",p," is attracted by cp1")
            show_be_attracted1[show_num1] = p
            show_num1+=1
        if(fems[0].skin_be_attracted[p] == 2):
            print("p ",p," is attracted by cp2")
            show_be_attracted2[show_num2] = p
            show_num2+=1


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

if __name__ == "__main__":
    frame = 1
    paused = ti.field(int, shape=())
    paused[None] = 1
    init_cp_pos()
    mark_skin_attracted_particles()
    plys = read_animation()
    while window.running:
        # user controlling of control points
        fems[0].keyboard_move[None] = ti.Vector([0.0, 0.0, 0.0])
        cp_move_speed = 5* 0.001

        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                exit()
            if e.key == ti.ui.SPACE:
                paused[None] = not paused[None]
                print("paused:", paused[None])
            if e.key == "r":
                frame = 1 #reload animation
                
        move_dir = (fems[0].cp_attractor[fems[0].cp_id[None]]) - camera.curr_position
        if window.is_pressed("u"):#up y
            fems[0].keyboard_move[None][1] = cp_move_speed
        if window.is_pressed("o"):#down y
            fems[0].keyboard_move[None][1] = -cp_move_speed
        if window.is_pressed("j"):
            fems[0].keyboard_move[None] = -cp_move_speed * move_dir.cross(camera.curr_up)
        if window.is_pressed("l"):
            fems[0].keyboard_move[None] = cp_move_speed * move_dir.cross(camera.curr_up)
        if window.is_pressed("i"):
            fems[0].keyboard_move[None] = cp_move_speed * move_dir
        if window.is_pressed("k"):
            fems[0].keyboard_move[None] = -cp_move_speed * move_dir



        if not paused[None]:
            copy_cp(frame, plys)
            solve(1, fems)
            update_cp_pos(frame)
            frame += 1

        camera.track_user_inputs(window, movement_speed=0.005, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        # cp_on_skin是用来显示的在皮上的红点
        # cp_attractor是实际计算中的引力中心，
        # cp_user是用户控制或者导入动画的点
        scene.particles(fems[0].x, 1e-4, color = (0.5, 0.5, 0.5))
        scene.particles(fems[0].cp_on_skin, 1e-2, color = (1, 0, 0)) #在皮肤上的
        scene.particles(fems[0].cp_user, 1e-2, color = (1, 1, 0)) # 导入的动画/键盘控制的点
        scene.particles(fems[0].cp_attractor, 1e-2, color = (0, 1, 0)) # 实际计算的点 绿色
        scene.particles(show_be_attracted1_x, 5e-3, color = (1, 1, 1)) #只是为了visualize实际被吸引的一坨粒子
        scene.particles(show_be_attracted2_x, 5e-3, color = (1, 1, 0.5)) #只是为了visualize实际被吸引的一坨粒子
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
            gui.text("press space to pause/continue")
            fems[0].force_strength[None] = gui.slider_float("force_strength", fems[0].force_strength[None], 0, 1e5)

            gui.text("frame: " + str(frame))
            gui.text("camera.curr_position: " + str(camera.curr_position))
            gui.text("camera.curr_lookat: " + str(camera.curr_lookat))
            gui.text("control point id: " + str(fems[0].cp_id))
            gui.text("cp attractor pos: " + str(fems[0].cp_attractor[fems[0].cp_id[None]]))
            gui.text("attractor move dir: " + str(move_dir))
            switch = gui.button("switch control point")
            if switch and fems[0].cp_id[None] == 0:
                fems[0].cp_id[None] = 1
            elif switch and fems[0].cp_id[None] == 1:
                fems[0].cp_id[None] = 0
            if(gui.button("reload animation(r)")):
                frame = 1
            
            if(gui.button("pause/continue(SPACE)")):
                paused[None] = not paused[None]
            if paused[None]:
                gui.text("paused")

        if frame == anime_end_frame:
            paused[None] = 1
        window.show()

    ti.profiler.print_kernel_profiler_info() 
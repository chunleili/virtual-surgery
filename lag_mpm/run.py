import taichi as ti
import meshtaichi_patcher as Patcher
import read_ply

# ti.init(arch=ti.cuda, device_memory_GB=4,kernel_profiler=True)
ti.init(arch=ti.cuda, device_memory_GB=4)

from fem import *

def initialize_mesh(mesh_file_path):
    skin = Patcher.load_mesh(mesh_file_path, relations=["CV"])
    fem = FEM(skin)
    return skin, fem

@ti.kernel
def init_indices_tet(mesh: ti.template(), indices: ti.template()):
    for c in mesh.cells:
        ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id

@ti.kernel
def init_indices_surf(mesh: ti.template(), indices: ti.template()):
    for f in mesh.faces:
        for j in ti.static(range(3)):
            indices[f.id * 3 + j] = f.verts[j].id

def read_auxiliary_meshes():
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
    return ground, coord, ground_indices, coord_indices


def read_animation(anime_path, start_frame, stop_frame):
    anime = read_ply.read_ply(anime_path, start=start_frame, stop=stop_frame+1)
    return anime

def update_attractor(frame,anime_sequence, keyboard_move):
    #user controlling
    fem.cp_attractor[cp_id] += keyboard_move

    #read anime
    if(frame < len(anime_sequence)):
        fem.cp_attractor.from_numpy(anime_sequence[frame])


@ti.kernel
def update_visual_cp():
    #cp_on_skin是用来显示的在皮上的红点, cp_attractor是实际计算中的引力中心，cp_user是用户控制或者导入动画的点
    fem.cp_on_skin[0] = fem.x[cp1] 
    fem.cp_on_skin[1] = fem.x[cp2] 

    # 只是为了visualize 实际被吸引的一坨点
    for show_num in (show_be_attracted1):
        show_be_attracted1_x[show_num] = fem.x[show_be_attracted1[show_num]]
    for show_num in (show_be_attracted2):
        show_be_attracted2_x[show_num] = fem.x[show_be_attracted2[show_num]]

# 初始化皮上的控制点（红点）
#AD-HOC: 现在先直接通过tetview手动看出来控制点的编号，然后update它
def init_cp_on_skin_pos():
    fem.cp_on_skin[0] = fem.x[cp1]
    fem.cp_on_skin[1] = fem.x[cp2]

# 把皮上点（红点）周围的一圈点标记出来
@ti.kernel
def mark_skin_attracted_particles():
    show_num1 = 0
    show_num2 = 0
    for p in fem.x:
        #0是非被吸引的粒子，1是被cp1吸引的粒子，2是被cp2吸引的粒子
        #dist from cp_on_skin to skin
        dist_around_skin = fem.x[p] - fem.cp_on_skin[0] 
        if(dist_around_skin.norm()<0.01):
            fem.skin_be_attracted[p] = 1 

        dist_around_skin = fem.x[p] - fem.cp_on_skin[1]
        if(dist_around_skin.norm()<0.01):
            fem.skin_be_attracted[p] = 2

        if(fem.skin_be_attracted[p] == 1):
            print("p ",p," is attracted by cp1")
            show_be_attracted1[show_num1] = p
            show_num1+=1
        if(fem.skin_be_attracted[p] == 2):
            print("p ",p," is attracted by cp2")
            show_be_attracted2[show_num2] = p
            show_num2+=1


def export_ply_mesh_sequence(vert_pos, indices, frame, series_prefix):
    x = vert_pos.to_numpy()[:, 0]
    y = vert_pos.to_numpy()[:, 1]
    z = vert_pos.to_numpy()[:, 2]
    indices_np = indices.to_numpy()
    
    writer = ti.tools.PLYWriter(num_vertices=vert_pos.shape[0],
                                num_faces=int(indices_np.shape[0]/3),
                                face_type="tri")
    writer.add_vertex_pos(x, y, z)
    writer.add_faces(indices_np)
    writer.export_frame(frame, series_prefix)


window = ti.ui.Window("virtual surgery", (1920, 1080))
gui = window.get_gui()
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.up(0, 1, 0)
camera.fov(75)
camera.position(0.621,0.667,0.915)
camera.lookat(0.443, -0.055,  0.247)
camera.fov(75)

if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                paramters setup                               #
    # ---------------------------------------------------------------------------- #

    # 读取模型（皮）
    mesh_file_path = "models/initial_my_skin/initial_my_skin.1.node"
    skin, fem = initialize_mesh(mesh_file_path)
    skin_indices = ti.field(ti.u32, shape = len(skin.cells) * 4 * 3)
    init_indices_tet(skin, skin_indices)
    # cp1, cp2 = 15941,  20212
    cp1, cp2 = 13007, 2484

    # 读取辅助模型（地面、坐标轴）
    ground, coord, ground_indices, coord_indices = read_auxiliary_meshes()

    # 初始化cp_on_skin和它一圈周围的点
    init_cp_on_skin_pos()
    show_be_attracted1_x = ti.Vector.field(3, dtype=ti.f32, shape=(7))
    show_be_attracted2_x = ti.Vector.field(3, dtype=ti.f32, shape=(7))
    show_be_attracted1 = ti.field(ti.i32, shape=(7))
    show_be_attracted2 = ti.field(ti.i32, shape=(7))
    mark_skin_attracted_particles()

    # 读取动画
    anime_start_frame, anime_end_frame = 1, 250
    anime_path = "models/my_skin_cp/cp12_"
    anime_sequence = read_animation(anime_path, anime_start_frame, anime_end_frame)

    # 用户控制attractor
    kerboard_move_speed = 1 * 0.001
    cp_id = 0 # 0是cp1，1是cp2

    # 暂停以及计算帧数
    frame = 1
    paused = ti.field(int, shape=())
    paused[None] = 0

    fem.force_strength[None] = 500

    export_mesh = False #如果导出网格序列，这里改成True
    export_img = False #如果导出图片序列，这里改成True

    if export_mesh or export_img: #构造文件夹
        import os
        if not os.path.exists("results"):
            os.mkdir("results")
    
    # ---------------------------------------------------------------------------- #
    #                                  render loop                                 #
    # ---------------------------------------------------------------------------- #
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                exit()
            if e.key == ti.ui.SPACE:
                paused[None] = not paused[None]
                print("paused:", paused[None])
            if e.key == "r":
                frame = 1 #reload animation
                
        keyboard_move = ti.Vector([0.0, 0.0, 0.0])
        move_dir = (fem.cp_attractor[cp_id]) - camera.curr_position
        if window.is_pressed("u"):#up y
            keyboard_move[1] = kerboard_move_speed
        if window.is_pressed("o"):#down y
            keyboard_move[1] = -kerboard_move_speed
        if window.is_pressed("j"):
            keyboard_move = -kerboard_move_speed * move_dir.cross(camera.curr_up)
        if window.is_pressed("l"):
            keyboard_move = kerboard_move_speed * move_dir.cross(camera.curr_up)
        if window.is_pressed("i"):
            keyboard_move = kerboard_move_speed * move_dir
        if window.is_pressed("k"):
            keyboard_move = -kerboard_move_speed * move_dir

        # 真正的计算逻辑在这！
        if not paused[None]:
            update_attractor(frame, anime_sequence, keyboard_move)
            solve(1, fem)
            update_visual_cp()
            
            if(export_mesh):
                export_ply_mesh_sequence(fem.x, skin_indices, frame, "results/skin_sim_anime")
            frame += 1

        camera.track_user_inputs(window, movement_speed=0.005, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        # cp_on_skin是用来显示的在皮上的红点
        # cp_attractor是实际计算中的引力中心，
        # scene.particles(fem.x, 1e-4, color = (0.5, 0.5, 0.5))
        scene.particles(fem.cp_on_skin, 1e-2, color = (1, 0, 0)) #在皮肤上的
        scene.particles(fem.cp_attractor, 1e-2, color = (0, 1, 0)) # 实际计算的点 绿色
        scene.particles(show_be_attracted1_x, 5e-3, color = (1, 1, 1)) #只是为了visualize实际被吸引的一坨粒子
        scene.particles(show_be_attracted2_x, 5e-3, color = (1, 1, 0.5)) #只是为了visualize实际被吸引的一坨粒子
        scene.mesh(ground.verts.x, ground_indices, color = (0.5,0.5,0.5))
        scene.mesh(coord.verts.x, coord_indices, color = (0.5,0.5,0.5))

        scene.mesh(fem.x, skin_indices, color = (232/255, 190/255, 172/255))

        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        scene.ambient_light((0.2,0.2,0.2))

        canvas.scene(scene)

        # 加一个选项窗口
        with gui.sub_window("Options", 0, 0, 0.25, 0.3) as w:
            gui.text("w/a/s/d/q/e to move camera")
            gui.text("press j/l/i/k/u/o to move control point")
            gui.text("press space to pause/continue")
            fem.force_strength[None] = gui.slider_float("force_strength", fem.force_strength[None], 0, 1e3)

            gui.text("frame: " + str(frame))
            gui.text("camera.curr_position: " + str(camera.curr_position))
            gui.text("camera.curr_lookat: " + str(camera.curr_lookat))
            gui.text("control point id: " + str(cp_id))
            gui.text("cp attractor pos: " + str(fem.cp_attractor[cp_id]))
            gui.text("attractor move dir: " + str(move_dir))
            switch = gui.button("switch control point")
            if switch and cp_id == 0:
                cp_id = 1
            elif switch and cp_id == 1:
                cp_id = 0
            if(gui.button("reload animation(r)")):
                frame = 1
            release = gui.button("release control point")
            if(release):
                fem.force_strength[None] = 0
            
            if(gui.button("pause/continue(SPACE)")):
                paused[None] = not paused[None]
            if paused[None]:
                gui.text("paused")

        if frame == anime_end_frame:
            paused[None] = 1
        
        if export_img:
            img_filename = f'results/{frame:04d}.png'
            window.save_image(img_filename)
        
        window.show()

    # ti.profiler.print_kernel_profiler_info() 
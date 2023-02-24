import taichi as ti

frame_dt = 1.5e-2
dt = 5e-4
allowed_cfl = 0.8
# gravity = 10.0
gravity = 0

# n_grid = (256, 256, 512)
n_grid = (64, 64, 64)
dx = 1.0 / n_grid[0]
dx_inv = 1.0 / dx

# grid_block_size = 128
grid_block_size = 32
leaf_block_size = 4
grid_v = ti.Vector.field(3, dtype = ti.f32)
grid_m = ti.field(ti.f32)
grid = ti.root.pointer(ti.ijk, tuple([x // grid_block_size for x in n_grid]))
block = grid.pointer(ti.ijk, grid_block_size // leaf_block_size)

def block_component(c):
    block.dense(ti.ijk, leaf_block_size).place(c)
    
block_component(grid_m)
for d in range(3):
    block_component(grid_v.get_scalar_field(d))

fraction = 0.0

@ti.kernel
def buildPid(fem : ti.template(), pid : ti.template()):
    ti.loop_config(block_dim=32)
    for p in fem.mesh.verts:
        base = (fem.x[p.id] * dx_inv - 0.5).cast(int)
        base_pid = ti.rescale_index(grid_m, pid.parent(2), base)
        ti.append(pid.parent(), base_pid, p.id)

@ti.kernel
def p2g(fem : ti.template(), dt : ti.f32, pid : ti.template()):
    # attractor 引力施加到cp_on_skin附近的粒子上
    for p in fem.skin_be_attracted:
        if(fem.skin_be_attracted[p]==1):
            dist = fem.cp_attractor[0] - fem.cp_on_skin[0]
            fem.f[p] +=  dist * fem.force_strength[None]
        elif(fem.skin_be_attracted[p]==2):
            dist = fem.cp_attractor[1] - fem.cp_on_skin[1]
            fem.f[p] +=  dist * fem.force_strength[None]

    ti.block_local(grid_m)
    for d in ti.static(range(3)):
        ti.block_local(grid_v.get_scalar_field(d))
    for I in ti.grouped(pid):
        p = pid[I]
        base = (fem.x[p] * dx_inv - 0.5).cast(int)
        Im = ti.rescale_index(pid, grid_m, I)
        for D in ti.static(range(3)):
            base[D] = ti.assume_in_range(base[D], Im[D], 0, 1)
        local = fem.x[p] * dx_inv - base
        w = [0.5 * (1.5 - local) ** 2, 0.75 - (local - 1) ** 2, 0.5 * (local - 0.5) ** 2]
        mass = fem.m[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            dpos = (offset.cast(ti.f32) - local) * dx
            g_v = weight * mass * (fem.v[p] + fem.C[p] @ dpos)
            grid_v[base + offset] += weight * fem.f[p] * dt + g_v
            grid_m[base + offset] += weight * mass

bound = 3
@ti.kernel
def gridOp(fem : ti.template(), dt : ti.f32):
    v_allowed = dx * allowed_cfl / dt
    for I in ti.grouped(grid_m):
        mass = grid_m[I]
        if mass > 0.0:
            grid_v[I] = grid_v[I] / mass

        grid_v[I] = ti.min(ti.max(grid_v[I], -v_allowed), v_allowed)

        # bounding box
        for x in ti.static(range(3)):
            if (I[x] < bound and grid_v[I][x] < 0.0) or (I[x] > n_grid[x] - bound and grid_v[I][x] > 0.0): 
                grid_v[I][x] = 0.0
                grid_v[I] *= fraction

    # for ii in (range(2)):
    #     base = (fem.cp_on_skin[ii] * dx_inv - 0.5).cast(int)
    #     for i, j, k in (ti.ndrange(3, 3, 3)):
    #         offset = ti.Vector([i, j, k])
    #         dist = fem.cp_attractor[ii] - dx * (base + offset)
    #         grid_v[base + offset] += 0.1 + dist * dt * fem.force_strength[None]


@ti.kernel
def computeMaxGridV(grid_v : ti.template()) -> ti.f32:
    max_velocity = 0.0
    for I in ti.grouped(grid_v):
        v = grid_v[I]
        v_max = 0.0
        for i in ti.static(range(3)):
            v_max = ti.max(v_max, abs(v[i]))
        ti.atomic_max(max_velocity, v_max)
    return max_velocity

@ti.kernel
def g2p(fem : ti.template(), dt : ti.f32, pid : ti.template()):
    for d in ti.static(range(3)):
        ti.block_local(grid_v.get_scalar_field(d))
    for I in ti.grouped(pid):
        p = pid[I]
        base = (fem.x[p] * dx_inv - 0.5).cast(int)
        Im = ti.rescale_index(pid, grid_m, I)
        for D in ti.static(range(3)):
            base[D] = ti.assume_in_range(base[D], Im[D], 0, 1)
        local = fem.x[p] * dx_inv - base
        w = [0.5 * (1.5 - local) ** 2, 0.75 - (local - 1) ** 2, 0.5 * (local - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 3)
        new_C = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            dpos = (offset.cast(ti.f32) - local) * dx
            new_v += weight * grid_v[base + offset]
            new_C += weight * 4 * dx_inv * grid_v[base + offset].outer_product(dpos)

        # 显示时间积分 TODO: 我打算改成隐式的看看
        fem.v[p] = new_v
        fem.x[p] += new_v * dt
        fem.C[p] = new_C


def solve(cnt, fems, log=True):
    frame_time_left = frame_dt
    substep = 0
    # while frame_time_left > 0.0: 
    if(True):#DEBUG
        # if log: print(f"substep: {substep}")
        substep += 1

        max_grid_v = computeMaxGridV(grid_v)
        cfl_dt = allowed_cfl * dx / (max_grid_v + 1e-6)
        dt0 = ti.min(dt, cfl_dt, frame_time_left)   
        frame_time_left -= dt0

        grid.deactivate_all()
        for i in range(cnt):    #cnt=1
            fems[i].grid.deactivate_all()
            buildPid(fems[i], fems[i].pid)
            fems[i].f.fill(0.0)
            fems[i].computeForce(fems[i].mesh)
            p2g(fems[i], dt0, fems[i].pid)

        gridOp(fems[i], dt0)

        for i in range(cnt):
            g2p(fems[i], dt0, fems[i].pid)

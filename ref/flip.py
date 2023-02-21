# Modified from mpm88 and mgpcgflip from https://gitee.com/citadel2020/taichi_demos

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)
n_particles = 8192
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
# dt = 2e-4
dt = 1.0/40
n_substeps = 4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))


FLUID, AIR,SOLID, = 0, 1, 2
nx=ny=n_grid
pressure = ti.field(float, (nx, ny))
divergence = ti.field(float, (nx, ny))
marker = ti.field(int, (nx, ny))
px = ti.Vector.field(2, dtype=ti.f32, shape=(nx * 2, ny * 2))#particles
pv = ti.Vector.field(2, dtype=ti.f32, shape=(nx * 2, ny * 2))
pf = ti.field(dtype=ti.i32, shape=(nx * 2, ny * 2))
ux = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
uy = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
ux_temp = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
uy_temp = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
valid = ti.field(dtype=ti.i32, shape=(nx + 1, ny + 1))
valid_temp = ti.field(dtype=ti.i32, shape=(nx + 1, ny + 1))
ux_saved = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
uy_saved = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
flip_viscosity = 0.0


@ti.kernel
def initialize():
    for i, j in pressure:
        pressure[i, j] = 0
        divergence[i, j] = 0

        if (j > ny * 2 // 3 and j < ny - 2):
            marker[i, j] = FLUID
        else:
            marker[i, j] = AIR

        if (i == 0 or i == nx-1 or j == 0 or j == ny-1) \
                or (j == ny * 2 // 3 and i > nx // 3) \
                or (j == ny // 3 and i < nx * 2 // 3):
            marker[i, j] = SOLID

    #particles 也是根据横向和纵向编号的
    for m, n in px:
        i, j = m // 2, n // 2
        px[m, n] = [0.0, 0.0]
        if FLUID == marker[i, j]:
            pf[m, n] = 1

            x = i + ((m % 2) + 0.5) / 2.0
            y = j + ((n % 2) + 0.5) / 2.0

            px[m, n] = [x, y]

    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]
        J[i] = 1


@ti.kernel
def p2g():
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

@ti.kernel
def grid_op():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            inv_m = 1 / grid_m[i, j]
            grid_v[i, j] = inv_m * grid_v[i, j]
            grid_v[i, j].y -= dt * 9.8

            # box
            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0
            if i > n_grid - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0
            if j > n_grid - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0


@ti.kernel
def g2p():
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C



@ti.kernel
def add_force(dt: ti.f32):
    for i, j in uy:
        uy[i, j] += gravity * dt

@ti.kernel
def apply_bc():
    for i, j in marker:
        if SOLID == marker[i, j]:
            ux[i, j] = 0.0
            ux[i + 1, j] = 0.0
            uy[i, j] = 0.0
            uy[i, j + 1] = 0.0

@ti.kernel
def calc_divergence(ux: ti.template(), uy: ti.template(), div: ti.template(), marker: ti.template()):
    for i, j in div:
        if FLUID == marker[i, j]:
            div[i, j] = ux[i, j] - ux[i + 1, j] + uy[i, j] - uy[i, j + 1]



@ti.func
def is_fluid(i, j):
    return i >= 0 and i < nx and j >= 0 and j < ny and FLUID == marker[i, j]

@ti.kernel
def mark_valid_ux():
    for i, j in ux:
        # NOTE that the the air-liquid interface is valid
        if is_fluid(i - 1, j) or is_fluid(i, j):
            valid[i, j] = 1
        else:
            valid[i, j] = 0

@ti.kernel
def mark_valid_uy():
    for i, j in uy:
        # NOTE that the the air-liquid interface is valid
        if is_fluid(i, j - 1) or is_fluid(i, j):
            valid[i, j] = 1
        else:
            valid[i, j] = 0


@ti.kernel
def diffuse_quantity(dst: ti.template(), src: ti.template(),
                     valid_dst: ti.template(), valid: ti.template()):
    for i, j in dst:
        if 0 == valid[i, j]:
            sum = 0.0
            count = 0
            for m, n in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                if 1 == valid[i + m, j + n]:
                    sum += src[i + m, j + n]
                    count += 1
            if count > 0:
                dst[i, j] = sum / float(count)
                valid_dst[i, j] = 1


def extrap_velocity():
    mark_valid_ux()
    for i in range(10):
        ux_temp.copy_from(ux)
        valid_temp.copy_from(valid)
        diffuse_quantity(ux, ux_temp, valid, valid_temp)

    mark_valid_uy()
    for i in range(10):
        uy_temp.copy_from(uy)
        valid_temp.copy_from(valid)
        diffuse_quantity(uy, uy_temp, valid, valid_temp)

@ti.data_oriented
class MultigridPCGPoissonSolver:
    def __init__(self, marker, nx, ny):
        shape = (nx, ny)
        self.nx, self.ny = shape
        print(f'nx, ny = {nx}, {ny}')

        self.dim = 2
        self.max_iters = 300
        self.n_mg_levels = 4
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10
        self.use_multigrid = True

        def _res(l): return (nx // (2**l), ny // (2**l))

        self.r = [ti.field(dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # residual
        self.z = [ti.field(dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # M^-1 r
        self.d = [ti.field(dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # temp
        self.f = [marker] + [ti.field(dtype=ti.i32, shape=_res(_))
                             for _ in range(self.n_mg_levels - 1)]  # marker
        self.L = [ti.Vector.field(6, dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # -L operator

        self.x = ti.field(dtype=ti.f32, shape=shape)  # solution
        self.p = ti.field(dtype=ti.f32, shape=shape)  # conjugate gradient
        self.Ap = ti.field(dtype=ti.f32, shape=shape)  # matrix-vector product
        self.alpha = ti.field(dtype=ti.f32, shape=())  # step size
        self.beta = ti.field(dtype=ti.f32, shape=())  # step size
        self.sum = ti.field(dtype=ti.f32, shape=())  # storage for reductions

        for _ in range(self.n_mg_levels):
            print(f'r[{_}].shape = {self.r[_].shape}')
        for _ in range(self.n_mg_levels):
            print(f'L[{_}].shape = {self.L[_].shape}')

    @ti.func
    def is_fluid(self, f, i, j, nx, ny):
        return i >= 0 and i < nx and j >= 0 and j < ny and FLUID == f[i, j]

    @ti.func
    def is_solid(self, f, i, j, nx, ny):
        return i < 0 or i >= nx or j < 0 or j >= ny or SOLID == f[i, j]

    @ti.func
    def is_air(self, f, i, j, nx, ny):
        return i >= 0 and i < nx and j >= 0 and j < ny and AIR == f[i, j]

    @ti.func
    def neighbor_sum(self, L, x, f, i, j, nx, ny):
        ret = x[(i - 1 + nx) % nx, j] * L[i, j][2]
        ret += x[(i + 1 + nx) % nx, j] * L[i, j][3]
        ret += x[i, (j - 1 + ny) % ny] * L[i, j][4]
        ret += x[i, (j + 1 + ny) % ny] * L[i, j][5]
        return ret

    # -L matrix : 0-diagonal, 1-diagonal inverse, 2...-off diagonals
    @ti.kernel
    def init_L(self, l: ti.template()):
        _nx, _ny = self.nx // (2**l), self.ny // (2**l)
        for i, j in self.L[l]:
            if FLUID == self.f[l][i, j]:
                s = 4.0
                s -= float(self.is_solid(self.f[l], i - 1, j, _nx, _ny))
                s -= float(self.is_solid(self.f[l], i + 1, j, _nx, _ny))
                s -= float(self.is_solid(self.f[l], i, j - 1, _nx, _ny))
                s -= float(self.is_solid(self.f[l], i, j + 1, _nx, _ny))
                self.L[l][i, j][0] = s
                self.L[l][i, j][1] = 1.0 / s
            self.L[l][i, j][2] = float(
                self.is_fluid(self.f[l], i - 1, j, _nx, _ny))
            self.L[l][i, j][3] = float(
                self.is_fluid(self.f[l], i + 1, j, _nx, _ny))
            self.L[l][i, j][4] = float(
                self.is_fluid(self.f[l], i, j - 1, _nx, _ny))
            self.L[l][i, j][5] = float(
                self.is_fluid(self.f[l], i, j + 1, _nx, _ny))

    def solve(self, x, rhs):
        tol = 1e-12

        self.r[0].copy_from(rhs)
        self.x.fill(0.0)

        self.Ap.fill(0.0)
        self.p.fill(0.0)

        for l in range(1, self.n_mg_levels):
            self.downsample_f(self.f[l - 1], self.f[l],
                              self.nx // (2**l), self.ny // (2**l))
        for l in range(self.n_mg_levels):
            self.L[l].fill(0.0)
            self.init_L(l)

        self.sum[None] = 0.0
        self.reduction(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        print(f"init rtr = {initial_rTr}")

        if initial_rTr < tol:
            print(f"converged: init rtr = {initial_rTr}")
        else:
            # r = b - Ax = b    since x = 0
            # p = r = r + 0 p
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            self.update_p()

            self.sum[None] = 0.0
            self.reduction(self.z[0], self.r[0])
            old_zTr = self.sum[None]

            iter = 0
            for i in range(self.max_iters):
                # alpha = rTr / pTAp
                self.apply_L(0, self.p, self.Ap)

                self.sum[None] = 0.0
                self.reduction(self.p, self.Ap)
                pAp = self.sum[None]

                self.alpha[None] = old_zTr / pAp

                # x = x + alpha p
                # r = r - alpha Ap
                self.update_x_and_r()

                # check for convergence
                self.sum[None] = 0.0
                self.reduction(self.r[0], self.r[0])
                rTr = self.sum[None]
                if rTr < initial_rTr * tol:
                    break

                # z = M^-1 r
                if self.use_multigrid:
                    self.apply_preconditioner()
                else:
                    self.z[0].copy_from(self.r[0])

                # beta = new_rTr / old_rTr
                self.sum[None] = 0.0
                self.reduction(self.z[0], self.r[0])
                new_zTr = self.sum[None]

                self.beta[None] = new_zTr / old_zTr

                # p = z + beta p
                self.update_p()
                old_zTr = new_zTr

                iter = i
            print(f'converged to {rTr} in {iter} iters')

        x.copy_from(self.x)

    @ti.kernel
    def apply_L(self, l: ti.template(), x: ti.template(), Ax: ti.template()):
        _nx, _ny = self.nx // (2**l), self.ny // (2**l)
        for i, j in Ax:
            if FLUID == self.f[l][i, j]:
                r = x[i, j] * self.L[l][i, j][0]
                r -= self.neighbor_sum(self.L[l], x,
                                       self.f[l], i, j, _nx, _ny)
                Ax[i, j] = r

    @ti.kernel
    def reduction(self, p: ti.template(), q: ti.template()):
        for I in ti.grouped(p):
            if FLUID == self.f[0][I]:
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x_and_r(self):
        a = float(self.alpha[None])
        for I in ti.grouped(self.p):
            if FLUID == self.f[0][I]:
                self.x[I] += a * self.p[I]
                self.r[0][I] -= a * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if FLUID == self.f[0][I]:
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]


ps = MultigridPCGPoissonSolver(marker, nx, ny)

def solve_pressure():
    divergence.fill(0.0)
    calc_divergence(ux, uy, divergence, marker)
    pressure.fill(0.0)
    ps.solve(pressure, divergence)

@ti.kernel
def apply_pressure():
    for i, j in ux:
        if is_fluid(i - 1, j) or is_fluid(i, j):
            ux[i, j] += pressure[i - 1, j] - pressure[i, j]

    for i, j in uy:
        if is_fluid(i, j - 1) or is_fluid(i, j):
            uy[i, j] += pressure[i, j - 1] - pressure[i, j]


# data : field to sample from
# u : x coord of sample location [0, nx]
# v : y coord of sample location [0, ny]
# ox : x coord of data[0, 0], e.g. 0.5 for cell-centered data
# oy : y coord of data[0, 0], e.g. 0.5 for cell-centered data
# nx : x resolution of data
# ny : y resolution of data
@ti.func
def sample(data, u, v, ox, oy, nx, ny):
    s, t = u - ox, v - oy
    i, j = ti.clamp(int(s), 0, nx - 1), ti.clamp(int(t), 0, ny - 1)
    ip, jp = ti.clamp(i + 1, 0, nx - 1), ti.clamp(j + 1, 0, ny - 1)
    s, t = ti.clamp(s - i, 0.0, 1.0), ti.clamp(t - j, 0.0, 1.0)
    return \
        (data[i, j] * (1 - s) + data[ip, j] * s) * (1 - t) + \
        (data[i, jp] * (1 - s) + data[ip, jp] * s) * t

@ti.func
def vel_interp(pos, ux, uy):
    _ux = sample(ux, pos.x, pos.y, 0.0, 0.5, nx + 1, ny)
    _uy = sample(uy, pos.x, pos.y, 0.5, 0.0, nx, ny + 1)
    return ti.Vector([_ux, _uy])


@ti.kernel
def update_from_grid():
    for i, j in ux:
        ux_saved[i, j] = ux[i, j] - ux_saved[i, j]

    for i, j in uy:
        uy_saved[i, j] = uy[i, j] - uy_saved[i, j]

    for m, n in px:
        if 1 == pf[m, n]:
            gvel = vel_interp(px[m, n], ux, uy)
            dvel = vel_interp(px[m, n], ux_saved, uy_saved)
            pv[m, n] = flip_viscosity * gvel + \
                (1.0 - flip_viscosity) * (pv[m, n] + dvel)

@ti.kernel
def advect_markers(dt: ti.f32):
    for i, j in px:
        if 1 == pf[i, j]:
            midpos = px[i, j] + vel_interp(px[i, j], ux, uy) * (dt * 0.5)
            px[i, j] += vel_interp(midpos, ux, uy) * dt

@ti.kernel
def apply_markers():
    for i, j in marker:
        if SOLID != marker[i, j]:
            marker[i, j] = AIR

    for m, n in px:
        if 1 == pf[m, n]:
            i = ti.clamp(int(px[m, n].x), 0, nx-1)
            j = ti.clamp(int(px[m, n].y), 0, ny-1)
            if (SOLID != marker[i, j]):
                marker[i, j] = FLUID

@ti.func
def splat(data, weights, f, u, v, ox, oy, nx, ny):
    s, t = u - ox, v - oy
    i, j = ti.clamp(int(s), 0, nx - 1), ti.clamp(int(t), 0, ny - 1)
    ip, jp = ti.clamp(i + 1, 0, nx - 1), ti.clamp(j + 1, 0, ny - 1)
    s, t = ti.clamp(s - i, 0.0, 1.0), ti.clamp(t - j, 0.0, 1.0)
    data[i, j] += f * (1 - s) * (1 - t)
    data[ip, j] += f * (s) * (1 - t)
    data[i, jp] += f * (1 - s) * (t)
    data[ip, jp] += f * (s) * (t)
    weights[i, j] += (1 - s) * (1 - t)
    weights[ip, j] += (s) * (1 - t)
    weights[i, jp] += (1 - s) * (t)
    weights[ip, jp] += (s) * (t)

@ti.kernel
def transfer_to_grid(weights_ux: ti.template(), weights_uy: ti.template()):
    for m, n in pv:
        if 1 == pf[m, n]:
            x, y = px[m, n].x, px[m, n].y
            u, v = pv[m, n].x, pv[m, n].y
            splat(ux, weights_ux, u, x, y, 0.0, 0.5, nx + 1, ny)
            splat(uy, weights_uy, v, x, y, 0.5, 0.0, nx, ny + 1)

    for i, j in weights_ux:
        if weights_ux[i, j] > 0.0:
            ux[i, j] /= weights_ux[i, j]

    for i, j in weights_uy:
        if weights_uy[i, j] > 0.0:
            uy[i, j] /= weights_uy[i, j]

def substep(dt):
    add_force(dt)
    apply_bc()

    extrap_velocity()
    apply_bc()

    solve_pressure()
    apply_pressure()

    extrap_velocity()
    apply_bc()

    update_from_grid()
    advect_markers(dt)
    apply_markers()

    ux.fill(0.0)
    uy.fill(0.0)
    ux_temp.fill(0.0)
    uy_temp.fill(0.0)
    transfer_to_grid(ux_temp, uy_temp)  # reuse buffers

    ux_saved.copy_from(ux)
    uy_saved.copy_from(uy)


window = ti.ui.Window("MPM", (1500, 1080))
canvas = window.get_canvas()
paused = ti.field(int, shape=())

def main():
    initialize()

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                paused[None] = not paused[None]
            if e.key == 'r': initialize()
        if not paused[None]:
            # for s in range(int(1e-2 // dt)):
            for _ in range(n_substeps):
                grid_m.fill(0)
                grid_v.fill(0)
                p2g()
                grid_op()
                g2p()

        canvas.circles(x,radius=0.002,color=(1,1,0))
        window.show()


if __name__ == '__main__':
    main()

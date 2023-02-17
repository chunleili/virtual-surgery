# ref: https://blog.csdn.net/weixin_43940314/article/details/128935230

import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug=True)

dim=2
# n_particle_x = 2
# n_particle_y = 1
# n_particles = n_particle_x * n_particle_y
n_particles = 3
n_elements = 1
dt = 1e-4
E, nu = 1e3, 0.33  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters

x = ti.Vector.field(dim, dtype=float, shape=n_particles) #deformed position
force = ti.Vector.field(dim, dtype=float, shape=n_particles)
vel = ti.Vector.field(dim, dtype=float, shape=n_particles)
X = ti.Vector.field(dim, dtype=float, shape=n_particles) #undeformed position
S = ti.Matrix.field(n=dim, m=dim, dtype=float, shape=n_elements) #Second Piola Kirchhoff stress
F = ti.Matrix.field(n=dim, m=dim, dtype=float, shape=n_elements) #deformation gradient
G = ti.Matrix.field(n=dim, m=dim, dtype=float, shape=n_elements) #green strain

Dm_inv = ti.Matrix.field(n=dim, m=dim, dtype=float, shape=n_elements) 
Ds = ti.Matrix.field(n=dim, m=dim, dtype=float, shape=n_elements) 
vertices = ti.Vector.field(3, dtype=ti.i32, shape=(n_elements))
area = ti.field(float, shape=n_elements)

@ti.kernel
def initialize():
    X[0] = [0.5, 0.5]
    X[1] = [0.5, 0.6]
    X[2] = [0.6, 0.5]
    # X[3] = [0.6, 0.6]

    for i in x:
        x[i] = X[i]
    # x[0] += [0, 0.01]

    vertices[0] = [0, 1, 2]
    # vertices[1] = [1, 3, 2]

    # material space shape matrix
    for i in range(n_elements):
        Dm = compute_shape_matrix(i,X)
        Dm_inv[i] = Dm.inverse()
        ii0 = vertices[i][0]
        ii1 = vertices[i][1]
        ii2 = vertices[i][2]
        area[i] = ti.abs((X[ii2] - X[ii0]).cross(X[ii1] - X[ii0])) / 2 


def substep():
    compute_force()
    time_integration()

@ti.func
def compute_shape_matrix(i, x_:ti.template()):
    ii0 = vertices[i][0]
    ii1 = vertices[i][1]
    ii2 = vertices[i][2]
    return  ti.Matrix([[x_[ii1][0]-x_[ii0][0], x_[ii2][0]-x_[ii0][0]],
                       [x_[ii1][1]-x_[ii0][1], x_[ii2][1]-x_[ii0][1]]])

@ti.kernel
def compute_force():
    #compute deformation gradient
    for i in range(n_elements):
        Ds[i] = compute_shape_matrix(i,x)
        F[i] = Ds[i] @ Dm_inv[i]
        print(F[i])

    #compute green strain
    for i in range(n_elements):
        G[i] = 0.5 * (F[i].transpose() @ F[i] - ti.Matrix([[1, 0], [0, 1]]))

    #compute second Piola Kirchhoff stress
    for i in range(n_elements):
        S[i] = 2 * mu *G[i] + lam * (G[i][0,0]+G[i][1,1]) * ti.Matrix([[1, 0], [0, 1]])

    #compute force(先暂且就计算一个三角形的力，后面再考虑多个三角形的情况)
    for i in range(n_elements):
        ii0 = vertices[i][0]
        ii1 = vertices[i][1]
        ii2 = vertices[i][2]
        force_matrix =  F[i] @ S[i] @ Dm_inv[i].transpose() * area[i]
        force[ii1] = ti.Vector([force_matrix[0, 0], force_matrix[1, 0]])
        force[ii2] = ti.Vector([force_matrix[0, 1], force_matrix[1, 1]])
        force[ii0] = -force[ii1] - force[ii2]

    #gravity
    for i in range(n_particles):
        force[i][1] -= 1

@ti.kernel
def time_integration():
    #time integration(with boundary condition)
    for i in range(n_particles):
        x_prev = x[i]
        vel[i] += dt * force[i]
        x[i] += dt * vel[i]

        BC(i,x_prev)


#boundary condition
@ti.func
def BC(i,x_prev):
    eps = 0.01
    cond = (x[i] < eps) & (vel[i] < 0) | (x[i] > 1) & (vel[i] > eps)
    for j in ti.static(range(dim)):
        if cond[j]:
            vel[i][j] = 0  
            x[i] = x_prev

window = ti.ui.Window("MPM", (500, 500))
canvas = window.get_canvas()
gui = window.get_gui()
paused = ti.field(int, shape=())
paused[None] = 1

line_ind = ti.field(int, shape=(n_elements * 3 *2))
def compute_ind():
    vertices_ = vertices.to_numpy()
    a = vertices_.reshape(n_elements * 3)
    b = np.roll(vertices_, shift=1, axis=1).reshape(n_elements * 3)
    line_ind_ = np.stack([a, b], axis=1).flatten()
    print(line_ind_)
    line_ind.from_numpy(line_ind_)

def main():
    initialize()

    compute_ind()

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                paused[None] = not paused[None]
            if e.key == "s":
                substep()
                print("substep")
        if not paused[None]:
            for s in range(30):
                substep()
        
        canvas.lines(x,width=0.001,indices=line_ind, color=(79/255,185/255,159/255))
        canvas.circles(x,radius=0.005,color=(1,1,0))
        window.show()

if __name__ == '__main__':
    main()

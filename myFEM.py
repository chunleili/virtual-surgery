# ref: https://blog.csdn.net/weixin_43940314/article/details/128935230

import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug=True)

dim=2
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


@ti.kernel
def init():
    X[0] = [0.5, 0.5]
    X[1] = [0.5, 0.6]
    X[2] = [0.6, 0.5]

    for i in x:
        x[i] = X[i]
    x[0] += [0, 0.01]


Dm_inv = ti.Matrix.field(n=dim, m=dim, dtype=float, shape=n_elements) 


def substep():
    compute_force()
    time_integration()

@ti.kernel
def compute_force():
    #compute deformation gradient
    for i in range(n_elements):
        Dm =ti.Matrix([[x[1][0]-x[0][0], x[2][0]-x[0][0]], [x[1][1]-x[0][1], x[2][1]-x[0][1]]])
        Dm_inv[i] = Dm.inverse()
        Ds = ti.Matrix([[X[1][0]-X[0][0], X[2][0]-X[0][0]], [X[1][1]-X[0][1], X[2][1]-X[0][1]]])
        F[i] = Ds @ Dm_inv[i]

    #compute green strain
    for i in range(n_elements):
        G[i] = 0.5 * (F[i].transpose() @ F[i] - ti.Matrix([[1, 0], [0, 1]]))

    #compute second Piola Kirchhoff stress
    for i in range(n_elements):
        S[i] = 2 * mu *G[i] + lam * (G[i][0,0]+G[i][1,1]) * ti.Matrix([[1, 0], [0, 1]])

    #compute force(先暂且就计算一个三角形的力，后面再考虑多个三角形的情况)
    for i in range(n_elements):
        ii0 = 3 * i + 0
        ii1 = 3 * i + 1
        ii2 = 3 * i + 2
        area = ti.abs((X[ii2] - X[ii0]).cross(X[ii1] - X[ii0])) / 2 
        force_matrix =  F[i] @ S[i] @ Dm_inv[i].transpose() * area
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


def main():
    init()
    gui = ti.GUI('my', (1024, 1024))
    while gui.running:
        for e in gui.get_events():
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == 'r':
                init()
        for i in range(30):
            substep()
        
        vertices_ = np.array([[0, 1, 2]], dtype=np.int32)
        particle_pos = x.to_numpy()
        a = vertices_.reshape(n_elements * 3)
        b = np.roll(vertices_, shift=1, axis=1).reshape(n_elements * 3)
        gui.lines(particle_pos[a], particle_pos[b], radius=1, color=0x4FB99F)
        gui.circles(particle_pos, radius=5, color=0xF2B134)
        gui.show()

if __name__ == '__main__':
    main()

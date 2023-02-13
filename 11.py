import taichi as ti
ti.init()
num_particles = 10
x = ti.field(dtype=ti.f32, shape=(num_particles), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def compute_x():
    for i in range(num_particles):
        x[i] = 1.0*i/num_particles
compute_x()
for i in range(num_particles):
    print('x =', x[i])

@ti.kernel
def compute_y():
    for i in range(num_particles):
        y[None]+= ti.sin(x[i])

with ti.ad.Tape(y):
    compute_y()

for i in range(num_particles):
    print('dy/dx =', x.grad[i], ' at x =', x[i])
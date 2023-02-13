import taichi as ti

ti.init()
num_particles = 2
x = ti.field(dtype=float, shape=(num_particles), needs_grad=True)
loss = ti.field(dtype=float, shape=(), needs_grad=True)


for i in range(num_particles):
    x[i] = i

@ti.kernel
def test():
    for i in range(num_particles):
        loss[None] += ti.sin(x[i])

with ti.ad.Tape(loss):
    test()

for i in range(num_particles):
    print(loss[None], x.grad[i])
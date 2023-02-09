import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug=True)

dim=2
n_particles = 3
n_elements = 1
x = ti.Vector.field(dim, dtype=float, shape=n_particles)
ele = ti.field(dtype=ti.i32, shape=n_elements) 

@ti.kernel
def init():
    x[0] = [0.5, 0.5]
    x[1] = [0.5, 0.6]
    x[2] = [0.6, 0.5]


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
            pass
        
        vertices_ = np.array([[0, 1, 2]], dtype=np.int32)
        particle_pos = x.to_numpy()
        a = vertices_.reshape(n_elements * 3)
        b = np.roll(vertices_, shift=1, axis=1).reshape(n_elements * 3)
        gui.lines(particle_pos[a], particle_pos[b], radius=1, color=0x4FB99F)
        gui.circles(particle_pos, radius=5, color=0xF2B134)
        gui.show()

if __name__ == '__main__':
    main()

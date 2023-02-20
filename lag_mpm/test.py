import taichi as ti
import numpy as np
ti.init(arch =ti.cuda)


a = np.array([0.456525177, 0.12310902 ,0.37015909,
              0.544431388, 0.125233486, 0.622134209]).reshape(2, 3)


b = ti.Vector.field(3, dtype = ti.f32, shape = 2)

b.from_numpy(a)

@ti.kernel 
def print_b():
    for i in range(2):
        print(b[i])
print_b()
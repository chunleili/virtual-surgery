-- Lua script.
p=tetview:new()
p:load_mesh("C:/Dev/virtual-surgery/models/skin2_surf.1.ele")
rnd=glvCreate(0, 0, 500, 500, "TetView")
p:plot(rnd)
glvWait()

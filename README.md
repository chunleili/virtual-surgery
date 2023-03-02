用taichi MPM实现虚拟手术的扭皮(skin twist)

## 1. 简介
本项目使用taichi实现虚拟手术的扭皮效果。方法采用的是MPM lagrangian forces。参考了[meshtaichi](https://github.com/taichi-dev/meshtaichi)中的lag_mpm。

## 2. pre-requisites
- taichi 1.4.0
- trimesh(仅用于读取控制点ply动画序列)
## 3. 运行
```
python ./lag_mpm/run.py
```
## 4. 效果
![demo](pics/demo.png)

视频展示： [video](pics/demo.mp4)

## 5. 控制方法

![control](pics/control.png)

按空格运行，在250帧内会自动加载控制点（绿色）路径动画。250帧后会自动暂停，需要按空格继续。可以使用j/l/i/k/u/o自主控制控制点，使用switch control point来切换控制点。

注意：由于使用的是MPM。因此网格超出计算域（[0,1]的方块）后会终止计算。

- wasdeq：移动视角
- 空格：暂停/继续
- j/l/i/k/u/o：控制扭皮的控制点（绿色点），分别是左右前后上下(相对当前视角的)。
- switch control point：切换控制点(一共有两个绿色的控制点)
- release control point：释放控制点
- reload animation：重新加载动画
- force_strength: 控制扭皮的力大小

## 6. 目录结构

- lag_mpm：主要代码
- models：模型文件
- pics：图片
- results：结果网格序列
- scripts：houdini的geo格式四面体转换为tetgen格式的脚本

## 7. 输出图片序列/网格序列

export_mesh更改为ture即可输出网格序列，export_img更改为ture即可输出图片序列。结果会保存在results文件夹下。

## 8. 使用自定义的网格的注意事项
1. 网格要求在[0,1]的方块内，实际建议保证在[0.1, 0.9]，这是因为MPM的计算区域的定义在这一区域。

2. meshtaichi目前只支持tetgen格式的网格。有两种方法生成tetgen网格：
   - 可以直接使用tetgen读取ply点云生成四面体网格。
   - 使用houdini生成的网格（tetconform节点）。由于houdini无法直接写出四面体网格，可以使用scripts文件夹下的转换脚本将houdini生成的geo格式四面体网格转换为tetgen格式。脚本中需要给定网格的路径。

PS: houdini转换脚本中genFace可以设置为ture，这样会生成face。（目前meshtaichi需要face，但我们的计算实际不需要，meshtaichi后续更新会去掉这一需求）。生产的tetgen文件为mesh.node, mesh.ele, mesh.face。分别对应点，四面体，面。meshtaichi读取网格只需要指定node即可，会自动搜索同名的ele和face。

3. 需要手动给出控制点的id，并且在run.py中给到cp1和cp2。
    - 如果是使用tetgen生成的网格，建议使用tetview查看网格，然后在网格上查看控制点id。
    - 如果是使用houdini生成的网格，直接在houdini查看其id即可。

4. 我们采用attractor来吸引控制点，实际被吸引的是控制点周围一圈的点。因此我们先mark周围一圈的点。为了保证可并行，实际会存储为一个field skin_be_attracted，该field值为1则代表cp1, 值为2代表cp2。attractor是绿色的点。控制点则是红色的点。

# AWR-Dataset

## TODO Items [10.18 - 10.24]
10.18
1. Process urdf for https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go2_description
2. First sample data using urdf2awr/sample_interpolate.py, and then batch_fk_pc.py (Please make sure using debug mode to see whether the mesh is good for data)
3. We need data for 1k and 100 for go2.
10.19 - 24
1. rig2awr/dataprocess_augment_v2_parellel: 这个是rignet dataset 的data process script 主要是把他的rig_info.txt读出来存成numpy可以读的格式
2. rig2awr/vis_download.py 这个是把上面成的numpy格式的skeleton / skinning weight 和obj mesh 存成一个animatable glb file可以被导入blender
3. 然后我们需要从blender到跟batch_fk_pc.py一样的script
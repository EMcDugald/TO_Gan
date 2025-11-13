import numpy as np
import os

topo = np.load(os.getcwd()+'/TO_3D_data_scratch/data/topologies.npy',allow_pickle=True)
print(topo.shape)

shapes = np.load(os.getcwd()+'/TO_3D_data_scratch/data/shapes.npy',allow_pickle=True)
print(shapes.shape)

import trimesh
# cd 'C:\Users\hdb\Documents\Research\ML RCP 2024\3DTopos'
for count, ele in enumerate(topo):
    if count<20 and count>1:
        arr_3d = ele.reshape(shapes[count]).transpose()
        mcubes = trimesh.voxel.ops.matrix_to_marching_cubes(arr_3d, pitch=1)
        mesh_new=mcubes.split(only_watertight=True)
        face_count_biggest=0
        biggest_body_count=0
        #Find the biggest unconnected mesh item and save it
        for count2, ele2 in enumerate(mesh_new):
            mesh_new_big=mesh_new[count2]
            if (len(mesh_new_big.faces))>face_count_biggest:
                face_count_biggest=len(mesh_new_big.faces)
                biggest_body_count=count2
            mesh_new_big=mesh_new[biggest_body_count]
        filename = os.getcwd()+f"/TO_3D_data_scratch/stls/{count}.stl"
        print(filename)
        mesh_new_big.export(filename)

tst_mesh = trimesh.load(os.getcwd()+'/TO_3D_data_scratch/stls/19.stl')
print("verts shape:",tst_mesh.vertices.shape)
print("faces shape:",tst_mesh.faces.shape)
#tst_mesh.show()   #on hpc suppress this


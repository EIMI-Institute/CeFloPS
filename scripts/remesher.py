import glob
import os
import open3d as o3d
import trimesh


# this script is used to convert the not watertight meshes into a poisson surface resonstructed watertight version.
# run with python3 remesher.py in this folder (the folder with the meshes in stl format).

def categorize_vessel(name):
    artery_keywords = [
        "posterior_inferior_cerebellar", "vertebral", "basilar",
        "superior_cerebellar", "anterior_inferior_cerebellar",
        "medial_occipital_P2", "posterior_cerebral_P1", "parieto_occipital",
        "lateral_occipital", "anterior_communicating", "posterior_communicating",
        "calcarine", "anterior_cerebral_A2", "anterior_cerebral_A1",
        "pericallosal", "medial_frontobasal", "frontopolar",
        "mediomedial_frontal", "posteromedial_frontal", "temporal_polar",
        "paracentral", "middle_cerebral_M2", "middle_cerebral_M1",
        "middle_cerebral_M3", "prefrontal", "middle_temporal",
        "anterior_temporal", "posterior_temporal", "rolandic",
        "internal_carotid", "prerolandic", "anterior_choroidal",
        "lateral_frontobasal"
    ]

    vein_keywords = [
        "superior_saggital_sinus", "central_veins", "straight_sinus",
        "lateral_sinus", "sigmoid_sinus", "basal_cerebral_vein",
        "parietal_superficial_veins", "internal_cerebral_vein",
        "septal_vein", "precentral_veins"
    ]

    for keyword in artery_keywords:
        if keyword in name:
            return "al.arteries"

    for keyword in vein_keywords:
        if keyword in name:
            return "vl.veins"

    return "unknown"
 
def split_to_waterproof(split_m):
    res=[]   
    for p in split_m:
        if p.is_watertight:
            res.append(p)
    return res
def conv_mesh(filepath, new_name):
    mesh = o3d.io.read_triangle_mesh(filepath)

    if mesh.is_empty():
        print("Mesh could not be loaded. Please check your STL file.")
    else:
        print("Mesh loaded successfully.")
    
    mesh.compute_vertex_normals() 
    pcd = mesh.sample_points_poisson_disk(number_of_points=20000) 
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)  
    mesh_poisson.compute_vertex_normals()   
    o3d.io.write_triangle_mesh('./'+new_name+".stl", mesh_poisson)
    #load with trimesh and check if it splits into one waterproof meshobject
    m= trimesh.load('./'+new_name+".stl")
    s=m.split()
    assert len(s)>=1
    if len(s)>1:
        m=split_to_waterproof(s)
        #assert len(m)==1
        for i,n in enumerate(m):
            if i==0:#this is the biggest for our meshes TODO change this for new data
                n.apply_scale(1/10)
                n.export('./'+new_name+"_scaled_split_"+str(i)+".stl")
    else:
        m.apply_scale(1/10)
        m.export('./'+new_name+"_scaled.stl")
def rename_stl_files(): 
    collision_counter=dict()
    stl_files = glob.glob("*.stl");c=0
    namelist=[]
    for stl_file in stl_files: 
        if "conv_new" in stl_file:
            print(c)
            c+=1
            prefix = categorize_vessel(stl_file) 
            assert prefix != "unknown"
            new_name = f"{prefix}.brain.{stl_file.split('_', 1)[1].replace('_conv_new.stl', '')}" 
            if new_name not in collision_counter:
                collision_counter[new_name]=0
            collision_counter[new_name]+=1
            new_name=new_name+"_"+str(collision_counter[new_name])
            namelist.append(new_name)
            print(f"Old Name: {stl_file} -> New Name: {new_name}")
    print("num renames",len(namelist),"uniques:",len(set(namelist)))
def rewrite_stl_files(): 
    collision_counter=dict()
    stl_files = glob.glob("*.stl");c=0
    namelist=[]
    for stl_file in stl_files: 
        if "conv_new" in stl_file:
            print(c)
            c+=1
            prefix = categorize_vessel(stl_file) 
            assert prefix != "unknown"
            new_name = f"{prefix}.brain.{stl_file.split('_', 1)[1].replace('_conv_new.stl', '')}" 
            if new_name not in collision_counter:
                collision_counter[new_name]=0
            collision_counter[new_name]+=1
            new_name=new_name+"_"+str(collision_counter[new_name])
            conv_mesh(stl_file,new_name)
def count_stl_files():
    stl_files = glob.glob("*.stl")
    c=0;cc=0
    for stl_file in stl_files: 
        if "conv_new" in stl_file:
            cc+=1
        c+=1
    print(c,cc)
#count_stl_files()
#rename_stl_files()
rewrite_stl_files()
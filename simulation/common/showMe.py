import trimesh
import pickle
import numpy
from PIL import Image
import sys, io
import numpy as np

spheres = []
# Daten direkt aus dem RAM über stdin lesen
to_show = pickle.load(sys.stdin.buffer)

showables = []
print("Shower called with", len(to_show), "inputs")
for element in to_show:
    color = None
    if type(element) == tuple and len(element) == 2:
        color = element[1]
        element = element[0]
    if type(element) == tuple and len(element) == 3:
        sphere_mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=2)
        colors = np.array(
            [element[1][0:3] + [255]] * len(sphere_mesh.vertices)
        )  # Rot mit voller Deckkraft
        sphere_mesh.visual.vertex_colors = colors
        sphere_mesh.apply_translation(element[0])
        spheres.append(sphere_mesh)
    if type(element) == list or type(element) == numpy.ndarray:
        try:
            pc = trimesh.PointCloud(element)
            if color:
                pc.visual.vertex_colors = color
            showables.append(pc)
        except Exception:
            print(element)
            print(Exception)
    elif type(element) == trimesh.voxel.base.VoxelGrid:
        print("Voxel det")
        showables.append(element)
    else:
        if color:
            try:
                element.visual.vertex_colors = color
            except Exception:
                print(Exception)
            try:
                element.visual.face_colors = color
            except Exception:
                print(Exception)
        showables.append(element)
try:
    scene = trimesh.Scene(showables)
    for mesh in spheres:
        scene.add_geometry(geometry=mesh)

    # automatic camera settings
    bounds = scene.bounds
    if bounds is not None:
        # Szenenausdehnung
        size = bounds[1] - bounds[0]
        max_dim = max(size)
        centroid = scene.centroid

        # cam far clipping as thrice dimension
        scene.camera.z_far = max_dim * 3

        # Position
        scene.set_camera(
            angles=np.deg2rad([45, 20, 0]),
            distance=max_dim * 1.5,
            center=centroid,  # Focus centre
        )
    else:
        print("No bounds available, using default camera")

        scene.camera.z_far = 10000

    scene.show()
except Exception as e:
    print("Error")
    print(e)

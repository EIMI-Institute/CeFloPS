import os
import numpy as np
import trimesh
import pickle
import subprocess
from PIL import Image
import io
from scipy.interpolate import interp1d
import sys


def pick_color(index):
    kelly_colors = dict(
        vivid_yellow=[255, 179, 0],
        strong_purple=[128, 62, 117],
        vivid_orange=[255, 104, 0],
        very_light_blue=[166, 189, 215],
        vivid_red=[193, 0, 32],
        grayish_yellow=[206, 162, 98],
        medium_gray=[129, 112, 102],
        # these aren't good for people with defective color vision:
        vivid_green=[0, 125, 52],
        strong_purplish_pink=[246, 118, 142],
        strong_blue=[0, 83, 138],
        strong_yellowish_pink=[255, 122, 92],
        strong_violet=[83, 55, 122],
        vivid_orange_yellow=[255, 142, 0],
        strong_purplish_red=[179, 40, 81],
        vivid_greenish_yellow=[244, 200, 0],
        strong_reddish_brown=[127, 24, 13],
        vivid_yellowish_green=[147, 170, 0],
        deep_yellowish_brown=[89, 51, 21],
        vivid_reddish_orange=[241, 58, 19],
        dark_olive_green=[35, 44, 22],
    )
    for i, key in enumerate(kelly_colors):
        if i == index:
            return kelly_colors[key]


def interpolate_points(points_dict, interval=1, extrapolate=False):
    # get timestamps and points
    timestamps = sorted(points_dict.keys())
    points = [points_dict[time] for time in timestamps]

    # transpose points to obtain separate lists
    x_points, y_points, z_points = zip(*points)

    # interpolation for axis
    if extrapolate:
        f_x = interp1d(
            timestamps,
            x_points,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
        f_y = interp1d(
            timestamps,
            y_points,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
        f_z = interp1d(
            timestamps,
            z_points,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
    else:
        f_x = interp1d(
            timestamps,
            x_points,
            kind="linear",
            fill_value=(x_points[0], x_points[-1]),
            bounds_error=False,
        )
        f_y = interp1d(
            timestamps,
            y_points,
            kind="linear",
            fill_value=(y_points[0], y_points[-1]),
            bounds_error=False,
        )
        f_z = interp1d(
            timestamps,
            z_points,
            kind="linear",
            fill_value=(z_points[0], z_points[-1]),
            bounds_error=False,
        )

    # Create new dictionary with regular intervals
    interpolated_points_dict = {}
    start_time = int(min(timestamps)) if extrapolate else 0
    end_time = int(max(timestamps)) + 1
    for t in range(start_time, end_time, interval):
        interpolated_points_dict[t] = (f_x(t), f_y(t), f_z(t))

    return interpolated_points_dict


def get_color(name, i):
    r = ord(name[len(name) // 4])
    g = ord(name[len(name) // 3])
    b = ord(name[len(name) // 2])
    r = r * (3 * i) % 255
    g = g * (3 * i) % 255
    b = b * (3 * i) % 255
    return [r, g, b, 100]


def show(showable_objects):
    cmd = [
        sys.executable,
        r"C:\Users\User\Documents\micreduced\mic_vesselgraph\CeFloPS\simulation\common\showMe.py",
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,  # Daten über stdin senden
        stdout=subprocess.PIPE,  # stdout des Kindprozesses erfassen
        stderr=subprocess.PIPE,  # stderr ebenfalls erfassen
        text=False,  # Rohdaten (Bytes) statt Strings
    )

    # Daten pickeln und an den Prozess senden, stdout/stderr empfangen
    stdout_data, stderr_data = process.communicate(input=pickle.dumps(showable_objects))

    # Ausgabe dekodieren und anzeigen
    print("STDOUT:", stdout_data.decode("utf-8"))
    if stderr_data:
        print("STDERR:", stderr_data.decode("utf-8"))


def img(showable_objects, no_cam_trans=False):
    to_show = showable_objects
    showables = []
    look_at = []
    spheres = []
    for element in to_show:
        color = None
        if type(element) == tuple and len(element) == 2:
            color = element[1]
            element = element[0]
        if type(element) == tuple and len(element) == 3:
            sphere_mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=2)

            sphere_mesh.apply_translation(element[0])
            look_at.append(element[0])
            spheres.append(sphere_mesh)
        if type(element) == list or type(element) == np.ndarray:
            try:
                pc = trimesh.PointCloud(element)
                if color:
                    pc.visual.vertex_colors = color
                showables.append(pc)
            except Exception:
                print(element)

                print(Exception)
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

        # scene.camera.distance = 300
        """ scene.set_camera(angles=(0, 0, 0), distance=200, center=(100, 300, 0))
        scene.camera.z_far = 4000
        scene.camera.look_at(
            [[149, 182, 244], [149, 82, 244]],
            rotation=trimesh.transformations.rotation_matrix(180, [200, 100, 100]),
        ) """
        if not no_cam_trans:
            scene.set_camera(angles=(0.7, 0, 90), distance=300, center=(100, 300, 0))
        scene.camera.z_far = 4000
        if len(look_at) == 0:
            look_at = [[449, -8982, 444]]
            print("setlook")
        scene.camera.look_at(
            look_at,
            # rotation=trimesh.transformations.rotation_matrix(180, [1, 0, 0]),
        )
        return scene.save_image(resolution=(1080, 1080))

    except Exception as e:
        print(e)
    return


def img_scene(showable_objects):
    to_show = showable_objects
    showables = []
    for element in to_show:
        color = None
        if type(element) == tuple:
            color = element[1]
            element = element[0]
        # if(valid(element)):
        if type(element) == list or type(element) == np.ndarray:
            try:
                pc = trimesh.PointCloud(element)
                if color:
                    pc.visual.vertex_colors = color
                showables.append(pc)
            except Exception:
                print(element)

                print(Exception)
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
        # scene.camera.distance = 300
        """ scene.set_camera(angles=(0, 0, 0), distance=200, center=(100, 300, 0))
        scene.camera.z_far = 4000
        scene.camera.look_at(
            [[149, 182, 244], [149, 82, 244]],
            rotation=trimesh.transformations.rotation_matrix(180, [200, 100, 100]),
        ) """

        scene.set_camera(angles=(1, 0, 0), distance=2100)  # )), center=(100, 300, 0))
        scene.camera.z_far = 4000
        # Get the current camera transform
        camera_transform = scene.camera_transform.copy()

        # Define the translation vector to move the camera up
        # For example, to move the camera 100 units up along the y-axis
        translation_vector = [0, 270, 0]

        # Apply the translation to the camera transform
        camera_transform[:3, 3] += translation_vector

        # Update the scene with the new camera transform
        scene.camera_transform = camera_transform
        """ scene.camera.look_at(
            [[149, 182, 244], [149, 82, 244]],
            rotation=trimesh.transformations.rotation_matrix(180, [200, 100, 100]),
        ) """
        return scene

    except Exception as e:
        print(e)
    return


def save(showable_objects, filep):
    image = img(showable_objects)
    # creating a image object (main image)
    im1 = Image.open(io.BytesIO(image))
    # save a image using extension
    im1 = im1.save(filep)

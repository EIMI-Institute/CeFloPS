from .visualization import img_scene, show
import numpy as np
import io
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib


def show_pointsequence(path, color1, color2):
    # create pointcloud and make each point that closer to the end closer in color to the second color in gradient
    number_colors = len(path)
    fraction = 0
    last_point_index = 0
    result = [(color2[i] - color1[i]) * fraction + color1[i] for i in range(3)]
    resultn = result
    print(resultn)
    next_index = 0
    to_show = []
    notfirst = False
    while next_index < number_colors:
        while resultn == result:
            next_index += 1
            fraction += 1 / number_colors
            resultn = [(color2[i] - color1[i]) * fraction + color1[i] for i in range(3)]
        print(resultn)
        to_show.append((path[last_point_index : next_index + 1], result))
        last_point_index = next_index + 1
        result = [(color2[i] - color1[i]) * fraction + color1[i] for i in range(3)]

    show(to_show)


def show_pointsequence_labeled(points, val_labels):
    assert len(points) == len(val_labels)
    assert all([type(v) == int for v in val_labels])
    norm = matplotlib.colors.Normalize(vmin=min(val_labels), vmax=max(val_labels))
    cmap = plt.cm.get_cmap("viridis", 1000)
    # cmap(norm(2))#(0.269944, 0.014625, 0.341379, 1.0)
    # create pointcloud and make each point have a color according to the values
    to_show = []
    for i, p in enumerate(points):
        color = cmap(norm(val_labels[i]))[0:3]
        to_show.append(([p], color))
    show(to_show)

    """ ------------------------------------------------
        coordinate in bodyscene rendering
    
    """


def interpolate_points(points_dict, interval=1, extrapolate=False):
    """interpolate_points Takes a dict with time as key and points as values and interpolates points to fit a given timeinterval for new keys

    Args:
        points_dict (_type_): _description_
        interval (int, optional): _description_. Defaults to 1.
        extrapolate (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Extract times and corresponding points
    times = sorted(points_dict.keys())
    points = [points_dict[time] for time in times]

    # separate lists for x, y, and z
    x_points, y_points, z_points = zip(*points)

    # interpolation functions for x, y, and z
    if extrapolate:
        f_x = interp1d(
            times, x_points, kind="linear", fill_value="extrapolate", bounds_error=False
        )
        f_y = interp1d(
            times, y_points, kind="linear", fill_value="extrapolate", bounds_error=False
        )
        f_z = interp1d(
            times, z_points, kind="linear", fill_value="extrapolate", bounds_error=False
        )
    else:
        f_x = interp1d(
            times,
            x_points,
            kind="linear",
            fill_value=(x_points[0], x_points[-1]),
            bounds_error=False,
        )
        f_y = interp1d(
            times,
            y_points,
            kind="linear",
            fill_value=(y_points[0], y_points[-1]),
            bounds_error=False,
        )
        f_z = interp1d(
            times,
            z_points,
            kind="linear",
            fill_value=(z_points[0], z_points[-1]),
            bounds_error=False,
        )

    # new dictionary with regular intervals
    interpolated_points_dict = {}
    start_time = int(min(times)) if extrapolate else 0
    end_time = int(max(times)) + 1
    for t in range(start_time, end_time, interval):
        interpolated_points_dict[t] = (f_x(t), f_y(t), f_z(t))

    return interpolated_points_dict


def get_frame(scene):
    image = scene.save_image(resolution=(1080, 1080))
    frame = Image.open(io.BytesIO(image))
    frame = frame.convert("RGB")
    return frame


def add_spheres_to_scene(scene, pointlist, radius=1.0):
    sphere_keys = []
    colors = [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 165, 0],  # Orange
        [128, 0, 128],  # Purple
    ]
    colormap = plt.cm.get_cmap("viridis", len(pointlist))
    for i, point in enumerate(pointlist):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        # print(vis.pick_color(i)+[250])
        sphere.apply_translation(point)

        color = colors[i]
        sphere.visual.vertex_colors = np.tile(color, (len(sphere.vertices), 1))
        """ color = colormap(i)[:3]  # Get RGB values, ignore alpha
        color = np.array(color) * 255  # Scale color values to 0-255
        color = color.astype(np.uint8)  # Convert to integers
        sphere.visual.vertex_colors = np.tile(color, (len(sphere.vertices), 1)) """
        key = scene.add_geometry(sphere)
        sphere_keys.append(key)
    return sphere_keys, scene


def remove_spheres_from_scene(scene, sphere_keys):
    print(sphere_keys)
    for key in sphere_keys:
        # Use the key to remove the geometry from the scene
        if key in scene.geometry:
            scene.delete_geometry(key)
            print("del")


def load_points_from_file(file_path):
    points_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            # Split the line by spaces and convert to float
            x, y, z, time = map(float, line.split()[0:4])
            # Use time as the key and a tuple (x, y, z) as the value
            points_dict[time] = (x, y, z)
    return points_dict


import time


def coordinates_to_images(vessel_paths, coordinates_dicts, limit=None):
    maxes = [len(coordinates_dicts[k]) for k in range(len(coordinates_dicts))]
    maxdictlen = max(maxes)
    frames = []
    x = 0
    step = 1
    if limit == None:
        limit = maxdictlen
    else:
        limit = min([maxdictlen, limit])

    # cap=len(scene.geometry)
    while x < limit:
        scene = img_scene(vessel_paths)
        points = [
            (
                np.asarray(coordinates_dicts[k][x]).reshape(3)
                if x in coordinates_dicts[k]
                else np.asarray(coordinates_dicts[k][maxes[k] - 1]).reshape(3)
            )
            for k in range(len(coordinates_dicts))
        ]

        print(len(scene.geometry))
        spheres, scene = add_spheres_to_scene(scene, points, radius=10)
        print("added", len(scene.geometry), spheres)

        frame = get_frame(scene)
        frames.append(frame)
        """ delkeys=[]
        for i,key in enumerate(scene.geometry.keys()):
            if i >= cap: delkeys.append(key)
        for key in delkeys:
            scene.delete_geometry(key) 
        time.sleep(0.01) """

        # del scene.geometry[key]
        # scene =remove_spheres_from_scene(imag_scene, spheres)
        print("rem", len(scene.geometry))

        # frame =get_frame(imag_scene)
        # frames.append(frame)

        """ if x>=300:
            step=5
        if x>=900:
            step=15 """

        x += step
        print("frame: " + str(x))
    return frames


# Assuming `imgs` is your list of PIL.Image.Image objects
# imgs = [...]  # Your list of PIL.Image.Image objects


class ImageSliderApp:
    def __init__(self, root, image_files, opt=[], scale=1):
        self.root = root
        self.root.title("Image Viewer")

        # Initialize the list to hold the PhotoImage references
        self.photo_images = []

        # Convert the PIL images to ImageTk.PhotoImage objects for display
        for image in image_files:
            # Resize the image to be 4 times bigger
            original_size = image.size
            new_size = (original_size[0] * scale, original_size[1] * scale)
            """ new_size = (original_size[0] * scale, original_size[1] * scale)
            resized_image = image.resize(new_size)  # ), Image.ANTIALIAS) """
            resized_image = image.resize((1920, 1080))  # , Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(resized_image)
            self.photo_images.append(photo)  # Keep a reference
        for image in opt:
            # Resize the image to be 4 times bigger
            original_size = image.size
            new_size = (original_size[0] * scale, original_size[1] * scale)
            """ new_size = (original_size[0] * scale, original_size[1] * scale)
            resized_image = image.resize(new_size)  # ), Image.ANTIALIAS) """
            resized_image = image.resize((1920, 1080), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(resized_image)
        print(self.photo_images[0])
        # Set up the image display label with the first image
        self.image_label = tk.Label(self.root, image=self.photo_images[0])

        self.image_label.pack()

        # Set up the slider
        self.slider = ttk.Scale(
            self.root,
            from_=0,
            to=len(self.photo_images) - 1,
            orient="horizontal",
            command=self.update_image,
        )
        self.slider.pack()

        # Set up the image number display
        self.image_number_label = tk.Label(
            self.root, text=f"Image 1 of {len(self.photo_images)}"
        )
        self.image_number_label.pack()

    def update_image(self, index):
        index = int(float(index))  # Convert string or float to integer
        # Update the image label with the selected image
        self.image_label.config(image=self.photo_images[index])
        # Update the image number label
        self.image_number_label.config(
            text=f"Image {index + 1} of {len(self.photo_images)}"
        )
        # Keep a reference to the image to prevent garbage collection
        self.image_label.image = self.photo_images[index]


def slide_imgs(frames, scale=1):
    root = tk.Tk()
    app = ImageSliderApp(root, frames, scale=scale)
    root.mainloop()

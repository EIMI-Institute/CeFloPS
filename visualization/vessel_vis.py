import glob
import re
import numpy as np
import trimesh
from CeFloPS.simulation.common.vessel_functions import (
    find_connected_structures,
    get_traversion_regions,
    same_counts,
    endpoint_pair_relation,
    create_connection,
    closest_point_in_path_index,
    apply_links,
    get_vessel_endpoints_per_mesh,
    generate_directions,
)
from .visualization import pick_color, show, img_scene
from CeFloPS.simulation.common.functions import calculate_distance_euclidian
import os
import networkx as nx
import matplotlib.pyplot as plt


def show_highlighted(
    vessels,
    highlighted_vessels,
    submesh_not_path=False,
    show_link_kind=False,
    names_to_colors=None,
):
    """Visualize vessels connections, optionally highlighting some vessels

    Args:
        vessels (_type_): _description_
        highlighted_vessels (_type_): _description_
        submesh_not_path (bool, optional): _description_. Defaults to False.
        show_link_kind (bool, optional): _description_. Defaults to False.
        names_to_colors (_type_, optional): _description_. Defaults to None.
    """
    mesh_color = [100, 100, 100, 50]
    path_point_color = [100, 100, 100, 50]
    highlight_color = [255, 0, 0, 250]

    associated_meshes = []
    vesselarray = vessels
    connected = highlighted_vessels
    to_show = []
    for o, part in enumerate(connected):
        color_i = o % 20
        special = False
        if names_to_colors != None:
            for vessel in part:
                if "jump_vessel_left_chamber" in vessel.associated_vesselname:
                    color_i = 4
                    special = True
                if "jump_vessel_right_chamber" in vessel.associated_vesselname:
                    color_i = 9
                    special = True
            if special == False:
                if color_i == 9 or color_i == 4:
                    color_i = (color_i - 6) % 20

        for vessel in part:
            if submesh_not_path:
                if hasattr(vessel, "submesh"):
                    associated_meshes.append(
                        (vessel.submesh, pick_color(color_i) + [170])
                    )
                else:
                    associated_meshes.append((vessel.path, pick_color(color_i) + [170]))
            else:
                associated_meshes.append((vessel.path, pick_color(color_i) + [170]))

    to_show = associated_meshes

    picker = {}
    picker["abc 2 connect to d"] = 1
    picker["ac 2 connect to biggetr between b and d"] = 2
    picker["ac connect to b"] = 3
    picker["ab connect to c"] = 4
    picker["abc 2 connect to d"] = 5
    picker["nur a zu iwem connect to closest biggest"] = 6
    picker["normalh"] = 7
    picker["normalz"] = 8
    picker["nur a zu iwem connect to closest biggest 2"] = 9
    picker["None"] = 10

    for vessel in vesselarray:
        if submesh_not_path:
            to_show.append((vessel.path, path_point_color))
        for link in vessel.links_to_path:  # fuegt die doppelt hinzu, ist visuell egal
            if (
                calculate_distance_euclidian(
                    vessel.path[link.source_index],
                    link.target_vessel.path[link.target_index],
                )
                > 0
            ):
                showable_path = trimesh.load_path(
                    [
                        vessel.path[link.source_index],
                        link.target_vessel.path[link.target_index],
                        # link.target_vessel.path[link.target_vessel.lookup_index(vessel)],
                    ]
                )
                # print("linktag", link.tag, vessel.associated_vesselname)
                if link.tag == None:
                    link.tag = "None"
                showable_path.colors = [
                    pick_color(picker[link.tag]) + [150]
                ]  # [[244, 25, 25, 150]]
                to_show.append(showable_path)
                # pick highlight color for link type:
                if show_link_kind:
                    if len(link) == 3:
                        print(link)
                    print("------------------choosing color for  ", link.tag)
                    highlight_color = pick_color(picker[link.tag])
                    to_show.extend(
                        (
                            [
                                vessel.path[link.source_index],
                                link.target_vessel.path[link.target_index],
                            ],
                            highlight_color,
                        )
                        for link in vessel.links_to_path
                    )
    showables = []
    for element in to_show:
        color = None
        if type(element) == tuple:
            color = element[1]
            element = element[0]
        # if(valid(element)):
        if type(element) == list:
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
    scene = trimesh.Scene(showables)
    show(to_show)


def show_linked(vessels, submesh_not_path):
    """Show connected structures

    Args:
        vessels (_type_): _description_
        submesh_not_path (_type_): _description_
    """
    show_highlighted(
        vessels,
        find_connected_structures(vessels),
        submesh_not_path=submesh_not_path,
        names_to_colors=2,
    )


def show_trav(vessels, submesh_not_path):
    """Show traversable structures

    Args:
        vessels (_type_): _description_
        submesh_not_path (_type_): _description_
    """
    show_highlighted(
        vessels, get_traversion_regions(vessels), submesh_not_path=submesh_not_path
    )


def links_at(vessel, index):
    out = []
    for link in vessel.links_to_path:
        if link.source_index == index:
            out.append(link)
    return out


def get_traversable_links(new_vessel, all_highest_links):
    """get_traversable_links return a list of links that are eligible for traversion

    Args:
        new_vessel (_type_): _description_

    Returns:
        _type_: _description_
    """
    if new_vessel.type == "vein":
        if all_highest_links:
            return links_at(new_vessel, len(new_vessel.path) - 1)
        try:
            if new_vessel.highest_link() != None:
                return [new_vessel.highest_link()]
            return []
        except:
            print("vein has no links or wrong direction")
            return []
    else:
        return new_vessel.next_links(0)


def create_graph_from_vessels(vessels, vesselnames=False):
    G = nx.DiGraph()
    for i, vessel in enumerate(vessels):
        if vesselnames:
            G.add_node(f"{vessel.id[0:17]}{vessel.associated_vesselname}")
        else:
            G.add_node(vessel.id)
        for link in get_traversable_links(vessel):  # .links_to_path:#
            if vesselnames:
                G.add_edge(
                    f"{vessel.id[0:17]}{vessel.associated_vesselname}",
                    f"{link.target_vessel.id[0:17]}{link.target_vessel.associated_vesselname}",
                )
            else:
                G.add_edge(f"{vessel.id}", f"{link.target_vessel.id}")
    return G


def save_graph(vessels, gpath):
    """create and save vesselgraph

    Args:
        vessels (_type_): _description_
        gpath (_type_): _description_
    """
    G = create_graph_from_vessels(vessels)
    pos = nx.nx_agraph.graphviz_layout(G)
    fig = plt.figure(1, figsize=(160, 90), dpi=60)
    nx.draw_networkx(G, pos, node_size=100, font_size=10, arrowsize=25)
    fig.savefig("" + gpath, bbox_inches="tight")
    plt.clf()


def create_output(v):
    """Create textual summary of connected and traversable regions in given vessels

    Args:
        v (_type_): _description_
    """
    travel_connected = get_traversion_regions(v)
    link_connected = find_connected_structures(v)
    link_connected_c = [len(con) for con in link_connected]
    travel_connected_c = [len(con) for con in travel_connected]
    kk = False
    print(travel_connected_c)
    if len(link_connected_c) != 1:
        print(f"Nicht komplett verbunden: {v[0].associated_vesselname} \n")
        link_connected_c.sort()
        print(f"connected_parts:{ link_connected_c} \n")
        kk = True

    if not same_counts(link_connected_c, travel_connected_c):
        print(f">>>>>>{v[0].associated_vesselname} nicht alle links traversierbar \n")
        print(f"connected_parts: {link_connected_c} \n")
        print(f"traversable_parts: {travel_connected_c} \n")
        kk = True
    if kk:
        print("------ \n")


def show_vessel(
    vesselarray,
    associated_meshes,
    mesh_color=None,
    show_paths_as_path=False,
    all_links_incl_eligible=True,
    show_paths=True,
    show_path_links_as_paths=True,
    highlight_link_indices=True,
    path_point_color=None,
    highlight_color=None,
):
    """Function to show vessels and their associated meshes, highlights link by default"""
    if mesh_color is None:
        mesh_color = [100, 100, 100, 50]
    if path_point_color is None:
        path_point_color = [100, 100, 100, 50]
    if highlight_color is None:
        highlight_color = [255, 0, 0, 250]
    to_show = [(mesh, mesh_color) for mesh in associated_meshes]
    for vessel in vesselarray:
        if show_paths_as_path:
            to_show.append(trimesh.load_path(vessel.path))
        if show_paths:
            to_show.append((vessel.path, path_point_color))
        if highlight_link_indices:
            to_show.extend(
                ([vessel.path[link.source_index]], highlight_color)
                for link in vessel.links_to_path
            )

        if show_path_links_as_paths:
            for (
                link
            ) in vessel.links_to_path:  # fuegt die doppelt hinzu, ist visuell egal
                showable_path = trimesh.load_path(
                    [
                        vessel.path[link.source_index],
                        link.target_vessel.path[link.target_index],
                    ]
                )

                showable_path.colors = [[244, 25, 25, 150]]
                if True:  # vessel.eligible_link(link) or all_links_incl_eligible:
                    to_show.append(showable_path)
        """ if vessel.richtung == "nan":
            # to_show.append((vessel.submesh,[0,200,0,190]))
            ...
        if vessel.richtung == "ASC":
            # to_show.append(([vessel.path[len(vessel.path)-1]],[250,0,0,250]))
            ...
        if vessel.richtung == "DESC":
            # to_show.append(([vessel.path[0]],[250,0,0,250]))
            ... """
    show(to_show)


import re


def show_travel_route(
    travel_return, paths_to_stl_files=[], color=[200, 255, 10, 200], vessel_paths=None
):
    """Show STLs and a pointcloud path together

    Args:
        travel_return (_type_): _description_
        paths_to_stl_files (list, optional): _description_. Defaults to [].
        color (list, optional): _description_. Defaults to [200, 255, 10, 200].
        vessel_paths (_type_, optional): _description_. Defaults to None.
    """
    to_show = [(travel_return[0], color)]
    to_show.append(([travel_return[0][0]], [255, 0, 0, 255]))
    organ_names = []
    print("Distanz: ", travel_return[1], "Zeit: ", travel_return[2])
    for name in glob.glob(str("../*organ*")):
        stlRegex = re.compile(r".stl$")
        mo1 = stlRegex.search(name)
        if mo1 != None:
            # if("pulmonary" not in name):
            organ_names.append(name)
    print(organ_names)
    organs = []
    for name in organ_names:
        organs.append(trimesh.load(name))
    for organ in organs:
        to_show.append((organ, [100, 100, 100, 50]))
    if vessel_paths != None:
        for path in vessel_paths:
            if "ajsbgfksfg" not in path.associated_vesselname:
                to_show.append((path.path, [100, 100, 100, 50]))
            else:
                to_show.append((path.path, [50, 50, 100, 40]))
    for path in paths_to_stl_files:
        try:
            to_show.append((trimesh.load_mesh(path), [100, 100, 100, 100]))
        except Exception:
            print("couldnt load path")
    show(to_show)


def show_vessels(
    vesselarray,
    associated_meshes=[],
    point_clouds=[],
    mesh_color=None,
    show_paths_as_path=False,
    show_paths=True,
    show_path_links_as_paths=True,
    highlight_link_indices=True,
    path_point_color=None,
    highlight_color=None,
):
    if mesh_color is None:
        mesh_color = [100, 100, 100, 50]
    if path_point_color is None:
        path_point_color = [100, 100, 100, 50]
    if highlight_color is None:
        highlight_color = [255, 0, 0, 250]
    print(2)
    to_show = [(mesh, mesh_color) for mesh in associated_meshes]
    for vessel in vesselarray:
        if show_paths_as_path:
            to_show.append(trimesh.load_path(vessel.path))
        if show_paths:
            to_show.append((vessel.path, path_point_color))
        if highlight_link_indices:
            to_show.extend(
                (
                    [
                        vessel.path[link.source_index],
                        link.target_vessel.path[link.target_index],
                    ],
                    highlight_color,
                )
                for link in vessel.links_to_path
            )

        if show_path_links_as_paths:
            for (
                link
            ) in vessel.links_to_path:  # fuegt die doppelt hinzu, ist visuell egal
                showable_path = trimesh.load_path(
                    [
                        vessel.path[link.source_index],
                        link.target_vessel.path[link.target_index],
                        # link.target_vessel.path[link.target_vessel.lookup_index(vessel)],
                    ]
                )
                showable_path.colors = [[244, 25, 25, 150]]
                to_show.append(showable_path)
    for points in point_clouds:
        to_show.append(points)
    # print(to_show)
    # show(to_show)
    return img_scene(to_show)


plt.ioff()


def create_images(vessels, k, meshname):
    vessel_dict = {}
    for vessel in vessels:
        vessel_dict[vessel.id] = vessel
    """{1919205010992: <vessel_functions.Vessel at 0x28a6068c4c0>,
    1919213011104: <vessel_functions.Vessel at 0x28a2a4fc640>,
    1919213011296: <vessel_functions.Vessel at 0x28a2ad7fd60>,"""

    mesh_color = [100, 100, 100, 50]
    path_point_color = [100, 100, 100, 50]
    highlight_color = [255, 0, 0, 250]

    linkoperations = []  # [operationtarget, link]

    vessel_endpoints = []
    vessel_boundaries = []
    insert_index = -1
    for vessel in vessels:
        vessel_endpoints.extend((vessel.path[0], vessel.path[len(vessel.path) - 1]))
        vessel_boundaries.append([insert_index + 1, vessel])
        insert_index = insert_index + 2

    vessel_endpoints = np.asarray(vessel_endpoints)

    link_collect = [
        get_vessel_endpoints_per_mesh(
            vessels[i], i, vessel_endpoints, vessel_boundaries
        )
        for i in range(len(vessels))
    ]
    endpoint_dict = {}
    for endp_submesh_collection in link_collect:
        if len(endp_submesh_collection) > 1:
            for endpoints in endp_submesh_collection[1::]:
                if endpoints not in endpoint_dict:
                    endpoint_dict[endpoints] = []
                endpoint_dict[endpoints].append(endp_submesh_collection[0])
    """{(1919213011824, 42): [1919213011680],
    (1919213012400, 28): [1919213011680, 1919213015424],
    (1919213013888, 58): [1919213011680],...}"""

    connections = []
    count = 0
    for endpointkey in endpoint_dict:
        best_vessel_to_connect = None
        if len(endpoint_dict[endpointkey]) > 1:
            if len(endpoint_dict[endpointkey]) == 2:
                # in 2 anderen meshes -> 64 moegliche in B,C relationen (a in b, a in c, b in c, b in a, c in a, c in b)
                # wenn a in b und b in a , auch fuer c-> dann seien diese verbunden und alle dazwichen werden geloescht
                a = vessel_dict[endpointkey[0]]
                b = vessel_dict[endpoint_dict[endpointkey][0]]
                c = vessel_dict[endpoint_dict[endpointkey][1]]
                if endpoint_pair_relation(endpointkey, b, vessel_dict)[0]:
                    count += 1
                    if endpoint_pair_relation(endpointkey, c, vessel_dict)[0]:
                        print("a-b-c")
                        assert 1 == 2, "das wird nicht verwendet"
                    else:
                        print("ab connect to c")
                        create_connection(
                            c, endpointkey, vessel_dict, connections, "ab connect to c"
                        )
                        continue

                elif endpoint_pair_relation(endpointkey, c, vessel_dict)[0]:  # nur a-c
                    count += 1
                    print("ac connect to b")
                    create_connection(
                        b, endpointkey, vessel_dict, connections, "ac connect to b"
                    )
                    continue
                else:
                    count += 1
                    print("nur a zu iwem connect to closest biggest")
                    highest_volume_vessel = b
                    if c.volume > highest_volume_vessel.volume:
                        highest_volume_vessel = c
                    create_connection(
                        highest_volume_vessel,
                        endpointkey,
                        vessel_dict,
                        connections,
                        "nur a zu iwem connect to closest biggest",
                    )
                    continue

            if len(endpoint_dict[endpointkey]) == 3:
                # in 2 anderen meshes -> 128 moegliche relationen von endpunkt in mesh
                a = vessel_dict[endpointkey[0]]
                b = vessel_dict[endpoint_dict[endpointkey][0]]
                c = vessel_dict[endpoint_dict[endpointkey][1]]
                d = vessel_dict[endpoint_dict[endpointkey][2]]
                if endpoint_pair_relation(endpointkey, b, vessel_dict)[0]:
                    if endpoint_pair_relation(endpointkey, b, vessel_dict)[0]:
                        if endpoint_pair_relation(endpointkey, c, vessel_dict)[0]:
                            if endpoint_pair_relation(endpointkey, d, vessel_dict)[0]:
                                print("abcd 2")
                                assert 1 == 2, "das hier wird nicht gebraucht"
                                continue
                        print("abc 2 connect to d")  # TODO 2
                        count += 1
                        create_connection(
                            d,
                            endpointkey,
                            vessel_dict,
                            connections,
                            "abc 2 connect to d",
                        )
                        continue
                    print("ab 2")
                    assert 1 == 2, "das hier wird nicht gebraucht"
                    continue
                if endpoint_pair_relation(endpointkey, c, vessel_dict)[0]:
                    if endpoint_pair_relation(endpointkey, d, vessel_dict)[0]:
                        print("acd 2")
                        assert 1 == 2, "das hier wird nicht gebraucht"
                        continue
                    print("ac 2 connect to biggetr between b and d")  # TODO 1
                    highest_volume_vessel = b
                    if d.volume > highest_volume_vessel.volume:
                        highest_volume_vessel = d
                    create_connection(
                        highest_volume_vessel,
                        endpointkey,
                        vessel_dict,
                        connections,
                        "ac 2 connect to biggetr between b and d",
                    )
                    continue
                if endpoint_pair_relation(endpointkey, d, vessel_dict)[0]:
                    print("ad 2")
                    assert 1 == 2, "das hier wird nicht gebraucht"
                    continue

                else:
                    count += 1
                    print("nur a zu iwem connect to closest biggest 2")
                    highest_volume_vessel = b
                    if c.volume > highest_volume_vessel.volume:
                        highest_volume_vessel = c
                    create_connection(
                        highest_volume_vessel,
                        endpointkey,
                        vessel_dict,
                        connections,
                        "nur a zu iwem connect to closest biggest 2",
                    )
                    continue

        else:
            count += 1
            best_vessel_to_connect = vessel_dict[endpoint_dict[endpointkey][0]]
            nearest_connection_point_index = closest_point_in_path_index(
                best_vessel_to_connect, vessel_dict[endpointkey[0]].path[endpointkey[1]]
            )
            connections.append(
                (
                    vessel_dict[endpointkey[0]],
                    [
                        endpointkey[1],
                        best_vessel_to_connect,
                        nearest_connection_point_index,
                        "normalh",
                    ],
                    "normal h",
                )
            )  # von ende zu naechster vessel
            connections.append(
                (
                    best_vessel_to_connect,
                    [
                        nearest_connection_point_index,
                        vessel_dict[endpointkey[0]],
                        endpointkey[1],
                        "normalz",
                    ],
                    "normal z",
                )
            )  # zurueck

    # print("connected",count,"of",len(endpoint_dict))

    apply_links(connections)

    # shorten_ends(vessels)

    connected = find_connected_structures(vessels)
    to_show = []
    try:
        os.mkdir("./validation_pics")
    except Exception:
        ...
    for i, structure in enumerate(connected):
        for vessel in structure:
            col = pick_color(i)
            col.append(200)
            to_show.append((vessel.submesh, col))
    scenestuff = []
    for element in to_show:
        colored = element[0]
        colored.visual.face_colors = element[1]
        scenestuff.append(colored)
    scene = trimesh.Scene(scenestuff)
    image = scene.save_image()

    with open(
        f"./validation_pics/connected_vessels_{meshname}.png", "wb"
    ) as binary_file:
        binary_file.write(image)

    # print("LINKS: ",[vessels[i].links_to_path for i in range(len(vessels))])

    associated_meshes = []
    vesselarray = vessels
    for vessel in vesselarray:
        associated_meshes.append(vessel.submesh)
    to_show = [(mesh, mesh_color) for mesh in associated_meshes]
    for vessel in vesselarray:
        to_show.append((vessel.path, path_point_color))
        for link in vessel.links_to_path:  # fuegt die doppelt hinzu, ist visuell egal
            showable_path = trimesh.load_path(
                [
                    vessel.path[link[0]],
                    link[1].path[link[2]],
                    # link[1].path[link[1].lookup_index(vessel)],
                ]
            )
            showable_path.colors = [[244, 25, 25, 150]]
            to_show.append(showable_path)
            # pick highlight color for link type:
            picker = {}
            picker["abc 2 connect to d"] = 1
            picker["ac 2 connect to biggetr between b and d"] = 2
            picker["ac connect to b"] = 3
            picker["ab connect to c"] = 4
            picker["abc 2 connect to d"] = 5
            picker["nur a zu iwem connect to closest biggest"] = 6
            picker["normalh"] = 7
            picker["normalz"] = 8
            picker["nur a zu iwem connect to closest biggest 2"] = 9
            if len(link) == 3:
                print(link)
            print("------------------choosing color for  ", link[3])
            highlight_color = pick_color(picker[link[3]])
            to_show.extend(
                ([vessel.path[link[0]], link[1].path[link[2]]], highlight_color)
                for link in vessel.links_to_path
            )
    showables = []
    for element in to_show:
        color = None
        if type(element) == tuple:
            color = element[1]
            element = element[0]
        # if(valid(element)):
        if type(element) == list:
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
    scene = trimesh.Scene(showables)
    image = scene.save_image()

    with open(
        f"./validation_pics/connected_vessels_{meshname}_links.png", "wb"
    ) as binary_file:
        binary_file.write(image)

    generate_directions(vessels)

    save_graph(
        vessels, f"./validation_pics/connected_vessels_{meshname}_directions.png"
    )

    generate_directions(vessels, reversed_vein_method=False)

    save_graph(
        vessels,
        f"./validation_pics/connected_vessels_{meshname}_directions_Artery_method.png",
    )

    with open("output.txt", "a") as f:
        ...
        """
            for v in vessels:
                create_connections(v)
                generate_directions(v,startvessels=None, reversed_vein_method=True)

                changed_directions=True
                while(changed_directions):
                    trav=get_traversion_regions(v)
                    change_directions(v)
                    if(len(get_traversion_regions(v))==len(trav)):
                        changed_directions=False
                travel_connected=get_traversion_regions(v)
                link_connected=find_connected_structures(v)
                link_connected_c=[len(con) for con in link_connected]
                travel_connected_c=[len(con) for con in travel_connected]
                kk=False
                if(len(link_connected_c)!=1):
                    f.write(f"Nicht komplett verbunden: {v[0].associated_vesselname} \n" )
                    link_connected_c.sort()
                    f.write(f"connected_parts:{ link_connected_c} \n")
                    kk=True

                if(not same_counts(link_connected_c, travel_connected_c)):
                    f.write(f">>>>>>{v[0].associated_vesselname} nicht alle links traversierbar \n" )
                    f.write(f"connected_parts: {link_connected_c} \n")
                    f.write(f"traversable_parts: {travel_connected_c} \n")
                    kk=True
                if(kk):
                    f.write("------ \n") """

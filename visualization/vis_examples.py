# EXAMPLES


def create_example(
    meshname="../vl.blood_vessels.pulmonary_veins.right_pulmonary_veins.stl",
):
    meshname = meshname
    mesh = trimesh.load_mesh(meshname)
    submeshes = mesh.split()

    selected_submesh = submeshes[0]
    for submesh in submeshes:
        if submesh.volume > selected_submesh.volume:
            selected_submesh = submesh
    # check for watertightness
    for submesh in submeshes:
        assert submesh.is_watertight, "One Submesh is not watertight, aborting"
    print("calculating paths", " for ", meshname)
    path_path_sel = create_guide_points(
        [selected_submesh], 0, meshname, 4.5, demonstration=True
    )
    """ #paths=center_paths(opaths,submeshes)
    print("calculating normals"" for ", meshname)
    normals=get_normals(paths,submeshes)
    print("cutting submeshes"" for ", meshname)
    cuts_in_mesh=[]
    for si, submesh in enumerate(submeshes):
        cuts_in_mesh.append([])
        cuts_in_mesh[si]=create_cuts(submesh,paths[si], normals[si], meshname)
    print("creating vessels"" for ", meshname) 
    create_vessels(cuts_in_mesh,meshname,submeshes,paths)"""

    normals = get_normals(path_path_sel, [selected_submesh], 1)
    normals[1]
    cuts = create_cuts(selected_submesh, path_path_sel[0], normals[0], meshname, 1)
    len(path_path_sel)
    len(normals[0])
    print(cuts[0])
    cuts
    # show normals
    to_show = []
    to_show.append((selected_submesh, [100, 100, 100, 50]))

    for cut in cuts:
        to_show.append((cut[0], [255, 104, 0, 100]))

    to_show.append((path_path_sel, [166, 189, 215, 250]))

    show(to_show)
    # show normals
    to_show = []
    to_show.append((selected_submesh, [100, 100, 100, 50]))

    # normals = get_normals([gp],[selected_submesh])
    nps = []
    for i, point in enumerate(gp):
        if i < len(normals[0]):
            showable_path = trimesh.load_path(
                [
                    np.asarray(point),
                    np.asarray(point) + np.asarray(normals[0][i]),
                    # link[1].path[link[1].lookup_index(vessel)],
                ]
            )
            showable_path.colors = [[255, 104, 0, 250]]
            to_show.append(showable_path)

    to_show.append((gp, [166, 189, 215, 250]))

    show(to_show)

    # show endpoints
    shape = {i for sub in selected_submesh.faces for i in sub}

    meshneighbours = read_neighbours(
        selected_submesh.faces, len(selected_submesh.vertices)
    )
    startindices = highestNeighbourCount(shape, meshneighbours)
    startpoints = [selected_submesh.vertices[startindex] for startindex in startindices]
    to_show = []
    to_show.append((selected_submesh, [100, 100, 100, 50]))
    to_show.append((startpoints, [250, 100, 100, 250]))

    show(to_show)
    # show guidepoints
    to_show = []
    to_show.append((selected_submesh, [100, 100, 100, 50]))
    for i in range(len(gps)):
        if i < len(gp) - 1:
            to_show.append(([gp[i]], [100, 100, 250, 250]))
            to_show.append((gps[i], [250, (127.5 * i) % 255, 100, 50]))
            to_show.append(([gp[i + 1]], [100, 250, 100, 250]))

    show(to_show)


def create_images(mesh_names, k):
    meshname = mesh_names[k][3::]
    vessels = []
    vessels = process_mesh(mesh_names[k])
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

    print("connected", count, "of", len(endpoint_dict))

    apply_links(connections)

    print(vessels[0].links_to_path)

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


def example2():
    mesh_names = []
    for name in glob.glob(str("../*vessel*.stl")):
        stlRegex = re.compile(r".stl$")
        mo1 = stlRegex.search(name)
        if mo1 != None:
            mesh_names.append(name)
    start = 0
    steps = 4
    # with Parallel(n_jobs=-2) as parallel:
    momentan = 0
    for k in range(start, len(mesh_names) - 1):
        momentan = k

        print(k)
        if "pulmonary" in mesh_names[k]:
            sets = k
            meshname = mesh_names[k][3::]
            vessels = []
            vessels = process_mesh(mesh_names[k])
            create_images(vessels, k)
            break


def ex3():
    mesh_color = [100, 100, 100, 50]
    path_point_color = [100, 100, 100, 50]
    highlight_color = [255, 0, 0, 250]
    to_show = []
    for vessel in vessels:
        to_show.append((vessel.path, path_point_color))
        # to_show.extend(([vessel.path[link[0]], link[1].path[link[2]]],
        # highlight_color) for link in vessel.links_to_path)
        to_show.append((selected_vessel.submesh, [155, 100, 100, 10]))
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
    # print(to_show)
    show(to_show)

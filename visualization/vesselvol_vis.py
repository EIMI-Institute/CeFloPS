import networkx as nx
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import math


def no_refs(volume, volumes):
    for v in volumes:
        if volume in v.next_vol:
            return False
    return True


def get_vol_for_vessel_ind(vessel, ind, lower=False):
    for volume in vessel.volumes:
        i = 1
        if lower or ind == 0:
            i = 0
        if volume.path_indices[i] == ind:
            return volume
    print(
        vessel.associated_vesselname, [vol.path_indices for vol in vessel.volumes], ind
    )


def create_graph_from_volumes_append(volumes, vesselnames=False):
    G = nx.DiGraph()
    if vesselnames:
        for i, vol in enumerate(volumes):
            G.add_node(f"{vol.id},{vol.vessel.associated_vesselname[-20::]}")

            for app in vol.to_append:
                vol2 = get_vol_for_vessel_ind(app[0], app[1], True)
                if vol2 != None:
                    G.add_edge(
                        f"{vol.id},{vol.vessel.associated_vesselname[-20::]}",
                        f"{vol2.id},{vol2.vessel.associated_vesselname[-20::]}",
                    )
                """ if vol2.path_indices[1]==app[1]:
                for 
                G.add_edge(
                    f"{vol.id},{vol.vessel.associated_vesselname[-20::]}",
                    f"{vol2.id},{vol2.vessel.associated_vesselname[-20::]}",
                ) """
    else:
        for i, vol in enumerate(volumes):
            G.add_node(f"{vol.id}")

            for vol2 in vol.next_vol:
                G.add_edge(f"{vol.id}", f"{vol2.id}")
    return G


def create_graph_from_volumes(volumes, vesselnames=False):
    G = nx.DiGraph()
    if vesselnames:
        for i, vol in enumerate(volumes):
            G.add_node(
                f"{vol.id},{vol.vessel.associated_vesselname[-20::]}",
                node_color="tab:red",
            )

            for vol2 in vol.next_vol:
                G.add_edge(
                    f"{vol.id},{vol.vessel.associated_vesselname[-20::]}",
                    f"{vol2.id},{vol2.vessel.associated_vesselname[-20::]}",
                )
    else:
        for i, vol in enumerate(volumes):
            G.add_node(f"{vol.id}")

            for vol2 in vol.next_vol:
                G.add_edge(f"{vol.id}", f"{vol2.id}")
    return G


def set_prevs(volumes):
    for vol in volumes:
        vol.prevs = set()
    for vol in volumes:
        for v2 in vol.next_vol:
            v2.prevs.add(vol)


def orderOfMagnitude(number):
    if type(number) == float or type(number) == int or type(number) == np.float64:
        try:
            return math.floor(math.log(number, 10))
        except:
            return number
    return number


def colored_volumegraph(
    volumes,
    restype=False,
    spacing=1,
    print_attr=None,
    print_attr_only=False,
    print_attrs=[],
    attr_funs=[],
    path="",
    font_size=8,
    node_size_multiplier=1,
    highlight_vols=[],
    ret_image=False,
    verbose=False,
    arrowsize=25,
):
    volumes = list(volumes)
    if not all([hasattr(vol, "prevs") for vol in volumes]):
        print("SETPREVS")
        set_prevs(volumes)

    def ident(vol):
        lies_outside_of_volumes = False
        try:
            ident = [i for i in range(len(volumes)) if volumes[i] == vol][
                0
            ]  # str(vol.id)#str(vol.id)
            ident = vol.id
        except:
            print("Volume in next but not in volumes", vol.vessel.associated_vesselname)
            ident = vol.id
            lies_outside_of_volumes = True
        return ident, lies_outside_of_volumes

    if print_attr_only == True:
        assert all(
            [hasattr(volume, print_attr) for volume in volumes]
        ), "All volumes need to have printattr if print_attr_only is set"

    def name(volume):
        """name return the name of the volume in the graph

        Args:
            volume (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(print_attrs) == 0:
            prefix = ""
            if hasattr(volume, "name"):
                prefix = f"{volume.name}\n"
            appendix = ""
            if print_attr != None:
                if hasattr(volume, print_attr):
                    if print_attr_only:
                        return f"\n {getattr(volume, print_attr)}  \n {orderOfMagnitude(getattr(volume, print_attr))}"
                    appendix = f"\n {getattr(volume, print_attr)}  \n {orderOfMagnitude(getattr(volume, print_attr))}"

            if hasattr(volume, "path_indices") and not hasattr(volume, "lower_volumes"):
                if (
                    hasattr(volume, "vessel")
                    and volume.vessel != None
                    and hasattr(volume.vessel, "associated_vesselname")
                ):
                    main = f"{volume.path_indices},{volume.vessel.associated_vesselname[-20::]}"
                else:
                    main = str(int(id(volume)))
            elif hasattr(volume, "lower_volumes"):
                main = ""  # f"lowerlen {len(volume.lower_volumes)}"
            else:
                main = str(int(id(volume)))
        else:
            name = ""
            for i, attr in enumerate(print_attrs):
                if attr == "vessel.associated_vesselname":
                    if len(attr_funs) == len(print_attrs):
                        name = (
                            name
                            + f"{attr_funs[i](volume.vessel.associated_vesselname[-20::])} \n"
                        )
                    else:
                        name = name + f"{volume.vessel.associated_vesselname[-20::]} \n"
                elif hasattr(volume, attr):
                    if len(attr_funs) == len(print_attrs):
                        name = name + f"{attr_funs[i](getattr(volume, attr))} \n"
                    else:
                        name = name + f"{getattr(volume, attr)}"[0:8] + "\n"
            return name
        return prefix + main + appendix[0:8]

    labeldict = {}
    for vol in volumes:
        labeldict[ident(vol)[0]] = name(vol)

    fig = plt.figure(1, figsize=(160, 90), dpi=60)
    G = nx.DiGraph()
    nodelist = []
    entrylist = []
    outlist = []
    highlightlist = []
    outlier = []
    for volume in volumes:
        if volume not in highlight_vols:
            if ident(volume)[1] == False:
                if no_refs(volume, volumes):
                    entrylist.append(ident(volume)[0])
                elif len(volume.next_vol) == 0:
                    outlist.append(ident(volume)[0])
                else:
                    nodelist.append(ident(volume)[0])
            else:
                outlier.append(ident(volume)[0])
        else:
            highlightlist.append(ident(volume)[0])
    # nodes
    G.add_nodes_from(nodelist, color="blue")
    G.add_nodes_from(entrylist, color="green")
    G.add_nodes_from(outlist, color="red")
    G.add_nodes_from(highlightlist, color="yellow")
    G.add_nodes_from(outlier, color="grey")
    print("nodes pre egdes:", len([node for node in G]))

    if verbose:
        print("Entry:")
    for elem in entrylist:
        if verbose:
            print(elem)
    for vol in volumes:
        for vol2 in vol.next_vol:
            if verbose:
                print(
                    vol.resistance,
                    len(vol.next_vol),
                    "->",
                    vol2.resistance,
                    len(vol2.next_vol),
                )
            G.add_edge(
                ident(vol)[0],
                ident(vol2)[0],
            )
    if path != "":
        fig = plt.figure(1, figsize=(160, 90), dpi=60)
    if restype:
        node_size = 200
        pos = circuit_layout(G, node_size=200)
        fig, ax = plt.subplots()
        color = nx.get_node_attributes(G, "color")
        color_map = [color[node] for node in G]
        # Draw the network with rectangular nodes
        # nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_size, node_color='lightblue', font_size=8,arrowsize=20, node_shape='s')
        nx.draw_networkx(
            G,
            pos,
            node_size=[
                (
                    300 * node_size_multiplier
                    if node not in entrylist
                    else 1200 * node_size_multiplier
                )
                for node in G
            ],
            font_size=font_size,
            node_color=color_map,
            arrowsize=25,
            labels=labeldict,
            with_labels=True,
            node_shape="s",
        )
        # Find starting nodes (nodes with no predecessors)
        starting_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

        # Add arrows for starting nodes
        for n in starting_nodes:
            # Create an invisible node above the starting node
            invisible_node_pos = (pos[n][0], pos[n][1] + 1.1)

            # Draw an arrow from the invisible node to the starting node
            ax.annotate(
                "",
                xy=(pos[n][0], pos[n][1] + 0.1),
                xycoords="data",
                xytext=invisible_node_pos,
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="->",
                    shrinkA=0,
                    shrinkB=0,
                    patchA=None,
                ),
            )
    else:
        if True:
            print("nodes:", len([node for node in G]))
            print(
                "nodelistlengths",
                len(nodelist),
                len(entrylist),
                len(outlist),
                len(outlier),
                len(highlightlist),
            )

        if len(G) == 1:
            pos = {[node for node in G][0]: (0, 0)}
        else:
            try:
                pos = nx.nx_agraph.graphviz_layout(
                    G, args=f"-Gsplines=true -Gsep={spacing}.0"
                )
            except Exception as e:
                print(e)

                def get_vol_x_y(node):
                    def get_volend_coord(volume):
                        return volume.resistance, volume.resistance

                    for volume in volumes:
                        if ident(volume) == node:
                            if hasattr(volume.vessel, "path"):
                                print("haspath")
                                return volume.vessel.path[volume.path_indices[1]][0:2]
                            else:
                                return (
                                    sum(
                                        [
                                            np.asarray(get_volend_coord(v))
                                            for v in volume.next_vol
                                        ]
                                    )
                                    - 10 / len(volume.next_vol)
                                    + sum(
                                        [
                                            np.asarray(get_volend_coord(v))
                                            for v in volume.prevs
                                        ]
                                    )
                                    - 10 / len(volume.prevs)
                                )[0:2]

                pos = nx.random_layout(
                    G
                )  # dict((node, get_vol_x_y(node)) for node in G)
                print("pos", pos)
        # print(nx.get_node_attributes(G, "color"))
        color = nx.get_node_attributes(G, "color")
        try:
            color_map = [color[node] for node in G]
            nx.draw_networkx(
                G,
                pos,
                node_size=[
                    (
                        300 * node_size_multiplier
                        if node not in entrylist
                        else 1200 * node_size_multiplier
                    )
                    for node in G
                ],
                font_size=font_size,
                node_color=color_map,
                arrowsize=arrowsize,
                labels=labeldict,
                with_labels=True,
            )
        except:
            print("len nodes", len([node for node in G]), len(volumes))
            nx.draw_networkx(
                G,
                pos,
                node_size=[
                    (
                        300 * node_size_multiplier
                        if node not in entrylist
                        else 1200 * node_size_multiplier
                    )
                    for node in G
                ],
                font_size=font_size,
                arrowsize=arrowsize,
                labels=labeldict,
                with_labels=True,
            )

    # plt.tight_layout()
    print(list(nx.simple_cycles(G)))

    plt.axis("off")
    if not ret_image:
        plt.show()
    else:
        fig.canvas.draw()
        image = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
    if path != "":
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if ret_image:
        return image


def circuit_layout(G, node_size=700):
    pos = {}
    levels = {}
    horizontal_spacing = 2  # Adjust horizontal spacing as needed
    vertical_spacing = 1  # Adjust vertical spacing as needed

    def add_to_level(level, node):
        if level not in levels:
            levels[level] = []
        levels[level].append(node)

    def place_nodes(node, level=0):
        if node in pos:
            return
        add_to_level(level, node)
        width = len(levels[level])
        x = (width - 1) * horizontal_spacing
        y = -level * vertical_spacing
        pos[node] = (x, y)
        for succ in G.successors(node):
            place_nodes(succ, level + 1)

    # Initialize the placement with the first node
    root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    for root in root_nodes:
        place_nodes(root)

    # Adjust the x positions to center-align each level
    for level, nodes in levels.items():
        min_x = min(pos[node][0] for node in nodes)
        max_x = max(pos[node][0] for node in nodes)
        offset = (max_x - min_x) / 2
        for node in nodes:
            x, y = pos[node]
            pos[node] = (x - offset, y)

    return pos

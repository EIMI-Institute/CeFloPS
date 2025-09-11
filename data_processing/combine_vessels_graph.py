# %%
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import os
import sys

module_path = os.path.abspath(os.path.join("./../.."))
if module_path not in sys.path:
    sys.path.append(module_path)
from joblib import Parallel, delayed
import argparse
import glob
import re
import math
import os
import numpy as np
import trimesh
import scipy
import random
import sys, os, argparse
import CeFloPS.visualization as vis
from CeFloPS.data_processing.submesh_processing import (
    extract_centerline,
    extract_diameter_and_centre_from_cut,
    extract_diameter_and_centre_from_cut_two_sided,
    filter_centerline_by_distance,
)
from CeFloPS.simulation.common.vessel2 import Vessel, Link
from CeFloPS.data_processing.vessel_graph_fun import *
import pickle
from CeFloPS.simulation.common.functions import *
from CeFloPS.simulation.common.vessel_functions import *
import CeFloPS.simulation.settings as settings

# vessels_split_reduced_pre_graph_leg_inc
path_to_vesselsplit = r"./vessels_split_reduced_pre_graph.pickle"
# path_to_vesselsplit = r"./vessels_split_reduced_pre_graph_leg_inc.pickle"
SHOW = False
vessels = []
for filepath in glob.glob(path_to_vesselsplit):
    with open(
        filepath,
        "rb",
    ) as input_file:
        vessels += pickle.load(input_file)
if SHOW:
    vis.show_linked(vessels[::], None)
settings.ROI_VOXEL_PITCH
# %%
# remove isolated vessels
to_rem = set()
for v in vessels:
    if len(v.links_to_path) == 0 and "cor_vein" not in v.associated_vesselname:
        to_rem.add(v)
        # print(v)
for v in to_rem:
    vessels.remove(v)
# %%
for v in vessels:
    v.resistance = calculate_resistance(v)

p_nodes = create_p_nodes(vessels)

# Convert to networkx graph
G, doubles = create_graph(p_nodes, vessels)
print("DOUBLES", len(doubles))
# draw_resistor_networks(G)

# Example usage
# number_of_structures,cycled_vess = count_connected_structures(G)
# print(f"The graph has {number_of_structures} connected structures.")

# %%
to_show = [(v.path, [100, 100, 100, 40]) for v in vessels]

for x in doubles:
    for v in x:
        to_show.append((v.path, [200, 100, 100, 200]))
##vis.show(to_show)
# %%
for s in set(doubles):
    merge_doubled_vessels(s, vessels)

p_nodes = create_p_nodes(vessels)
p_node_vessels = set()
for node in p_nodes:
    p_node_vessels.update(node.connected_vessels)
print(
    len(vessels),
    len(p_node_vessels),
    [str(v) for v in vessels if v not in p_node_vessels],
)
# Convert to networkx graph
G, doubles = create_graph(p_nodes, vessels, single_mode=True)
print("DOUBLES", len(doubles))
# %%
# draw_resistor_networks(G)


# %%
remove_border_leq5(vessels)
# %%
p_node_vessels = set()
for node in p_nodes:
    p_node_vessels.update(node.connected_vessels)
len(vessels), len(p_node_vessels), [str(v) for v in vessels if v not in p_node_vessels]
# %%


p_nodes = create_p_nodes(vessels)
# Convert to networkx graph
G, doubles = create_graph(p_nodes, vessels)
border_vessels = get_borders(vessels, G, p_nodes)
to_show = []
links = []
for v in vessels:
    for link in v.links_to_path:
        showable_path = trimesh.load_path(
            [
                v.path[link.source_index],
                link.target_vessel.path[link.target_index],
            ]
        )
        showable_path.colors = [[24, 25, 25, 150]]
        links.append(showable_path)
p = [node.calculate_position() for node in p_nodes]
for x in border_vessels:
    for v in x:
        to_show.append((v.path, [200, 100, 100, 200]))
to_show += (
    [(v.path, [100, 100, 100, 40]) for v in vessels] + [(p, [20, 200, 200, 20])] + links
)
# %%
##vis.show(to_show)


# %%
# %%
def show_graph_vessels(G):
    edge_vessels = set()
    for _, _, data in G.edges(data=True):
        if "vessel" in data:
            edge_vessels.add(data["vessel"])
    to_show = [(v.path, [100, 100, 100, 40]) for v in vessels] + [
        (v.path, [200, 100, 100, 140]) for v in edge_vessels
    ]

    ##vis.show(to_show)


show_graph_vessels(G)


def group_p_nodes(p_nodes, graph):
    """
    Group PNodes by the connected components they belong to.

    :param p_nodes: List of PNode objects.
    :param graph: The networkx graph object representing the network.
    :return: A list of sets, where each set contains the PNodes belonging to a connected component.
    """
    # Map from node ID to PNode for quick lookup
    node_to_pnode = {p_node.get_id(): p_node for p_node in p_nodes}

    # Initialize a list to store groups of PNodes
    p_node_connected_segments = []

    # Iterate over each connected component in the graph
    for component in nx.connected_components(graph):
        # Initialize a set to collect PNodes for the current component
        component_p_nodes = set()

        # Iterate over nodes in the current component
        for node_id in component:
            # Ensure the node has a corresponding PNode
            if node_id in node_to_pnode:
                component_p_nodes.add(node_to_pnode[node_id])

        # Append the collected PNodes if any are found
        if component_p_nodes:
            p_node_connected_segments.append(component_p_nodes)

    return p_node_connected_segments


# %%
"""
pul sys:heart
aorta:heart
right:
    119,258,278
    2 stueck
left:
    143,258,185
    auch 2
topend:
     111,218,264
botend:
        120,217,248
leber:
    sup_port,mesenteric, splenic, rest output
"""


heart_chambers = [
    np.array([179.24761044, 203.81855294, 248.20464232]),
    np.array([148.71463263, 181.67490973, 244.12171739]),
    # Add other chambers as needed
]
# find outer vessels/borders at heart to set voltage node there


p_nodes = create_p_nodes(vessels)
# Convert to networkx graph
G, doubles = create_graph(p_nodes, vessels)
full_graph_edge_vessels = {
    data["vessel"] for _, _, data in G.edges(data=True) if "vessel" in data
}
# Process each subgraph using the precomputed full graph edge vessels
vesset = set(vessels)

border_vessels = get_borders(vessels, G, p_nodes)
for node in p_nodes:
    node.isolated_vessels = [
        v for v in node.connected_vessels if any([v in x for x in border_vessels])
    ]

p_node_connected_segments = group_p_nodes(p_nodes, G)
print(len(p_node_connected_segments))
borders_per_subgraph = []
nodes_per_subgraph = []
full_vessels = []
full_graph_edge_vessels = {
    data["vessel"] for _, _, data in G.edges(data=True) if "vessel" in data
}
for i, conn in enumerate(p_node_connected_segments):
    borders_per_subgraph.append(set())
    nodes_per_subgraph.append(set())
    full_vessels.append(set())
    for pnode in conn:
        pnode_vessels = pnode.connected_vessels
        isolated_vessels = {
            v for v in pnode_vessels if v not in full_graph_edge_vessels
        }
        borders_per_subgraph[i].update(isolated_vessels)
        nodes_per_subgraph[i].add(pnode)
        full_vessels[i].update({v for v in pnode_vessels})

# split into categories
for i, bb in enumerate(zip(borders_per_subgraph, nodes_per_subgraph, full_vessels)):
    bo = bb[0]
    # Check which category the current set belongs to based on the vessel names
    if any("aorta" in vessel.associated_vesselname for vessel in bo):
        aorta = bb
    elif any("top" in vessel.associated_vesselname for vessel in bo):
        top_veins = bb
    elif any("bot" in vessel.associated_vesselname for vessel in bo):
        bottom_veins = bb
    elif any("sys_pul" in vessel.associated_vesselname for vessel in bo):
        sys_pul = bb
    elif any(
        "pulmonary_vein" in vessel.associated_vesselname
        and "left" in vessel.associated_vesselname
        for vessel in bo
    ):
        pulmonary_veins_l = bb
    elif any(
        "pulmonary_vein" in vessel.associated_vesselname
        and "right" in vessel.associated_vesselname
        for vessel in bo
    ):
        pulmonary_veins_r = bb
    elif any("sup_port" in vessel.associated_vesselname for vessel in bo):
        liver = bb
# vis.show_linked(vessels[::],None)

# %%
variables = [
    top_veins,
    bottom_veins,
    aorta,
    sys_pul,
    pulmonary_veins_l,
    pulmonary_veins_r,
    liver,
]
var_reference_points = [
    [np.array([111, 218, 264])],  # top
    [
        np.array([179.24761044, 203.81855294, 248.20464232]),
        np.array([148.71463263, 181.67490973, 244.12171739]),
    ],  # bot heart
    [
        np.array([179.24761044, 203.81855294, 248.20464232]),
        np.array([148.71463263, 181.67490973, 244.12171739]),
    ],  # aorta heart
    [
        np.array([179.24761044, 203.81855294, 248.20464232]),
        np.array([148.71463263, 181.67490973, 244.12171739]),
    ],  # sys pul heart
    [np.array([143, 258, 285])],  # left
    [np.array([119, 258, 278])],  # right
]
# Iterate over the variables and assert their length is greater than zero
for i, var in enumerate(variables, start=1):
    assert len(var[0]) > 0, f"Set {i} is empty"


# %%
def find_closest_vessel(borders_per_subgraph, reference_points):
    """
    Find the closest vessel in each subgraph to the given reference points.

    :param borders_per_subgraph: List of sets, each containing vessels.
    :param reference_points: Dictionary of reference point names to their coordinates.
    :param keyword: Keyword to mark special handling of vessels containing this keyword.
    :return: List of tuples containing the closest vessel, point, and reference point name for each subgraph.
    """
    min_distance = float("inf")
    closest_one = None

    for vessel in borders_per_subgraph:

        # Check distances from both endpoints of the vessel to each reference point
        for point in [vessel.path[0], vessel.path[-1]]:
            for ref_center in reference_points:
                distance = np.linalg.norm(np.asarray(point) - np.asarray(ref_center))
                if distance < min_distance:
                    min_distance = distance
                    closest_one = (vessel, point)
    return closest_one


def find_closest_node(nodes, reference_points):
    """
    Find the closest vessel in each subgraph to the given reference points.

    :param borders_per_subgraph: List of sets, each containing vessels.
    :param reference_points: Dictionary of reference point names to their coordinates.
    :param keyword: Keyword to mark special handling of vessels containing this keyword.
    :return: List of tuples containing the closest vessel, point, and reference point name for each subgraph.
    """
    min_distance = float("inf")
    closest_one = None

    for node in nodes:

        # Check distances from both endpoints of the vessel to each reference point
        point = node.calculate_position()
        for ref_center in reference_points:
            distance = np.linalg.norm(np.asarray(point) - np.asarray(ref_center))
            if distance < min_distance:
                min_distance = distance
                closest_one = (node, point)
    return closest_one


# %%
closest_per_subgraph = []
for i, v in enumerate(variables[0:4], start=0):
    var = v[0]
    print(i, len(var_reference_points), len(variables))
    closest_one = find_closest_vessel(var, var_reference_points[i])
    closest_per_subgraph.append(closest_one)
closest_nodes = []
for i, var in enumerate(var_reference_points[4:]):
    print(var)
    closest_one = find_closest_node(p_nodes, var)
    closest_nodes.append(closest_one)

# %%
to_show = [(v.path, [100, 100, 100, 40]) for v in vessels]
for x in var_reference_points[4::]:
    for s in x:
        print(([ss for ss in s], [250, 100, 100, 210]))
        to_show.append(([[ss for ss in s]], [250, 100, 100, 210]))
for v in closest_per_subgraph:
    # print(p)
    to_show.append((v[0].path, [250, 100, 100, 250]))
for k in closest_nodes:
    to_show.append(([k[1]], [1, 250, 100, 250]))

##vis.show(to_show)


# %%
def show_colored_resistances(vessels, use_log_scale=False, override_property=None):
    to_show = (
        []
    )  # Entries like (vessel.path,[100,100,100,100]) where the second list is the RGBA color

    # Assuming each vessel has an attribute `resistance`
    resistances = [vessel.resistance for vessel in vessels]

    if override_property is not None:
        resistances = [getattr(vessel, override_property) for vessel in vessels]

    # Handle potential issues with zero or negative resistances when using log scale
    if use_log_scale:
        resistances = np.array(resistances)
        resistances = np.where(
            resistances <= 0, np.finfo(float).eps, resistances
        )  # Avoid log(0) or log(negative)

    # Normalize resistances for color mapping
    if use_log_scale:
        log_resistances = np.log(resistances)
        min_resistance = log_resistances.min()
        max_resistance = log_resistances.max()
        normalized_resistances = (log_resistances - min_resistance) / (
            max_resistance - min_resistance
        )
    else:
        min_resistance = min(resistances)
        max_resistance = max(resistances)
        normalized_resistances = [
            (r - min_resistance) / (max_resistance - min_resistance)
            for r in resistances
        ]

    # Create a color map
    cmap = plt.get_cmap("viridis")   

    # Iterate over vessels and create color based on normalized resistance
    for vessel, norm_resistance in zip(vessels, normalized_resistances):
        # Get color from colormap (RGBA)
        color = cmap(norm_resistance)

        # Convert color to a list with values between 0 and 255 for RGB
        color_255 = [int(255 * component) for component in color]

        # Add vessel path and color to the list
        to_show.append((vessel.path, color_255))

    # Show the visualized items
    vis.show(to_show)

    # Plotting the reference colormap
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    # Set up the color bar
    if use_log_scale:
        # Using a log scale normalizer for the color bar
        from matplotlib.colors import LogNorm

        norm = LogNorm(vmin=min(resistances), vmax=max(resistances))
        colorbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
        )

        # Using LogLocator for better tick placement on a log scale
        from matplotlib.ticker import LogLocator

        colorbar.locator = LogLocator(base=10.0)  # or use a different base if needed
        colorbar.update_ticks()
    else:
        norm = plt.Normalize(vmin=min_resistance, vmax=max_resistance)
        colorbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
        )

    # Customize the color bar with a label
    attr = "Resistance"
    if override_property is not None:
        attr = override_property
    colorbar.set_label(
        attr + " (Log Scale)" if use_log_scale else "Resistance (Linear Scale)"
    )

    plt.show()


# show_colored_resistances(vessels,True,"q")
# %%
def make_graph(volumes):

    def get_label(vol):
        vessel_name = (
            vol.vessel.associated_vesselname[-32:]
            if vol.vessel.associated_vesselname
            else ""
        )

        symval = vol.get_symval(vol.Q_1) * 1000000
        base = f"{vessel_name}\n{symval}"
        return base

    fig = plt.figure(1, figsize=(320, 180), dpi=60)
    G = nx.DiGraph()

    nodelist = []
    entrylist = []
    outlist = []
    labels = {}

    for volume in volumes:
        node_id = str(volume.vessel.id)
        label = get_label(volume)
        labels[node_id] = label

        if volf.no_refs(volume, volumes):
            entrylist.append(node_id)
        elif len(volume.next_vol) == 0:
            outlist.append(node_id)
        else:
            nodelist.append(node_id)

    # respective colors
    G.add_nodes_from(nodelist, color="blue")
    G.add_nodes_from(entrylist, color="green")
    G.add_nodes_from(outlist, color="red")
    # labels for each node
    nx.set_node_attributes(G, labels, "label")

    # Add edges based on vessel.id
    for vol in volumes:
        source_id = str(vol.vessel.id)
        for vol2 in vol.next_vol:
            target_id = str(vol2.vessel.id)
            G.add_edge(source_id, target_id)

    return G


# Plotting the graph
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.components import weakly_connected_components


# connected_subgraphs = [G.subgraph(c).copy() for c in sorted(weakly_connected_components(G), key=len)]
def show_graphq(volumes):
    a = make_graph(volumes, [])
    pos = nx.nx_agraph.graphviz_layout(a)
    color = nx.get_node_attributes(a, "color")
    color_map = [color[node] for node in a.nodes()]
    labels = nx.get_node_attributes(a, "label")
    plt.figure(figsize=(80, 80))
    nx.draw_networkx(
        a,
        pos,
        labels=labels,
        node_size=4000,
        font_size=20,
        node_color=color_map,
        arrowsize=100,
    )
    plt.show()


# %%
# group pos vessels and pnodes
assign_nodes_to_vessels(vessels, p_nodes)
grouped_posnodes = []

for i in range(6):
    if i > 3:
        grouped_posnodes.append((variables[i][1], closest_nodes[i - 4]))
    else:
        grouped_posnodes.append((variables[i][1], closest_per_subgraph[i][0]))


for i in range(4):
    continue
    to_show = []
    col_vessels = set()
    for node in variables[i][1]:
        col_vessels.update(node.connected_vessels)
    for v in col_vessels:
        colorlist = create_color_list(len(v.path))
        for p in range(len(v.path) // 2):
            to_show.append(([v.path[p]], colorlist[p]))
    # vis.show(to_show)
collect_nodes_vess = [set(), set()]
for nodes, pos_vess in grouped_posnodes[0:2]:
    collect_nodes_vess[0].update(nodes)
    collect_nodes_vess[1].add(pos_vess)


G = create_potential_graph_from_pos_vessels(
    collect_nodes_vess[0], collect_nodes_vess[1]
)
# %%
coronary_veins = [v for v in vessels if "cor_vein" in v.associated_vesselname]
# pos of coronary veins should also be the same as for bot and topvein
# add these with pos node at the side thats closer to the right heart or [144.958,239.241 ,243.906 ]
# GND - POS POS NODE NODE GND
# add edges and treat the closest node as pos!

# vis.show_linked(coronary_veins[::],None)
filtered_nodes = [
    node
    for node in p_nodes
    if any(["cor_vein" in v.associated_vesselname for v in node.connected_vessels])
]
target_position = np.array([144.958, 239.241, 243.906])

closest_node = None
min_distance = float("inf")

for node in filtered_nodes:
    node_position = node.calculate_position()
    distance = np.linalg.norm(np.array(node_position) - target_position)
    if distance < min_distance:
        min_distance = distance
        closest_node = node
for v in coronary_veins:
    if closest_node in v.connected_nodes:
        v.mark_to_pos_node = True
# add this node and a very small resistance to pos:
closest_node.isolated_vessels
for i, node1 in enumerate(filtered_nodes):
    for j, node2 in enumerate(filtered_nodes):
        if i < j:
            vessels_between = node1.connected_vessels.intersection(
                node2.connected_vessels
            )
            if vessels_between:
                assert len(vessels_between) == 1
                node_1_name = node1.get_id()
                node_2_name = node2.get_id()
                for shared_vessel in vessels_between:
                    G.add_edge(
                        node_1_name,
                        node_2_name,
                        resistance=shared_vessel.resistance,
                        vessel=shared_vessel,
                    )
    # add the nodes from isolated for node1
    for v in node1.isolated_vessels:
        # alll isolated to ground
        G.add_edge(node1.get_id(), "GND", resistance=v.resistance, vessel=v)
G.add_edge(closest_node.get_id(), "POS", resistance=0.00001, vessel=None)


# %%
updated_vessels = list()
check_unique_vessels(G)
updated_vessels.append([])
updated_vessels[0] = solve_and_save_vesselnetwork(
    G, positive_node="POS", ground_node="GND"
)
# %%
positive_graphvesseledges = []
for u, v, key, edge_data in G.edges(data=True, keys=True):
    if "POS" in [u, v]:
        positive_graphvesseledges.append(edge_data["vessel"])
        print(u, v)
for v in positive_graphvesseledges:
    if v != None:
        print(v.q)


# %%
def draw_resistor_network(G):
    plt.figure(figsize=(24, 24))
    pos = nx.nx_agraph.graphviz_layout(G)

    # Node coloring
    node_colors = [
        "red" if node == "POS" else "yellow" if node == "GND" else "lightblue"
        for node in G.nodes()
    ]

    # Create width list proportional to current (with minimum width)
    edge_currents = [abs(data["current"]) for _, _, data in G.edges(data=True)]
    max_current = max(edge_currents) if edge_currents else 1  # Prevent division by zero
    width_scale = 5.0  # Adjust this to control maximum line width
    edge_widths = [
        0.05 + (abs(data["current"]) / max_current * width_scale)
        for _, _, data in G.edges(data=True)
    ]

    # Draw elements
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_size=6, font_color="black")

    # Draw edges with proportional widths and arrows
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        node_size=100,  # Match node size for proper arrow placement
    )

    # Create edge labels with formatted current
    edge_labels = {
        (u, v): f"{data['current']:.2e}A"  # Scientific notation for small values
        for u, v, data in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=5,
        label_pos=0.3,  # Adjust label position along edge
        font_color="darkgreen",  # Different color for visibility
    )

    plt.title("Resistor Network with Current Flow (Width Proportional to Current)")
    plt.axis("off")
    plt.show()


def mark_vessels(G, vessels):
    positive_graphvesseledges = []
    others = []
    for u, v, key, edge_data in G.edges(data=True, keys=True):
        if "POS" in [u, v]:
            positive_graphvesseledges.append(edge_data["vessel"])
        else:
            others.append(edge_data["vessel"])
    vis.show(
        [(v.path, [100, 100, 100, 100]) for v in vessels]
        + [
            (v.path, [200, 100, 100, 200])
            for v in positive_graphvesseledges
            if v is not None
        ]
        + [(v.path, [200, 100, 200, 200]) for v in others]
    )


# mark_vessels(G,vessels)
draw_resistor_network(G)


# %%
for nodes, pos_vess in grouped_posnodes[2:4]:
    assert any([pos_vess in n.isolated_vessels for n in nodes])
    G = create_potential_graph(nodes, pos_vess)
    check_unique_vessels(G)
    updated_vessels.append([])
    updated_vessels[-1] = solve_and_save_vesselnetwork(
        G, positive_node="POS", ground_node="GND"
    )

    # mark_vessels(G,vessels)
    draw_resistor_network(G)

    # draw_resistor_networks(G)
    # G,doubles=create_graph(p_nodes,vessels)
    """ draw_resistor_networks(G)
    solve_and_save_vesselnetwork(G,positive_node='POS',ground_node='GND') """

# %%
# TODO directly plot


# %%
# use the closest nodes for the pulmonary veins, these should be CONNECTED together to spread the 100 ml!
nodes1, pos_nodes = set(), set()
for nodes, pos_node in grouped_posnodes[4:]:
    print(1)
    assert pos_node[0] in nodes
    nodes1.update(nodes)
    pos_nodes.add(pos_node[0])
G = create_potential_graph_from_pos_node(nodes1, pos_nodes)
check_unique_vessels(G)
updated_vessels.append([])
updated_vessels[-1] = solve_and_save_vesselnetwork(
    G, positive_node="POS", ground_node="GND"
)

# mark_vessels(G,vessels)
draw_resistor_network(G)

# %% collect all mesenteric,... vessels for liver and set them as positive, creatinng a positive node for them
# border vessels from liver

keywords = ["sup_port", "mesenteric", "splenic"]
pos_vessels = set()
for livervess in liver[0]:
    if any([keyword in livervess.associated_vesselname for keyword in keywords]):
        pos_vessels.add(livervess)
if SHOW:
    vis.show(
        [(v.path, [100, 100, 100, 100]) for v in vessels]
        + [(v.path, [200, 100, 100, 200]) for v in pos_vessels]
    )
# %%
G = create_potential_graph_from_pos_vessels(liver[1], pos_vessels)
check_unique_vessels(G)
updated_vessels.append([])
updated_vessels[-1] = solve_and_save_vesselnetwork(
    G, positive_node="POS", ground_node="GND", input_current=0.00002
)  # liver portvein

# mark_vessels(G,vessels)
draw_resistor_network(G)

# %% create new links and reverse the veins
# set final directions:
# reverse veins pulmonary, systemic
for vess in (
    list(top_veins[2])
    + list(bottom_veins[2])
    + list(pulmonary_veins_l[2])
    + list(pulmonary_veins_r[2])
    + coronary_veins
):
    vess.reverse()
    # swap potential sides too
    # vess.lower_potential_node, vess.higher_potential_node = vess.higher_potential_node, vess.lower_potential_node

# %%
for i in range(8):
    to_show = []
    if i == 7:

        for v in coronary_veins:
            colorlist = create_color_list(len(v.path))
            for p in range(len(v.path)):
                to_show.append(([v.path[p]], colorlist[p]))
            to_show.append(([v.path[-1]], [250, 10, 10, 250]))
    else:
        if not (i < 3 and i >= 2) and False:
            continue
        to_show = []
        col_vessels = set()
        for node in variables[i][1]:
            col_vessels.update(node.connected_vessels)
        for v in col_vessels:
            colorlist = create_color_list(len(v.path))
            for p in range(len(v.path)):
                to_show.append(([v.path[p]], colorlist[p]))
            to_show.append(([v.path[-1]], [250, 10, 10, 250]))
    if to_show:
        ...  # vis.show(to_show)
# %%
""" for v in vessels:
    for l in v.links_to_path:
        if l.source_vessel.relink(l)==None:
            l.target_vessel.add_link(Link(l.target_vessel,l.target_index,l.source_vessel,l.source_index))

#delete all relinks (links from 0)
for v in vessels:
    to_rem=[]

    for l in v.links_to_path:
        if l.source_index==0 and len(v.path)>1:
            to_rem.append(l)
    for l in to_rem:
        v.links_to_path.remove(l) """

# %%
for (
    vessel
) in vessels:  # [v for v in vessels if "cor_vein" not in v.associated_vesselname]:
    vessel.links_to_path = []


def get_pnodes_vessel(vessel, p_nodes):
    return {node for node in p_nodes if vessel in node.connected_vessels}


for (
    vessel
) in vessels:  # [v for v in vessels if "cor_vein" not in v.associated_vesselname]:
    assoc_nodes = get_pnodes_vessel(vessel, p_nodes)
    assert (
        1 <= len(assoc_nodes) <= 2
    ), f"Vessel {vessel.associated_vesselname} has {len(assoc_nodes)} nodes"
    for connected_node in assoc_nodes:
        if vessel in liver[2]:
            # handle liver straightforward like all are not flipped
            # just connect from higher potential to lower (portvein side)
            if connected_node.get_id() == vessel.lower_potential_node:
                # Veins at higher potential node link to vessels with this node as their lower
                for v2 in connected_node.connected_vessels:
                    if connected_node.get_id() == v2.higher_potential_node:
                        vessel.add_link(Link(vessel, len(vessel.path) - 1, v2, 0))
            continue
        if connected_node.get_id() == vessel.higher_potential_node:
            # Higher potential node: vein flows away, others flow in
            if vessel.type == "vein":
                # Veins at higher potential node link to vessels with this node as their lower
                for v2 in connected_node.connected_vessels:
                    if v2 is not None and v2 != vessel:
                        if connected_node.get_id() == v2.lower_potential_node:
                            vessel.add_link(Link(vessel, len(vessel.path) - 1, v2, 0))
        else:
            # Lower potential node: non-veins link to vessels with this node as their higher
            if vessel.type != "vein":
                for v2 in connected_node.connected_vessels:
                    if v2 is not None and v2 != vessel:
                        if connected_node.get_id() == v2.higher_potential_node:
                            vessel.add_link(Link(vessel, len(vessel.path) - 1, v2, 0))


# %%
def no_links_to(vessel, vessels):
    return (
        len([l for l in v.links_to_path for v in vessels if l.target_vessel == vessel])
        == 0
    )


def draw_vessel_network(vessels, subgraphmode=True):
    G = nx.DiGraph()

    # First create all nodes with original IDs
    node_ids = {vessel.id for vessel in vessels}
    for vessel in vessels:
        G.add_node(vessel.id)

    # Add edges with validation
    for vessel in vessels:
        for link in vessel.links_to_path:
            # if link.source_index == len(vessel.path)-1:
            target_id = link.target_vessel.id
            # Ensure target exists in our vessel list
            if target_id in node_ids:
                G.add_edge(
                    vessel.id, target_id, source_name=vessel.associated_vesselname
                )

    # Create safe labels without newlines
    relabel_map = {}
    seen_labels = set()
    for vessel in vessels:
        # Clean label creation
        base_name = (
            vessel.associated_vesselname[-16:].strip().replace("\n", " ")
            if not no_links_to(vessel, vessels)
            else ""
        )
        q_value = f"{vessel.q*1000000:.2f}"
        pathlength = len(vessel.path)
        label = f"{base_name} | {q_value}ml | {pathlength} |"

        # Ensure uniqueness
        counter = 1
        original_label = label
        while label in seen_labels:
            label = f"{original_label}_{counter}"
            counter += 1
        seen_labels.add(label)
        # add length of path

        relabel_map[vessel.id] = label

    # Apply relabeling
    G = nx.relabel_nodes(G, relabel_map)

    if subgraphmode:
        components = list(nx.weakly_connected_components(G))

        # Create separate plots for each subgraph
        for idx, component in enumerate(components):
            plt.figure(figsize=(48, 48))
            subgraph = G.subgraph(component)
            layout_args = [
                "-Gnodesep=8.0",
                "-Granksep=10.0",
                "-Goverlap=prism",
                "-Gsplines=ortho",
                "-Gfontsize=10",
                "-Glabel_angle=30",
                "-Glabel_distance=2.5",
                "-Glabelfloat=true",
            ]

            try:
                pos = nx.nx_agraph.graphviz_layout(
                    subgraph, prog="dot", args=" ".join(layout_args)
                )
            except Exception as e:
                print(f"Graphviz failed: {e}, using spring layout")
                pos = nx.spring_layout(subgraph, k=12.0, iterations=500, seed=42)
            # Node coloring
            node_colors = [
                (
                    "red"
                    if subgraph.in_degree(n) == 0
                    else "blue" if subgraph.out_degree(n) == 0 else "lightgreen"
                )
                for n in subgraph.nodes()
            ]

            # Edge coloring
            edge_colors = [
                (
                    "red"
                    if subgraph.edges[e].get("source_name", "").lower().startswith("al")
                    else "darkblue"
                )
                for e in subgraph.edges()
            ]

            # Draw subgraph
            nx.draw(
                subgraph,
                pos,
                with_labels=True,
                node_size=1500,
                node_color=node_colors,
                edge_color=edge_colors,
                font_size=12,
                font_weight="bold",
                arrowsize=30,
                linewidths=1.5,
                edgecolors="black",
                width=2.5,
            )

            plt.title(f"Vessel Subgraph {idx+1}/{len(components)}", fontsize=28, pad=40)
            plt.tight_layout()
            plt.show()

        return components

    # Prepare visualization
    plt.figure(figsize=(96, 24))

    # Use safer layout engine
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(
            G, k=5, iterations=100, seed=42
        )  # Fallback layoutnx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Node coloring logic
    node_colors = [
        (
            "red"
            if G.in_degree(n) == 0
            else "blue" if G.out_degree(n) == 0 else "lightgreen"
        )
        for n in G.nodes()
    ]

    # Edge coloring logic
    edge_colors = [
        (
            "red"
            if G.edges[e].get("source_name", "").lower().startswith("al")
            else "darkblue"
        )
        for e in G.edges()
    ]

    # Draw with improved parameters
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=800,
        node_color=node_colors,
        edge_color=edge_colors,
        font_size=8,
        font_weight="bold",
        arrowsize=20,
        linewidths=0.5,
        edgecolors="gray",
    )

    plt.title("Vessel Network", fontsize=24)
    plt.tight_layout()
    plt.show()


draw_vessel_network(vessels)
# %%


# %%
# vis.show_linked(vessels[::],None)
# group pos node and pnodes
# set all others to ground
# %%
# calculate aorta flow:
fs = list()
for v in vessels:
    if "aorta" in v.associated_vesselname:
        fs.append(v.q)
        print(v.diameters)
fs = sorted(fs)[::-1]
print(fs)

""" [0.00010000000000000034, 9.7177625973775e-05, 8.903152040019792e-05, 7.955696971590017e-05, 7.508800326847071e-05, 6.0873956876483795e-05, 5.360567828279449e-05, 5.162617019645387e-05, 4.664141513980489e-05, 4.03604452403663e-05, 3.946077241920631e-05, 3.5550944944624376e-05, 3.390509140887465e-05, 3.2065850277376955e-05, 3.048551909537415e-05, 2.848710722662535e-05, 8.937820109796355e-06]
"""

# %%
# assign single volumes to all vessels
from CeFloPS.data_processing.volume_fun import resistant_volume

for v in vessels:
    new_vol = resistant_volume(
        np.mean(v.diameters) / 2,
        v.length + 0.0001,
        res=v.resistance,
        Q=v.q,
        P1=v.p1,
        P2=v.p2,
    )
    new_vol.path_indices = (0, len(v.path) - 1)
    new_vol.vessel = v
    # p1 -p2
    v.volumes = [new_vol]

# vessel.volumes[0].next_vol#add all llater ones volumes
# %%
import numpy as np
import trimesh


def add_voxels(voxelized_mesh_a, voxelized_mesh_b):
    """Combine two VoxelGrids by adding filled voxels from both grids.

    Args:
        voxelized_mesh_a (trimesh.voxel.base.VoxelGrid): First voxel grid.
        voxelized_mesh_b (trimesh.voxel.base.VoxelGrid): Second voxel grid.

    Returns:
        trimesh.voxel.base.VoxelGrid: A new VoxelGrid with combined filled voxels.
    """
    assert isinstance(voxelized_mesh_a, trimesh.voxel.base.VoxelGrid)
    assert isinstance(voxelized_mesh_b, trimesh.voxel.base.VoxelGrid)

    # Calculate bounding boxes
    bbox_a = voxelized_mesh_a.bounds
    bbox_b = voxelized_mesh_b.bounds

    # Determine overall minimum and maximum corners for the combined bounding box
    min_corner = np.minimum(bbox_a[0], bbox_b[0])
    max_corner = np.maximum(bbox_a[1], bbox_b[1])

    if np.any(max_corner < min_corner):
        raise ValueError(
            f"Invalid bounding box, max corner is smaller than min corner: {max_corner} < {min_corner}"
        )

    # Calculate the shape of the combined grid based on the bounding box and scale
    combined_size = ((max_corner - min_corner) / voxelized_mesh_a.scale).astype(int) + 1

    if np.any(combined_size < 0):
        raise ValueError(f"Calculated combined shape is negative: {combined_size}")

    # Initialize the combined grid
    combined_matrix = np.zeros(combined_size, dtype=bool)

    # Fill the combined matrix
    def fill_combined_matrix(voxel_grid, offset):
        filled_positions = np.argwhere(voxel_grid.matrix)
        for pos in filled_positions:
            combined_pos = tuple(pos + offset)
            try:
                combined_matrix[combined_pos] = True
            except IndexError as e:
                print(f"IndexError: {e} at position {combined_pos}")

    # Calculate offsets and fill combined matrix
    offset_a = np.floor(
        (voxelized_mesh_a.bounds[0] - min_corner) / voxelized_mesh_a.scale
    ).astype(int)
    fill_combined_matrix(voxelized_mesh_a, offset_a)

    offset_b = np.floor(
        (voxelized_mesh_b.bounds[0] - min_corner) / voxelized_mesh_b.scale
    ).astype(int)
    fill_combined_matrix(voxelized_mesh_b, offset_b)

    # Create a new VoxelGrid from the combined matrix
    combined_voxelgrid = trimesh.voxel.base.VoxelGrid(
        combined_matrix, transform=np.eye(4)
    )

    return combined_voxelgrid


import numpy as np
import trimesh


def add_voxels(voxelized_mesh_a, voxelized_mesh_b):
    """Combine two VoxelGrids by adding filled voxels from both grids.

    Args:
        voxelized_mesh_a (trimesh.voxel.base.VoxelGrid): First voxel grid.
        voxelized_mesh_b (trimesh.voxel.base.VoxelGrid): Second voxel grid.

    Returns:
        trimesh.voxel.base.VoxelGrid: A new VoxelGrid with combined filled voxels.
    """
    assert isinstance(voxelized_mesh_a, trimesh.voxel.base.VoxelGrid)
    assert isinstance(voxelized_mesh_b, trimesh.voxel.base.VoxelGrid)

    # Calculate bounding boxes
    bbox_a = voxelized_mesh_a.bounds
    bbox_b = voxelized_mesh_b.bounds

    # Determine overall minimum and maximum corners for the combined bounding box
    min_corner = np.minimum(bbox_a[0], bbox_b[0])
    max_corner = np.maximum(bbox_a[1], bbox_b[1])

    if np.any(max_corner < min_corner):
        raise ValueError(
            f"Invalid bounding box, max corner is smaller than min corner: {max_corner} < {min_corner}"
        )

    # Calculate the shape of the combined grid based on the bounding box and scale
    combined_size = ((max_corner - min_corner) / voxelized_mesh_a.scale).astype(int) + 1

    if np.any(combined_size < 0):
        raise ValueError(f"Calculated combined shape is negative: {combined_size}")

    # Initialize the combined grid
    combined_matrix = np.zeros(combined_size, dtype=bool)

    # Fill the combined matrix
    def fill_combined_matrix(voxel_grid, offset):
        filled_positions = np.argwhere(voxel_grid.matrix)
        for pos in filled_positions:
            combined_pos = tuple(pos + offset)
            try:
                combined_matrix[combined_pos] = True
            except IndexError as e:
                print(f"IndexError: {e} at position {combined_pos}")

    # Calculate offsets and fill combined matrix
    offset_a = np.floor(
        (voxelized_mesh_a.bounds[0] - min_corner) / voxelized_mesh_a.scale
    ).astype(int)
    fill_combined_matrix(voxelized_mesh_a, offset_a)

    offset_b = np.floor(
        (voxelized_mesh_b.bounds[0] - min_corner) / voxelized_mesh_b.scale
    ).astype(int)
    fill_combined_matrix(voxelized_mesh_b, offset_b)

    # Compute the combined transformation
    # Establish a translation to the new minimum corner
    translation = np.eye(4)
    translation[:3, 3] = min_corner

    # Create a new VoxelGrid from the combined matrix with the correct transform
    combined_voxelgrid = trimesh.voxel.base.VoxelGrid(
        combined_matrix, transform=translation
    )

    return combined_voxelgrid


# %%
from CeFloPS.simulation.simsetup import *

# load heart data, combine atrium and ventricel and check links from our vessels to the closest (all of them end in the heart now, yay)
base_path = r"pathto\test\heart/"
right_atrium, right_ventricle = trimesh.load(
    base_path + "beating_heart.systole.right_atrium.stl"
), trimesh.load(base_path + "beating_heart.systole.right_ventricle.stl")
left_atrium, left_ventricle = trimesh.load(
    base_path + "beating_heart.systole.left_atrium.stl"
), trimesh.load(base_path + "beating_heart.systole.left_ventricle_4.stl")
voxel_size = 1
right_heart = add_voxels(
    right_atrium.voxelized(voxel_size).fill(),
    right_ventricle.voxelized(voxel_size).fill(),
)

# %%
to_show = [
    (right_heart.points, [100, 100, 100, 100]),
    (right_atrium.voxelized(voxel_size).fill().points, [100, 100, 200, 100]),
    (right_ventricle.voxelized(voxel_size).fill().points, [100, 200, 100, 100]),
]
##vis.show(to_show)
# %%
outer_shell_m = trimesh.load(base_path + "beating_heart.systole.pericardium.stl")
right_heart = add_voxels(
    right_atrium.voxelized(voxel_size).fill(),
    right_ventricle.voxelized(voxel_size).fill(),
)
left_heart = add_voxels(
    left_atrium.voxelized(voxel_size).fill(),
    left_ventricle.voxelized(voxel_size).fill(),
)
outer_shell = outer_shell_m.voxelized(voxel_size).fill()

try:
    # omit as these now only contain bronchi and trachea subs = settings.NEGATIVE_SPACES
    subs_meshes = [trimesh.load(path) for path in settings.VESSELPATHS]  # vesselmeshes
except Exception as e:
    print(f"Couldn't load negative Spaces, Error: {e}")
negative_voxelgrids = [sub.voxelized(voxel_size).fill() for sub in subs_meshes]

# subtract all vessels from them:
for geometry in [outer_shell, left_heart, right_heart]:
    for negative_grid in negative_voxelgrids:
        subtract_voxels(geometry, negative_grid)
# for pericard also subtract sides:
subtract_voxels(outer_shell, left_heart)
subtract_voxels(outer_shell, right_heart)

# %%
to_show = [(v.path, [250, 100, 100, 100]) for v in vessels] + [
    (list(geometry.points), [100, 200, 100, 100])
    for geometry in [outer_shell, left_heart, right_heart]
]
if SHOW:
    vis.show(to_show)

# %%
# to_show=[(right_heart.points,[100,200,100,100]),(left_heart.points,[200,100,100,100]),(outer_shell.points,[10,10,10,10])]
to_show = [
    (list(negative_grid.points), [100, 200, 100, 100])
    for negative_grid in negative_voxelgrids
] + [
    (list(right_heart.points), [100, 200, 100, 200]),
    (list(left_heart.points), [200, 100, 100, 100]),
]
# vis.show(to_show)

# %%

# create links to correct side at closest point/index
vessends = []
for v in closest_per_subgraph:
    # print(p)
    to_show.append((v[0].path, [250, 100, 100, 250]))
for k in closest_nodes:
    to_show.append(([k[1]], [1, 250, 100, 250]))

##vis.show(to_show)
for v in vessels:
    if any(k in v.associated_vesselname for k in ["liver", "mesenteric", "splenic"]):
        continue
    if (
        "POS" == v.lower_potential_node
        or closest_node.get_id() == v.lower_potential_node
    ):
        if v.type == "artery":
            vessends.append((v, len(v.path) - 1))
            continue
        vessends.append((v, 0))
    if (
        "POS" == v.higher_potential_node
        or closest_node.get_id() == v.higher_potential_node
    ):
        if v.type == "artery":
            vessends.append((v, 0))
            continue
        vessends.append((v, len(v.path) - 1))
to_show = [(v.path, [100, 100, 100, 100]) for v in vessels] + [
    ([x[0].path[x[1]]], [200, 100, 100, 100]) for x in vessends
]
if SHOW:
    vis.show(to_show)
print(
    [(v[0].associated_vesselname, v[1]) for v in vessends]
)  # should be 4 pul 1 syspul 1 aorta 2 vena cava

# %%
# create VOI object for right, left and pericardium
left_heart_voi = Tissue_roi(
    "heart_left_atrium_p_ventricle",
    "MANUAL",  # path
    3,
    [0, 0, 0],
    vessels,
    np.asarray(left_heart.points),
    len(left_heart.points),  # *1*1*1
    np.mean(np.asarray(left_heart.points)),
    "NAN",  # =get_k_name(name),
    "heart_left_atrium_p_ventricle",  # =get_roi_name(name),
    store_loc=True,
)
right_heart_voi = Tissue_roi(
    "heart_right_atrium_p_ventricle",
    "MANUAL",  # path
    3,
    [0, 0, 0],
    vessels,
    np.asarray(right_heart.points),
    len(right_heart.points),
    np.mean(np.asarray(right_heart.points)),
    "NAN",  # =get_k_name(name),
    "heart_right_atrium_p_ventricle",  # =get_roi_name(name),
    store_loc=True,
)
outer_heart_voi = Tissue_roi(
    "heart_pericard_musc",
    "MANUAL",  # path
    3,
    [0.0263, 0.3165, 0.0461],  # muscle
    vessels,
    np.asarray(outer_shell.points),
    len(outer_shell.points),
    np.mean(np.asarray(outer_shell.points)),
    "NAN",  # =get_k_name(name),
    "heart_pericard",  # =get_roi_name(name),
    store_loc=True,
)
# %%
for v in vessels:
    v.links_to_vois = []


# add the voi to the vessels to conenct to:
def match_vessel(vessel, right_heart_voi, left_heart_voi):
    name = vessel.associated_vesselname
    if "vl" in name:
        if "pulmonary" in name:
            return left_heart_voi, True
        else:
            # cor and cava
            return right_heart_voi, True
    else:
        # treat as vein when adding link
        if "aorta" in name:
            return left_heart_voi, False
        return right_heart_voi, False  # sys_pul_art


tiss_indices = []
for v, i in vessends:
    voi_to_connect, as_artery = match_vessel(v, right_heart_voi, left_heart_voi)
    tiss_indices.append(
        (
            voi_to_connect,
            get_closest_index_to_point(v.path[i], voi_to_connect.geometry.get_points()),
            as_artery,
        )
    )
    print(
        v.associated_vesselname,
        get_closest_index_to_point(v.path[i], voi_to_connect.geometry.get_points()),
    )

to_show = []
# create
links = []
for vi, ti in zip(vessends, tiss_indices):
    p1 = vi[0].path[vi[1]]
    p2 = ti[0].geometry.get_points()[ti[1]]
    for link in v.links_to_path:
        showable_path = trimesh.load_path([p1, p2])
        showable_path.colors = [[24, 25, 25, 150]]
        links.append(showable_path)
to_show = [(v.path, [100, 100, 100, 100]) for v in vessels] + links
# vis.show(to_show)
# %%
from CeFloPS.simulation.common.vessel2 import TissueLink

# Tlink creation
""" for v in coronary_veins:
    if closest_node in v.connected_nodes:
        v.mark_to_pos_node=True """


def store_vein_info(vein_index, vein_vessel, voi_object, voi_index):
    """store_vein_info store veinindex and next vein in voi object"""
    voi_object.outlets.append((vein_vessel, vein_index, voi_index))
    voi_object.veins.append(
        (
            vein_vessel.path[vein_index],
            vein_vessel,
            voi_index,
            voi_object.geometry.get_points()[voi_index],
        )
    )


def store_artery_info(artery_index, artery_vessel, voi_object, voi_index):
    """store_artery_info crate tlink at arteryindex to voiindex"""
    assert artery_vessel
    # source_vessel, source_index, target_tissue, target_index,
    link_to_voi = TissueLink(artery_vessel, artery_index, voi_object, voi_index)
    voi_object.inlets.append((artery_vessel, artery_index, voi_index))
    artery_vessel.links_to_vois.append(link_to_voi)
    artery_vessel.reachable_rois.append((artery_index, voi_object))


# store veins like artereis for the right heart side
# store arteries like veins for the left side

for vi, ti in zip(vessends, tiss_indices):
    vessel, vessel_index = vi
    voi_object, voi_index, as_artery = ti
    assert hasattr(vessel, "associated_vesselname")
    if as_artery:
        store_artery_info(vessel_index, vessel, voi_object, voi_index)
    else:
        store_vein_info(vessel_index, vessel, voi_object, voi_index)
# %%
# craete conections to outer one for veinends without Tlink and
loose_veinends = [
    vessel
    for vessel in vessels
    if vessel.type == "vein"
    and "pulmonary" not in vessel.associated_vesselname
    and "sup_vc" not in vessel.associated_vesselname
    and "lung" not in vessel.associated_vesselname
]
loose_arteryends = [
    vessel
    for vessel in vessels
    if vessel.type != "vein"
    and len(links_at(vessel, len(vessel.path) - 1)) == 0
    and "pulmonary" not in vessel.associated_vesselname
]
len(loose_arteryends), len(loose_veinends)


# %%
def ends_in_mesh(vessel, mesh):
    """Improved version using calculated bbox and ray casting"""
    point = vessel.path[0] if vessel.type == "vein" else vessel.path[-1]
    is_inside = mesh.contains([point])
    return is_inside[0]


final_rois = [outer_heart_voi]
threshold = 3
verbose = True
# get distance against all vois and if distance lower threshold connect
for vein in loose_veinends:
    point = vein.path[0]
    min_dist = None
    roi_to_connect = None
    for roi in final_rois:
        if (
            "blood" not in roi.name
        ):  # limit amount of calculation by removing ones that are too far away
            index, m_point = get_closest_index_to_point(
                point, roi.geometry.get_points(), return_distance=True
            )
            distance_to_roi = calculate_distance_euclidian(point, m_point)
            if verbose:
                print(vein.associated_vesselname[-20::], roi.name, distance_to_roi)
            if distance_to_roi < threshold and (
                min_dist == None or min_dist > distance_to_roi
            ):
                min_dist = distance_to_roi
                roi_to_connect = roi
                roi_to_connect_index = index
            if ends_in_mesh(vein, outer_shell_m):
                min_dist = distance_to_roi
                roi_to_connect = roi
                roi_to_connect_index = index
                break

    if roi_to_connect != None:
        store_vein_info(0, vein, roi_to_connect, roi_to_connect_index)

for art in loose_arteryends:
    point = art.path[-1]
    min_dist = None
    roi_to_connect = None
    for roi in final_rois:
        if (
            "blood" not in roi.name
        ):  # limit amount of calculation by removing ones that are too far away
            index, m_point = get_closest_index_to_point(
                point, roi.geometry.get_points(), return_distance=True
            )
            distance_to_roi = calculate_distance_euclidian(point, m_point)
            if verbose:
                print(art.associated_vesselname[-20::], roi.name, distance_to_roi)
            if distance_to_roi < threshold and (
                min_dist == None or min_dist > distance_to_roi
            ):
                min_dist = distance_to_roi
                roi_to_connect = roi
                roi_to_connect_index = index

            if ends_in_mesh(art, outer_shell_m):
                min_dist = distance_to_roi
                roi_to_connect = roi
                roi_to_connect_index = index
                break

    if roi_to_connect != None:
        store_artery_info(len(art.path) - 1, art, roi_to_connect, roi_to_connect_index)
for voi in [right_heart_voi, left_heart_voi, outer_heart_voi]:
    print(voi.name)
    print([v[0].associated_vesselname for v in voi.inlets])
    print([v[0].associated_vesselname for v in voi.outlets])
print("vesselside")
for v in vessels:
    if "cor_vein" in v.associated_vesselname:
        print([(l.target_tissue.name, l.source_index) for l in v.links_to_vois])
# %%
"""
to_show=[]
#create
links=[]
for vi,ti in zip(vessends,tiss_indices):
    p1=vi[0].path[vi[1]]
    p2=ti[0].geometry.get_points()[ti[1]]
    for link in v.links_to_path:
            showable_path = trimesh.load_path(
                                [
                                    p1,p2

                                ]
                            )
            showable_path.colors = [[24, 25, 25, 150]]
            links.append(showable_path)
to_show=[(v.path,[100,100,100,100]) for v in vessels]+links
 """


def show_tlinks(vessels, final_rois, extra=[], highlight_empty_artends=False):
    to_show = [(vessel.path, [10, 10, 10, 10]) for vessel in vessels]
    for roi in final_rois:
        if "blood" not in roi.name:
            for inlet in roi.inlets:
                # create red path
                print(inlet, len(inlet[0].path), inlet[0].type)
                art, art_index, voi_index = inlet
                showable_path = trimesh.load_path(
                    [
                        art.path[art_index],
                        roi.geometry.get_points()[voi_index],
                        # link.target_vessel.path[link.target_vessel.lookup_index(vessel)],
                    ]
                )
                showable_path.colors = [[244, 25, 25, 150]]
                to_show.append(showable_path)
            for outlet in roi.outlets:
                # create blue path
                # print(outlet)
                vein, vein_index, voi_index = outlet
                showable_path = trimesh.load_path(
                    [
                        vein.path[vein_index],
                        roi.geometry.get_points()[voi_index],
                        # link.target_vessel.path[link.target_vessel.lookup_index(vessel)],
                    ]
                )
                showable_path.colors = [[25, 25, 250, 150]]

                to_show.append(showable_path)
    for vessel in vessels:
        if (
            vessel.type == "artery"
            and len(links_at(vessel, len(vessel.path) - 1)) == 0
            and hasattr(vessel, "links_to_vois")
            and len(vessel.links_to_vois) == 0
        ):
            to_show.append(([vessel.path[-1]], [0, 250, 0, 250]))
    vis.show(to_show + extra)


SHOW = True
if SHOW:
    show_tlinks(
        vessels,
        [right_heart_voi, left_heart_voi, outer_heart_voi],
        extra=[
            (list(geometry.points), [100, 200, 100, 100])
            for geometry in [outer_shell, left_heart, right_heart]
        ],
    )


# %%
for geometry in [outer_heart_voi, right_heart_voi, left_heart_voi]:
    geometry.pitch = 1
    for v, _, _ in geometry.inlets:
        print(v.associated_vesselname, geometry.name)
    for v, _, _ in geometry.outlets:
        print("OUT", v.associated_vesselname, geometry.name)
    geometry.create_flow_vectorfield(save=False, store_loc=True, safe_variant=True)

""" import importlib
importlib.reload(CeFloPS.simulation.common.tissue_roi_parallel) """
# %%
# calc and save flowfield
with open(
    r"pathto\micreduced\mic_vesselgraph\CeFloPS\simulation\vessel_output/vessels_heartvoi_p_1.pickle",
    "wb",
) as handle:
    pickle.dump(
        [vessels, [outer_heart_voi, right_heart_voi, left_heart_voi]],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# %%
import pickle
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import os
import sys

module_path = os.path.abspath(os.path.join("./../.."))
if module_path not in sys.path:
    sys.path.append(module_path)
from joblib import Parallel, delayed
import argparse
import glob
import re
import math
import os
import numpy as np
import trimesh
import scipy
import random
import sys, os, argparse
from CeFloPS.data_processing.vessel_processing_funs import *
import visualization as vis
from CeFloPS.data_processing.submesh_processing import (
    extract_centerline,
    extract_diameter_and_centre_from_cut,
    extract_diameter_and_centre_from_cut_two_sided,
    filter_centerline_by_distance,
)
from CeFloPS.simulation.common.vessel2 import Vessel, Link
from CeFloPS.simulation.common.flowwalk import generate_arrays, generate_gradient_array
from CeFloPS.simulation.common.tissue_roi_parallel import fit_capillary_speed
from CeFloPS.data_processing.vessel_graph_fun import *

# save as a combination (vessels,vois)
testload = simsetup.load_vessels()
for x in testload[1]:
    x.reload(testload[0])
print(x.pitch)
# %%
vessels = testload[0]

nohighvessels = []
for v in vessels:
    if v.type == "vein":
        if v.no_higher_links(0):
            nohighvessels.append(v)
# vis.show([(v.path,[100,100,100,100]) for v in vessels]+[(v.path,[200,100,100,200]) for v in nohighvessels])


# %%
# Run test paths across the vessels and voi:
def traverse_vessel(vessel, start_index, path_added, time_added, cell=None):
    """Traverse a vessel until a decision for further travel has to be made by a cell

    Args:
        start_index (_type_): Index at which traversal starts
        path_added (_type_): Previous path from traversal-chain
        time_added (_type_): Previous time from traversal-chain

    Returns:
        _type_: (pathpoints, times)
    """

    def flow_walk_step(roi, pos_index, speed=1, already_visited=set(), iteration=0):
        roi_array = roi.vectorfield.get_points()
        roi_points_sorted = roi.geometry.get_points()
        selection = [
            int(i)
            for k, i in enumerate(roi_array[pos_index][0])  # nbs
            if roi_array[pos_index][1][k] > 0
            and i != -1
            and roi_array[pos_index][1][k] not in already_visited
        ]  # possible indexes from current index
        # print("AROUND",roi_array[pos_index][0])
        chances = [
            roi_array[pos_index][1][k]
            for k, i in enumerate(roi_array[pos_index][0])
            if roi_array[pos_index][1][k] > 0 and i != -1
        ]
        if len(selection) == 0 or iteration == -1:

            return None

        chances = normalize(chances)
        new_index = random.choices(selection, weights=chances, k=1)[0]
        time_taken = (
            calculate_distance_euclidian(
                roi_points_sorted[pos_index], roi_points_sorted[new_index]
            )
            / speed
        )
        new_point = roi_points_sorted[new_index]
        return new_index, new_point, time_taken

    def traverse_voi(startindex, voi):
        path = [voi.geometry.get_points()[startindex]]
        times = [0]
        current_index = startindex
        iteration = 0
        next_vessel = None
        visited = set()
        next_index = -1
        while True:
            step = flow_walk_step(
                voi,
                current_index,
                speed=1,
                already_visited=visited,
                iteration=iteration,
            )
            if step is None:
                break
            new_index, new_point, time_taken = step

            # Check if the new index is an outlet
            for outlet in voi.outlets:
                print("outletpoint", outlet[2], new_index)
                if new_index == outlet[2]:
                    next_vessel = outlet[0]
                    next_index = outlet[1]
                    path.append(new_point)
                    times.append(time_taken)
                    return (path, times, next_vessel, next_index)

            path.append(new_point)
            times.append(time_taken)
            visited.add(new_index)
            current_index = new_index
            iteration += 1

            if iteration >= 50000:
                break

        return (path, times, next_vessel, next_index)

    print(
        "Current",
        vessel.associated_vesselname,
        vessel.highest_link(),
        "nohigherlinks",
        vessel.no_higher_links(start_index),
    )
    # returns pathpart and time added
    # or links and time added til links
    if vessel.type == "vein":
        if cell is not None:
            if cell.loc != "vein" and "jump" not in vessel.associated_vesselname:
                cell.update_cell_cvolume("vein", cell.time)
        # print("vein", path_added, time_added)
        # go in direction until entering artery
        if vessel.no_higher_links(start_index):

            # check if there is a voi connected:
            connected_links = vessel.links_at(len(vessel.path) - 1, ignore_tlinks=False)
            if len(connected_links) > 0:
                path_added += vessel.path[start_index + 1 : :]
                # time_added += vessel.ti mes[start_index::]
                time_added += vessel.get_times(
                    start_index,
                    len(vessel.path) - 1,
                    no_vol_speed=not (settings.USE_VOL_SPEED),
                )
                # traverse in voi until next vessel stands, then continue with next vessel
                assert len(connected_links) == 1, (
                    vessel.associated_vesselname,
                    [tl.target_tissue.name for tl in connected_links],
                )

                path, times, next_vessel, next_index = traverse_voi(
                    connected_links[0].target_index, connected_links[0].target_tissue
                )
                assert next_vessel is not None
                path_added += path
                time_added += times
                return traverse_vessel(
                    next_vessel,
                    next_index,
                    path_added,
                    time_added,
                    cell,
                )
            else:

                print("Model is not connected?", vessel.associated_vesselname)

                path_added += vessel.path[start_index + 1 : :]
                # time_added += vessel.ti mes[start_index::]
                time_added += vessel.get_times(
                    start_index,
                    len(vessel.path) - 1,
                    no_vol_speed=not (settings.USE_VOL_SPEED),
                )
                return (path_added, time_added)
        else:
            # recursively traverse next vessel
            hl = vessel.highest_link()

            path_added += vessel.path[start_index + 1 : hl.source_index + 1] + [
                hl.target_vessel.path[hl.target_index]
            ]
            # time_added += vessel.t imes[start_index : hl.source_index] + [hl.get_time()]
            time_added += vessel.get_times(
                start_index, hl.source_index, no_vol_speed=not (settings.USE_VOL_SPEED)
            ) + [hl.get_time()]
            return traverse_vessel(
                vessel.highest_link().target_vessel,
                hl.target_index,
                path_added,
                time_added,
                cell,
            )
    else:
        if cell is not None:
            if cell.loc != "artery" and "jump" not in vessel.associated_vesselname:
                cell.update_cell_cvolume("artery", cell.time)

        # go in direction until entering a linked position
        if vessel.no_higher_links(start_index, ignore_tlinks=settings.IGNORE_TLINKS):
            path_added += vessel.path[start_index + 1 : :]
            # time_added += vessel.ti mes[start_index::]
            time_added += vessel.get_times(
                start_index,
                len(vessel.path) - 1,
                no_vol_speed=not (settings.USE_VOL_SPEED),
            )
            return (path_added, time_added)
        else:
            path_added += vessel.path[start_index + 1 : :]
            # time_added += vessel.ti mes[start_index::]
            time_added += vessel.get_times(
                start_index,
                len(vessel.path) - 1,
                no_vol_speed=not (settings.USE_VOL_SPEED),
            )
            # return possible next links
            return (
                (
                    vessel,
                    vessel.next_links(
                        start_index, ignore_tlinks=settings.IGNORE_TLINKS
                    ),
                ),  # TODO in cell remove startlink
                path_added,
                time_added,
                start_index,
            )


testveins = [v for v in vessels if v.type == "vein" and no_links_to(v, vessels)]
testvein = testveins[0]
# create a path from 0 of that vein up to the end of an artery with the overworked travelling function
_, path, times, _ = traverse_vessel(testvein, 0, [], [], cell=None)
# %%
to_show = [(v.path, [100, 100, 100, 100]) for v in vessels] + [
    (path, [100, 100, 200, 250])
]
# add multiple
for v in testveins:
    if "liver" not in v.associated_vesselname:
        _, path, times, _ = traverse_vessel(v, 0, [], [], cell=None)
        to_show += [(path, [100, 100, 200, 250])]
vis.show(to_show)
# %%
to_show = [(v.path, [100, 100, 100, 100]) for v in vessels] + [
    (testvein.path, [100, 100, 200, 250])
]
vis.show(to_show)
# %%
for v in testload[0]:
    if "rleg" in v.associated_vesselname:
        print(v.id, v.diameters)
        if "2383692171088_al.blood_vessel_486" in v.id:
            # vis.show([(v.path,[200,20,20,250])]+[(vv.path,[100,100,100,100]) for vv in testload[0]])
            ...  # assert False

# %%
import numpy as np

x = np.arange(10, dtype=np.float32)
y = np.arange(10, dtype=np.float32)
xx, yy = np.meshgrid(x, y, indexing="ij")  # Key change: 'ij' indexing
zz = np.zeros_like(xx)

# Stack and filter out (5,5)
roi_points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=1)
mask = ~((xx.flatten() == 5) & (yy.flatten() == 5))  # Directly filter (5,5)
roi_points_sorted = roi_points[mask].astype(np.float32)
roi_points_sorted = np.asarray(sorted(list(roi_points_sorted), key=lambda x: tuple(x)))
vein_indices = np.concatenate(
    (
        np.arange(0, 10),  # Left edge (x=0, y=0-9)
        np.arange(9, 100, 10),  # Top edge (y=9, x=0-9)
    )
).astype(np.int32)
vein_indices = [10]
# Create two arterial indices (inlets) in the middle
art_indices = np.array([45, 55], dtype=np.int32)  # Points (4,5,0) and (5,5,0)

print("Grid points example:")
print(roi_points_sorted[::10])  # Show every 10th point
print("\nVein indices (first 10):", vein_indices[:10])
print("Art indices:", art_indices)

# Generate the gradient array
vectorfield = generate_gradient_array(
    roi_points_sorted=roi_points_sorted,
    pitch=1.0,
    vein_indices=vein_indices,
    art_indices=art_indices,
    DEBUG=True,
)

# Verify output shapes
print("\nOutput shapes:")
print(f"Points shape: {vectorfield[0].shape}")
print(f"Vectorfield shape: {vectorfield[1].shape}")
# %%

# %%
from CeFloPS.simulation.common.shared_geometry import SharedGeometry

vois = testload[1]
vectorfield = generate_arrays(
    vois[0].geometry.get_points(),
    1.0,
    vein_indices=vois[0].vein_indices,
    art_indices=vois[0].art_indices,
)
# %%
vis.show_linked(vessels[::], None)
# %%
# vois[0].vectorfield=SharedGeometry(vectorfield[1])

fit_capillary_speed(vois, visual=True, plot=True, store_loc=True)
# %%
vessels = testload[0]
for v in vessels:
    consec_volumes = [
        l.target_vessel.volumes[0]
        for l in v.links_to_path
        if l.source_index == len(l.source_vessel.path) - 1
    ]
    v.volumes[0].next_vol = consec_volumes
volumes = []
for v in vessels:
    volumes.append(v.volumes[0])


# %%
def make_graph(volumes, equations):
    equations = list(equations)

    def name(vol):
        base = f"Q_{vol.vessel.id} "  # \n {vol.prevs}"
        for equ in list(equations):
            if vol.Q_1 in equ.free_symbols or vol.P_1 in equ.free_symbols:
                base = base + "\n" + str(equ)
        return base

    fig = plt.figure(1, figsize=(160, 90), dpi=60)
    G = nx.DiGraph()

    nodelist = []
    entrylist = []
    outlist = []
    for volume in volumes:
        if volf.no_refs(volume, volumes):
            entrylist.append(name(volume))
        elif len(volume.next_vol) == 0:
            outlist.append(name(volume))
        else:
            nodelist.append(name(volume))
    # nodes
    G.add_nodes_from(nodelist, color="blue")
    G.add_nodes_from(entrylist, color="green")
    G.add_nodes_from(outlist, color="red")
    for vol in volumes:
        for vol2 in vol.next_vol:
            G.add_edge(
                name(vol),
                name(vol2),
            )
    return G


# %%
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.components import weakly_connected_components

# Assuming G is your original graph
G = make_graph(volumes, [])  # Replace this with your actual function
connected_subgraphs = [
    G.subgraph(c).copy() for c in sorted(weakly_connected_components(G), key=len)
]
for a in connected_subgraphs:
    pos = nx.nx_agraph.graphviz_layout(a)
    # print(nx.get_node_attributes(G, "color"))
    color = nx.get_node_attributes(a, "color")
    color_map = [color[node] for node in a]
    plt.figure(figsize=(80, 80))  # Adjust this value according to your needs
    nx.draw_networkx(
        a, pos, node_size=4000, font_size=80, node_color=color_map, arrowsize=100
    )
    plt.show()

# %%
import matplotlib.pyplot as plt
from networkx import connected_components, DiGraph

# assuming G is your original directed graph (DiGraph)
G = make_graph(volumes, [])
sub_graphs_out = [G.subgraph(c) for c in connected_components(G)]

for i, sub_G in enumerate(sub_graphs_out):
    plt.figure()  # create a new figure window for each plot
    pos = nx.nx_agraph.graphviz_layout(sub_G)
    color = nx.get_node_attributes(sub_G, "color")
    color_map = [color[node] for node in sub_G]
    nx.draw_networkx(
        sub_G, pos, node_size=4000, font_size=80, node_color=color_map, arrowsize=100
    )
    plt.title(
        "Outgoing Edges Subgraph {}".format(i + 1)
    )  # add a title to each plot for clarity
plt.show()
# %%
import CeFloPS.simulation.common.volumes as volf

G = make_graph(volumes, [])
pos = nx.nx_agraph.graphviz_layout(G)
# print(nx.get_node_attributes(G, "color"))
color = nx.get_node_attributes(G, "color")
color_map = [color[node] for node in G]
nx.draw_networkx(
    G, pos, node_size=4000, font_size=80, node_color=color_map, arrowsize=100
)

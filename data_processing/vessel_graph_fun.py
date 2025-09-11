import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from CeFloPS.simulation.common.vessel2 import Vessel, Link


def nodal_analysis(graph, input_current, positive_node, ground_node):
    nodes = list(graph.nodes)
    nodes.remove(ground_node)

    node_index = {node: i for i, node in enumerate(nodes)}

    n = len(nodes)
    A = np.zeros((n, n))
    B = np.zeros(n)

    # Dictionary to hold equivalent resistances and original edges
    combined_resistances = {}
    edge_map = {}  # Maps edges to a list of (resistance, vessel_name) tuples

    multigraph = isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph)
    for u, v, key, data in graph.edges(data=True, keys=True):
        resistance = data.get("resistance", 1)
        vessel_name = data.get(
            "vessel", "unknown"
        )  # Default vessel name if not specified
        assert vessel_name != "unknown"
        if multigraph:
            if (u, v) not in combined_resistances:
                combined_resistances[(u, v)] = 0
                edge_map[(u, v)] = []
            combined_resistances[(u, v)] += 1 / resistance
            edge_map[(u, v)].append((resistance, vessel_name, key))
        else:
            combined_resistances[(u, v)] = 1 / resistance
            edge_map[(u, v)] = [(resistance, vessel_name, key)]

    # Convert sum of conductances back to equivalent resistances
    combined_resistances = {k: 1 / v for k, v in combined_resistances.items()}

    for (u, v), R_uv in combined_resistances.items():
        if u != ground_node and v != ground_node:
            A[node_index[u], node_index[u]] += 1 / R_uv
            A[node_index[v], node_index[v]] += 1 / R_uv
            A[node_index[u], node_index[v]] -= 1 / R_uv
            A[node_index[v], node_index[u]] -= 1 / R_uv
        elif u != ground_node:
            A[node_index[u], node_index[u]] += 1 / R_uv
        elif v != ground_node:
            A[node_index[v], node_index[v]] += 1 / R_uv

    B[node_index[positive_node]] = input_current

    V_vector = np.linalg.solve(A, B)

    voltages = {node: V_vector[i] for node, i in node_index.items()}
    voltages[ground_node] = 0

    currents_combined = {}
    for (u, v), R_uv in combined_resistances.items():
        if u == ground_node:
            I_uv = voltages[v] / R_uv
        elif v == ground_node:
            I_uv = voltages[u] / R_uv
        else:
            I_uv = (voltages[u] - voltages[v]) / R_uv
        currents_combined[(u, v)] = I_uv

    return voltages, currents_combined, edge_map


def distribute_currents_across_edges(total_current, edge_data, nodepair):
    total_conductance = sum(1 / resistance for resistance, _, _ in edge_data)
    individual_currents = []
    for resistance, vessel_name, edge_key in edge_data:
        current = total_current * (1 / resistance) / total_conductance
        individual_currents.append(
            (current, vessel_name, (nodepair[0], nodepair[1], edge_key))
        )
    return individual_currents


def assign_nodes_to_vessels(vessels, p_nodes):
    for vessel in vessels:
        vessel.connected_nodes = set()
    for node in p_nodes:
        for vessel in node.connected_vessels:
            vessel.connected_nodes.add(node)


def get_node_from_vessel(vessel, search_terms):
    print("vesselnodes", vessel.connected_nodes)
    print("searchterms", search_terms)
    result = []
    for term in search_terms:
        if term in ["POS", "GND"]:
            result.append(term)
        else:
            # Assume term is a node object with .get_id()
            try:
                term_id = term.get_id()
            except AttributeError:
                term_id = term  # If it's not an object, keep as is
            for node in vessel.connected_nodes:
                node_id = node if isinstance(node, str) else node.get_id()
                if node_id == term_id:
                    result.append(node)
                    break
    print("result", result)
    print("search terms", search_terms, "node ids", [n if isinstance(n, str) else n.get_id() for n in vessel.connected_nodes])
    return result


def solve_and_save_vesselnetwork(
    G, input_current=0.0001, positive_node=None, ground_node=None
):
    updated_vessels = list()
    assert ground_node is not None and positive_node is not None
    voltages, currents_combined, edge_map = nodal_analysis(
        G, input_current, positive_node, ground_node
    )
    for key, combined_current in currents_combined.items():
        reverse = False
        edge_data = edge_map[key]
        individual_currents = distribute_currents_across_edges(
            combined_current, edge_data, key
        )
        node1_voltage = voltages[key[0]]
        node2_voltage = voltages[key[1]]
        for current, vessel, edge_key in individual_currents:
            print(edge_key)
            print(edge_key in G.edges)
            G.edges[edge_key]["current"] = abs(current)
            if vessel is None:
                print("Nonecurrent", current)
                continue
            updated_vessels.append(vessel)
            vessel.q = abs(current)
            noderef1, noderef2 = get_node_from_vessel(vessel, [key[0], key[1]])
            # print(vessel.associated_vesselname,key[0],key[1],noderef1, " --NR2--  ",noderef2)
            # print("Potentials",voltages[key[0]],voltages[key[1]],"resistance",vessel.resistance)
            # U/R=I
            # change flow to always go to the lower potential!
            potential_difference = node2_voltage - node1_voltage
            if node1_voltage > node2_voltage:
                if current < 0:
                    current *= -1
            else:
                if current > 0:
                    current *= -1
            end_1, end_2 = (
                vessel.path[0],
                vessel.path[-1],
            )  # Use the last element instead of a fixed index
            # store the nodes in the vessel to later reset the links for a better fitting connection
            vessel.lower_potential_node = (
                key[0] if node1_voltage < node2_voltage else key[1]
            )
            vessel.higher_potential_node = (
                key[1] if node1_voltage < node2_voltage else key[0]
            )
            direction_aligned = False
            node2_position = None
            node1_position = None
            if noderef1 in ["POS", "GND"]:
                if vessel is not None:
                    assert noderef2 not in ["POS", "GND"]
                # Check if end_2 is closer to noderef2, confirming the alignment
                node2_position = noderef2.calculate_position()
                direction_aligned = np.linalg.norm(
                    np.array(end_2) - np.array(node2_position)
                ) < np.linalg.norm(np.array(end_1) - np.array(node2_position))

            elif noderef2 in ["POS", "GND"]:
                if vessel is not None:
                    assert noderef1 not in ["POS", "GND"]
                # Check if end_1 is closer to noderef1, confirming the alignment
                node1_position = noderef1.calculate_position()
                direction_aligned = np.linalg.norm(
                    np.array(end_1) - np.array(node1_position)
                ) < np.linalg.norm(np.array(end_2) - np.array(node1_position))

            else:
                # For regular nodes, check that the ends are aligned with noderef1->noderef2
                node1_position = noderef1.calculate_position()
                node2_position = noderef2.calculate_position()
                direction_aligned = np.linalg.norm(
                    np.array(end_1) - np.array(node1_position)
                ) < np.linalg.norm(np.array(end_1) - np.array(node2_position))

            # print("Vessel: ",vessel.associated_vesselname,"Current: ",current,"To",key[1], "aligned?",direction_aligned,"reverse?",(direction_aligned and current<0 )or (not direction_aligned and current>0),end_1, end_2)
            """ try:
                print("N1",node1_position)
                print("N2",node2_position)

            except:
                try:
                    print("N2",node2_position)
                except:
                    ... """

            # assert flow towards ground
            if "GND" in [noderef1, noderef2]:
                if len(vessel.path) == 1:
                    ...
                    # assert False
                if not (
                    (direction_aligned and current < 0)
                    or (not direction_aligned and current > 0)
                ):
                    if direction_aligned:
                        # ...
                        assert (
                            "GND" == noderef2
                        )  # wenn nicht gedreht und alighned, dann muss 2 GND sein
                    else:
                        # ...
                        assert "GND" == noderef1
                else:
                    # to be reversed
                    if direction_aligned:
                        # ...
                        assert "GND" == noderef1
                    else:
                        # ...
                        assert "GND" == noderef2

            if (direction_aligned and current < 0) or (
                not direction_aligned and current > 0
            ):
                vessel.reverse()

            vessel.p_delta = abs(node1_voltage - node2_voltage)
            vessel.p1 = vessel.p_delta
            vessel.p2 = 0
    return updated_vessels


def create_color_list(num_positions, alpha=100):
    colors = []

    if num_positions == 1:
        intensity = 0
        colors.append([intensity, intensity, intensity, alpha])
    else:
        for i in range(num_positions):
            # Calculate the intensity for this position
            intensity = 255 - int((255 / (num_positions - 1)) * i)
            colors.append([intensity, intensity, intensity, alpha])

    return colors


def calculate_resistance(single_vessel):
    VISCOSITY = 0.0035  # Pa·s (Pascal-seconds, already in SI units)
    PI = 3.14159265359

    def frustum_volume(r_lower, r_upper, h):
        """Calculate the volume of a frustum section in CUBIC METERS (SI)."""
        return (1 / 3) * PI * h * (r_upper**2 + r_upper * r_lower + r_lower**2)

    def segment_length(vessel, index):
        """Calculate the length between two points in METERS."""
        if index >= len(vessel.path) - 1:
            return 0
        # Convert path points from mm to meters, then compute length
        point1 = np.asarray(vessel.path[index]) / 1000  # mm → m
        point2 = np.asarray(vessel.path[index + 1]) / 1000  # mm → m
        return np.linalg.norm(point1 - point2)

    # Base length and volume calculation
    length = sum(
        segment_length(single_vessel, i) for i in range(len(single_vessel.path) - 1)
    )
    volume = 0

    # Calculate volume for the vessel's own segments
    for i in range(len(single_vessel.path) - 1):
        # Convert diameters from mm to meters
        d_start = single_vessel.diameters[i] / 1000  # mm → m
        d_end = single_vessel.diameters[i + 1] / 1000  # mm → m
        r_lower = d_start / 2  # radius in meters
        r_upper = d_end / 2  # radius in meters
        h = segment_length(single_vessel, i)  # already in meters
        volume += frustum_volume(r_lower, r_upper, h)

    # Adjust for links
    for link in single_vessel.links_to_path:
        linked_vessel = link.target_vessel

        # Convert link length from mm to meters
        source_point = (
            np.asarray(single_vessel.path[link.source_index]) / 1000
        )  # mm → m
        target_point = (
            np.asarray(linked_vessel.path[link.target_index]) / 1000
        )  # mm → m
        link_length = np.linalg.norm(source_point - target_point)  # in meters

        # Convert diameters at link endpoints from mm to meters
        d_start = single_vessel.diameters[link.source_index] / 1000  # mm → m
        d_end = linked_vessel.diameters[link.target_index] / 1000  # mm → m
        r_start = d_start / 2  # radius in meters
        r_end = d_end / 2  # radius in meters

        # Calculate link's volume and add to total
        link_volume = frustum_volume(r_start, r_end, link_length)
        volume += link_volume

        # Add the full link length to this vessel's total length
        length += link_length

    # Calculate effective radius and resistance
    if length == 0:
        return float("inf")
    radius = (volume / (PI * length)) ** 0.5  # in meters
    resistance = (8 * VISCOSITY * length) / (PI * radius**4)  # SI units
    return resistance


def add_connect_info(link):
    # returns the tuple where the connection really should be:
    # if length greater one, then according to 0 or length 0 or 1. Format (vessel,index,before|after as 0 | 1)
    # link.ind=(source01,target01))
    def vess_idn(vess, vessind, other_vess, debug=True):
        if len(vess.path) > 1:
            return 0 if vessind == 0 else 1
        else:
            if other_vess.orig != vess.orig:  # we need to look at iter too!
                if debug:
                    print(f"{vess}-{other_vess}___different original")
                return 0
            if debug:
                print(f"{vess}-{other_vess}___same original")
            return 1 if vess.iteration_number == 0 else 0  # internal, same orig

    source_idn = vess_idn(link.source_vessel, link.source_index, link.target_vessel)
    target_idn = vess_idn(link.target_vessel, link.target_index, link.source_vessel)
    link.ind = (source_idn, target_idn)


class PNode:
    def __init__(self, link=None):
        self.vess_indices = set()
        self.connected_vessels = set()
        if link is not None:
            self.add_link(link)

    def get_id(self):
        # Create a consistent unique identifier for the node
        return str(id(self)) + str(
            list(self.connected_vessels)[0].associated_vesselname[-32::]
        )

    def add_link(self, link):
        self.vess_indices.add((link.target_index, link.target_vessel, link.ind[1]))
        self.vess_indices.add((link.source_index, link.source_vessel, link.ind[0]))
        self.connected_vessels.update([link.source_vessel, link.target_vessel])

    def has_link(self, link):
        return (
            link.target_index,
            link.target_vessel,
            link.ind[1],
        ) in self.vess_indices or (
            link.source_index,
            link.source_vessel,
            link.ind[0],
        ) in self.vess_indices

    def __str__(self):
        return f"Node {self.get_id()}, connected:{[(str(v)) for v in self.connected_vessels]}"

    def calculate_position(self, max_iterations=100, tolerance=1e-3):
        # we use a fixpoint iteration approach to make this method stable against reversing vessels
        endpoints = []
        for vessel in self.connected_vessels:
            endpoints.extend([np.asarray(vessel.path[0]), np.asarray(vessel.path[-1])])

        # Initialize with mean of all endpoints (better than random)
        if not endpoints:
            return None
        current_pos = np.mean(endpoints, axis=0)

        # Iteratively refine position
        for _ in range(max_iterations):
            closest_points = []
            for vessel in self.connected_vessels:
                # Get vessel endpoints
                start = np.asarray(vessel.path[0])
                end = np.asarray(vessel.path[-1])

                # Find which endpoint is closer to current position
                dist_start = np.linalg.norm(current_pos - start)
                dist_end = np.linalg.norm(current_pos - end)

                # Select closest endpoint (prefer start if equidistant)
                closest = start if dist_start <= dist_end else end
                closest_points.append(closest)

            # Update position
            new_pos = np.mean(closest_points, axis=0)

            # Check convergence
            if np.linalg.norm(new_pos - current_pos) < tolerance:
                break

            current_pos = new_pos

        return current_pos


def merge_nodes(nodelist):
    print("merge")
    new_node = PNode()
    for node in nodelist:
        new_node.vess_indices.update(node.vess_indices)
        new_node.connected_vessels.update(node.connected_vessels)
    return new_node


def create_p_nodes(vessels):
    #create nodes for linking points from vessels!
    p_nodes = []
    for v in vessels:
        for l in v.links_to_path:
            add_connect_info(l)
            print(
                v.associated_vesselname,
                l.target_vessel.associated_vesselname,
                l.source_index,
                l.target_index,
                l.ind,
            )
    for v in vessels:
        for l in v.links_to_path:
            # Find nodes that share ANY part of this link (source or target)
            found = [node for node in p_nodes if node.has_link(l)]
            if not found:
                p_nodes.append(PNode(l))
            else:
                # Merge all found nodes into one
                merged_node = merge_nodes(found)
                for n in found:
                    p_nodes.remove(n)
                merged_node.add_link(l)  # Add the new link to the merged node
                p_nodes.append(merged_node)
    return p_nodes


def create_potential_graph(p_nodes, potential_vessel):
    G = nx.MultiGraph()
    for i, node1 in enumerate(p_nodes):
        for j, node2 in enumerate(p_nodes):
            if i < j:
                vessels_between = node1.connected_vessels.intersection(
                    node2.connected_vessels
                )
                if vessels_between:
                    assert len(vessels_between) == 1
                    for shared_vessel in vessels_between:
                        G.add_edge(
                            node1.get_id(),
                            node2.get_id(),
                            resistance=shared_vessel.resistance,
                            vessel=shared_vessel,
                        )
        # add the nodes from isolated for node1
        for v in node1.isolated_vessels:
            if v != potential_vessel:
                G.add_edge(node1.get_id(), "GND", resistance=v.resistance, vessel=v)
            else:
                G.add_edge(node1.get_id(), "POS", resistance=v.resistance, vessel=v)
                print("set POS")
    return G


def create_potential_graph_from_pos_node(p_nodes, pos_node=[]):
    G = nx.MultiGraph()
    if type(pos_node) != set():
        try:
            pos_node = set(pos_node)
        except:
            pos_node = set(list(pos_node))

    for i, node1 in enumerate(p_nodes):
        for j, node2 in enumerate(p_nodes):
            if i < j:
                vessels_between = node1.connected_vessels.intersection(
                    node2.connected_vessels
                )
                if vessels_between:
                    assert len(vessels_between) == 1
                    node_1_name = node1.get_id() if node1 not in pos_node else "POS"
                    node_2_name = node2.get_id() if node2 not in pos_node else "POS"
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

    return G


def create_potential_graph_from_pos_vessels(p_nodes, pos_vessels):
    G = nx.MultiGraph()
    for i, node1 in enumerate(p_nodes):
        for j, node2 in enumerate(p_nodes):
            if i < j:
                vessels_between = node1.connected_vessels.intersection(
                    node2.connected_vessels
                )
                if vessels_between:
                    assert len(vessels_between) == 1
                    for shared_vessel in vessels_between:
                        G.add_edge(
                            node1.get_id(),
                            node2.get_id(),
                            resistance=shared_vessel.resistance,
                            vessel=shared_vessel,
                        )
        # add the nodes from isolated for node1
        for v in node1.isolated_vessels:
            if v not in pos_vessels:
                G.add_edge(node1.get_id(), "GND", resistance=v.resistance, vessel=v)
            else:
                G.add_edge(node1.get_id(), "POS", resistance=v.resistance, vessel=v)
                print("set POS")
    return G


def check_unique_vessels(G):
    vessel_list = []

    # Iterate over all edges in the MultiGraph
    for u, v, key, edge_data in G.edges(data=True, keys=True):
        if "vessel" in edge_data:
            vessel_list.append(edge_data["vessel"])
    unique_vessel_set = set(vessel_list)

    # Compare sizes
    if len(vessel_list) == len(unique_vessel_set):
        print("All vessels are unique.")
    else:
        print("There are duplicate vessels.")
        assert False


def create_graph(p_nodes, vessels, single_mode=False):
    doubles = []

    G = nx.MultiGraph()
    for i, node1 in enumerate(p_nodes):
        for j, node2 in enumerate(p_nodes):
            if i < j:
                vessels_between = node1.connected_vessels.intersection(
                    node2.connected_vessels
                )
                if vessels_between:
                    if single_mode:
                        assert len(vessels_between) == 1
                    shared_vessel = list(vessels_between)[0]
                    if len(vessels_between) > 1:
                        doubles.append(tuple([v for v in vessels_between]))
                    for shared_vessel in vessels_between:
                        G.add_edge(
                            node1.get_id(),
                            node2.get_id(),
                            resistance=shared_vessel.resistance,
                            vessel=shared_vessel,
                        )

    """G = nx.MultiGraph()
    unique_nodes = {node.get_id(): node for node in p_nodes}
    for node in unique_nodes.values():
        G.add_node(node.get_id())

    # Add edges based on shared vessels
    for vessel in vessels:
        connected_nodes = [
            node_id for node_id, node in unique_nodes.items()
            if vessel in node.connected_vessels
        ]
        # A vessel should connect exactly two nodes (its start and end)
        if len(connected_nodes) == 2:
            G.add_edge(connected_nodes[0], connected_nodes[1], vessel=vessel)
        else:
            if len(connected_nodes) ==1:#border vessels
                continue
            assert False, (str(vessel),connected_nodes)"""

    return G, doubles


# Function to draw the graph
def draw_resistor_networks(G):
    for i, component in enumerate(list(nx.connected_components(G))):
        # Extract the subgraph corresponding to the current component
        subgraph = G.subgraph(component).copy()
        plt.figure(figsize=(10, 10))

        # Use graphviz layout for positioning nodes
        pos = nx.nx_agraph.graphviz_layout(subgraph)

        # Draw the subgraph
        nx.draw_networkx_nodes(subgraph, pos, node_size=100, node_color="lightblue")
        nx.draw_networkx_labels(subgraph, pos, font_size=5, font_color="black")
        nx.draw_networkx_edges(subgraph, pos, width=1.5)

        edge_labels = nx.get_edge_attributes(subgraph, "resistance")
        nx.draw_networkx_edge_labels(
            subgraph, pos, edge_labels=edge_labels, font_size=6
        )

        # Set plot title and display loop information
        plt.axis("off")
        plt.show()


""" def draw_resistor_network(G):
    plt.figure(figsize=(12, 12))
    pos = nx.nx_agraph.graphviz_layout(G)


    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")
    nx.draw_networkx_edges(G, pos, width=1.5)
    edge_labels = nx.get_edge_attributes(G, 'resistance')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title("Resistor Network Graph")
    plt.axis('off')
    plt.show() """


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


def count_connected_structures(G):
    # Calculate the connected components of the graph
    connected_components = list(nx.connected_components(G))
    vessels_in_cycle_list = []
    for i, component in enumerate(connected_components):
        # Extract the subgraph corresponding to the current component
        subgraph = G.subgraph(component).copy()

        # Determine if the subgraph has a cycle and collect vessels if it exists
        try:
            cycles = list(nx.simple_cycles(subgraph))
            vessels_in_cycle = set()
            for edge in cycles:
                # Handle MultiGraph edge format (u, v, key)
                if len(edge) == 3:
                    u, v, key = edge
                else:
                    print(edge)
                    u, v = edge
                    key = 0  # Default key for non-MultiGraph edges
                # Get the specific edge data using the key
                edge_data = G.get_edge_data(u, v).get(key, {})
                if "vessel" in edge_data:
                    shared_vessel = edge_data["vessel"]
                    vessels_in_cycle.add(shared_vessel)
            if vessels_in_cycle:
                vessels_list = [v.associated_vesselname for v in vessels_in_cycle]

                print(
                    f"Connected Structure {i+1} Cycle Vessels: {vessels_list}",
                    len(vessels_in_cycle),
                )
            else:
                print(
                    f"Connected Structure {i+1} has a cycle but no vessels with 'vessel' attribute."
                )
            vessels_in_cycle_list.append(vessels_in_cycle)
        except nx.NetworkXNoCycle:
            print(f"Connected Structure {i+1} has no cycle.")
    # Get the number of connected components (structures)
    num_connected_structures = len(connected_components)
    return num_connected_structures, vessels_in_cycle_list


def merge_doubled_vessels(vessels, all_vessels):

    # TODO enhance to actually merge paths and diameters

    print([len(vess.path) for vess in vessels])
    longest_vessel = max(vessels, key=lambda v: len(v.path))
    for v in vessels:
        if np.linalg.norm(
            np.asarray(v.path[0]) - np.asarray(longest_vessel.path[0])
        ) > np.linalg.norm(np.asarray(v.path[0]) - np.asarray(longest_vessel.path[-1])):
            v.reverse()
    """ path = longest_vessel.path
    dia=longest_vessel.diameters#sum([np.asarray(v.diameters) for v in vessels])/len(vessels)#combined diameter
    dia_a, dia_b = create_dia_a_b(dia)
    new_vess=  Vessel(
            path,
            dia_a,
            dia_b,
            np.mean(dia),
            dia,
            longest_vessel.associated_vesselname,
            longest_vessel.speed_function,
        )"""
    to_rem = []
    for o in longest_vessel.links_to_path:
        if o.target_vessel in vessels:
            to_rem.append(o)
    for o in to_rem:
        longest_vessel.links_to_path.remove(o)
    # apply the links from the other vessels to this one!
    for v in vessels:
        if v != longest_vessel:
            for link in v.links_to_path:
                if link.target_vessel not in vessels:  # no links between them
                    longest_vessel.add_link(
                        Link(
                            longest_vessel,
                            link.source_index,
                            link.target_vessel,
                            link.target_index,
                        )
                    )
                    # update relink
                    for l in link.target_vessel.links_to_path:
                        if l.target_vessel == link.source_vessel:
                            l.target_vessel = longest_vessel
    for v in vessels:
        if v != longest_vessel:
            all_vessels.remove(v)
    return longest_vessel


def find_border_pnodes(G, p_nodes, full_graph_edge_vessels=None):
    """
    Parameters:
    - full_graph_edge_vessels: Precomputed set of vessels from the FULL graph's edges.
    """
    if full_graph_edge_vessels is None:
        # Default to using the provided graph (for backward compatibility)
        full_graph_edge_vessels = set()
        for _, _, data in G.edges(data=True):
            if "vessel" in data:
                full_graph_edge_vessels.add(data["vessel"])

    border_data = []
    for pnode in p_nodes:
        pnode_vessels = pnode.connected_vessels
        # Isolated vessels are those NOT in the FULL graph's edges
        isolated_vessels = set(
            [v for v in pnode_vessels if v not in full_graph_edge_vessels]
        )
        print(
            len(isolated_vessels),
            "/",
            len(pnode_vessels),
            "..",
            len(full_graph_edge_vessels),
        )
        if isolated_vessels:
            border_data.append((pnode, isolated_vessels))
    return border_data


def remove_vessel(vessel, vessels):
    if vessel in vessels:
        vessels.remove(vessel)

    outgoing_links = [
        (link.target_vessel, link.target_index) for link in vessel.links_to_path
    ]

    for v in vessels:
        # print(f"Before processing {v.associated_vesselname}, links:")
        # Capture current links for debug purposes
        # [print(f"  Links: {l.source_vessel.associated_vesselname} -> {l.target_vessel.associated_vesselname}") for l in v.links_to_path]

        # Collect list of links to be removed
        incoming_links = [
            link for link in v.links_to_path if link.target_vessel == vessel
        ]
        # Process removals and new link additions
        for incoming_link in incoming_links:
            v.links_to_path.remove(incoming_link)
            for target_vessel, target_index in outgoing_links:
                if incoming_link.source_vessel != target_vessel:
                    new_link = Link(
                        source_vessel=incoming_link.source_vessel,
                        source_index=incoming_link.source_index,
                        target_vessel=target_vessel,
                        target_index=target_index,
                    )
                    v.add_link(new_link)

        # Output post-processing state for verification
        # print(f"After processing {v.associated_vesselname}, links:")
        # [print(f"  Links: {l.source_vessel.associated_vesselname} -> {l.target_vessel.associated_vesselname}") for l in v.links_to_path]
        # Ensure all links are reprocessed correctly
        for l in v.links_to_path:
            assert l.target_vessel != vessel, "Lingering link found in links_to_path"


def find_border_pnodes(G, p_nodes, all_vessels, full_graph_edge_vessels=None):
    if full_graph_edge_vessels is None:
        # Ensure graph G has edges before proceeding
        full_graph_edge_vessels = {
            data["vessel"] for _, _, data in G.edges(data=True) if "vessel" in data
        }

    border_data = []
    # Determine if graph has nodes and edges for bridge computation
    if G.number_of_edges() == 0:
        print("Graph has no edges.")
        return border_data

    all_bridges = set(nx.bridges(G))  # Identify all bridge edges in the graph

    for pnode in p_nodes:
        # Collect vessels that are not in the predefined graph edge vessels set
        pnode_vessels = pnode.connected_vessels
        isolated_vessels = {
            v for v in pnode_vessels if v not in full_graph_edge_vessels
        }

        if isolated_vessels:
            for vessel in list(isolated_vessels)[:10]:
                # Iterate through associated edges with this vessel
                for u, v, data in G.edges(data=True):
                    # assert data.get("vessel") in all_vessels
                    # assert vessel in all_vessels
                    if str(data.get("vessel")) == str(vessel):

                        if (u, v) in all_bridges or (v, u) in all_bridges:
                            print(
                                f"Vessel {vessel} forms a bridge between nodes {u} and {v}."
                            )

            border_data.append((pnode, isolated_vessels))

    return border_data


def get_borders(vessels, G, p_nodes):
    # Precompute edge vessels from the FULL graph
    full_graph_edge_vessels = {
        data["vessel"] for _, _, data in G.edges(data=True) if "vessel" in data
    }

    # Process each subgraph using the precomputed full graph edge vessels
    border_vessels = []
    vesset = set(vessels)
    for i, component in enumerate(nx.connected_components(G)):
        subgraph = G.subgraph(component).copy()
        border_info = find_border_pnodes(
            subgraph, p_nodes, vesset, full_graph_edge_vessels
        )
        for pnode, isolated_vessels in border_info:
            border_vessels.append(isolated_vessels)
    return border_vessels


def remove_border_leq5(vessels):
    # create graph
    p_nodes = create_p_nodes(vessels)
    # Convert to networkx graph
    G, doubles = create_graph(p_nodes, vessels)
    border_vessels = get_borders(vessels, G, p_nodes)
    print("DOUBLES", len(doubles))
    cutoff_distance = lambda s: 2 if "pulmonary" in s or "cor_vein" in s else 5
    to_rem = [
        v
        for x in border_vessels
        for v in x
        if len(v.path) <= cutoff_distance(v.associated_vesselname)
    ]
    print(len(vessels))
    for v in set(to_rem):
        remove_vessel(v, vessels)
    print(len(vessels), len(to_rem))
    for vess in vessels:
        for l in vess.links_to_path:
            assert l.target_vessel not in to_rem


def solve_resistor_network(graph, ground_nodes, positive_node, V_source=1):
    all_nodes = list(graph.nodes())
    unknown_nodes = [
        n for n in all_nodes if n not in ground_nodes and n != positive_node
    ]

    # Handle case with no unknown nodes
    if not unknown_nodes:
        potentials = {n: 0.0 if n in ground_nodes else V_source for n in all_nodes}
        currents = {}
        for u, v in graph.edges():
            R = graph[u][v]["resistance"]
            current = (potentials[u] - potentials[v]) / R
            currents[(u, v)] = current
        return potentials, currents

    node_to_index = {n: i for i, n in enumerate(unknown_nodes)}
    m = len(unknown_nodes)
    G = np.zeros((m, m))
    b = np.zeros(m)

    for u in unknown_nodes:
        i = node_to_index[u]
        for v in graph.neighbors(u):
            if u == v:
                continue
            try:
                R = graph[u][v]["resistance"]
            except KeyError:
                raise ValueError(f"Resistance not specified for edge {u}-{v}")
            conductance = 1 / R
            G[i][i] += conductance
            if v in node_to_index:
                j = node_to_index[v]
                G[i][j] -= conductance
            elif v == positive_node:
                b[i] += V_source * conductance

    try:
        x = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Matrix is singular. Check if the network is properly connected."
        )

    potentials = {}
    for n in all_nodes:
        if n in ground_nodes:
            potentials[n] = 0.0
        elif n == positive_node:
            potentials[n] = V_source
        else:
            potentials[n] = x[node_to_index[n]]

    currents = {}
    for u, v in graph.edges():
        R = graph[u][v]["resistance"]
        V_u = potentials[u]
        V_v = potentials[v]
        current = (V_u - V_v) / R
        currents[(u, v)] = current

    return potentials, currents


"""

vis.show(to_show)
for s in set(doubles):
    merge_vessels(s,vessels)
"""

"""for i, node1 in enumerate(p_nodes):
        for j, node2 in enumerate(p_nodes):
            if i != j:
                vessels_between = node1.connected_vessels.intersection(node2.connected_vessels)
                if vessels_between:
                    print("Between")
                    #assert len(vessels_between) == 1
                    shared_vessel = list(vessels_between)[0]
                    if len(vessels_between)>1:
                        doubles.append(tuple([v[0] for v in vessels_between]))
                    for shared_vessel in vessels_between:
                        print(shared_vessel)
                        G.add_edge(node1.get_id(), node2.get_id(), resistance=shared_vessel.resistance,vessel=shared_vessel)
    return G,doubles """

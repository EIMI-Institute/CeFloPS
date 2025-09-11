#output generation from simulation results, some gneral plots as well as storage to Disk
import CeFloPS.simulation.settings as settings
import pickle
import numpy as np
import os  # Import the os module
from scipy.interpolate import interp1d
 
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
# ------------------ CELL REGIONS PLOT --------------------------


import numpy as np

import matplotlib.cm as cm

def plot_concentrations(
    concentrations,
    stepsize,
    name="plot",
    upper_limit=-1,
    save=False,
    t_string="",
    y_label="",
    x_label="",
    title="",
    return_bool=False,
    wo_blood=False,
):
    assert "time" in concentrations.keys(), "no time for concentrations in dict"
    time = concentrations["time"]
    regions = [key for key in concentrations.keys() if key != "time"]
    if len(time) < upper_limit or upper_limit < 0:
        upper_limit = len(time) + 1
    tm = np.array(time[0:upper_limit])
    fig = plt.figure()
    ax = plt.axes()
    regions_to_plot = []
    for r in regions:
        if "blood" in r and wo_blood:
            continue
        regions_to_plot.append(r)
    max_values = {}
    for r in regions_to_plot:
        data = concentrations[r][0:upper_limit]
        data_array = np.asarray(data)
        if data_array.size > 0:
            max_val = data_array.max()
        else:
            max_val = 0
        max_values[r] = max_val

    # Sort regions by max value and select top 8
    sorted_regions = sorted(regions_to_plot, key=lambda x: max_values[x], reverse=True)
    top_regions = sorted_regions[:8]

    color = iter(cm.rainbow(np.linspace(0, 1, len(regions))))
    
    
    for i, region in enumerate(regions):
        if "blood" in region and wo_blood:
            continue
        regionl = region if region in top_regions else '_nolegend_'
        if type(region) != str:
            regionl = region.name
        c = next(color)
        y = np.asarray(concentrations[region][0:upper_limit])
        x = tm
        minlen = min(len(x), len(y))
        if minlen == 0:
            continue
        plt.plot(
            x[:minlen],
            y[:minlen],
            color=c,
            label=regionl,
        )
        if "blood" in regionl:
            plt.text(x[minlen-1], y[minlen-1], regionl)
 
     

    if title == "":
        ax.set_title(f"Concentrations: {upper_limit} data_points/c")
    else:
        ax.set_title(title)

    if y_label == "":
        ax.set_ylabel("Concentrations")
    else:
        ax.set_ylabel(y_label)
    if x_label == "":
        ax.set_xlabel(f"Seconds in {stepsize}s steps")
        if stepsize == 60:
            ax.set_xlabel(f"Minutes")
    else:
        ax.set_xlabel(x_label)

    legend = plt.legend(loc="best", framealpha=1, frameon=False, prop={"size": 6})

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # export_legend(legend)
    # plt.legend(loc="best")
    x_interval = tm
    x_t = np.asarray(list(range(0, int(max(tm) / stepsize) + 1))) * stepsize
    plt.xticks(x_t, "")
    if save:
        plt.savefig(f"{t_string}/plt_{name}.pdf")
    elif return_bool:
        return plt.gcf()
    else:
        plt.show()


def normalized(concentration_dict):
    concentrations_normalized = dict()
    concentrations_normalized["time"] = concentration_dict["time"]
    relevant_keys = []
    print("Lens when normalizing concentration dict",len(concentration_dict["time"]),[len(concentration_dict[key]) for key in concentration_dict])
    for key in concentration_dict.keys():
        if key != "time":
            relevant_keys.append(key)
            concentrations_normalized[key] = []
    for i in range(len(concentration_dict["time"])):
        if i<len(concentration_dict[key]):#TODO check why we have different dimensions!
            normalized = [concentration_dict[key][i] for key in relevant_keys ]
            if not sum([concentration_dict[key][i] for key in relevant_keys]) == 0:
                normalized = f.normalize(
                    [concentration_dict[key][i] for key in relevant_keys]
                )

            for k, key in enumerate(relevant_keys):
                concentrations_normalized[key].append(normalized[k])
    return concentrations_normalized

 
def map_region_curves(curves, mapping, times=None):
    mapping["Muscle"].append("beating_heart.systole.pericardium")
    for key in mapping:
        mapping[key] = list(set(mapping[key]))
    # Track mapping counts for double-mapping check
    region_usage = {}
    for roi, reg_list in mapping.items():
        for r in reg_list:
            if r not in region_usage:
                region_usage[r] = []
            region_usage[r].append(roi)

    # Print double-mapped regions
    double_mapped = {r: groups for r, groups in region_usage.items() if len(groups) > 1}
    if double_mapped:
        print("WARNING: Some regions are mapped to multiple groups!")
        for r, group_list in double_mapped.items():
            print(f"  Region '{r}' is mapped to: {group_list}")
        print(f"Total double-mapped regions: {len(double_mapped)}")

    # Continue mapping as before
    curve_regions = set(curves.keys())
    mapped_regions = set()
    mapped_curves = {}
    for roi, reg_list in mapping.items():
        if not reg_list:
            continue
        summed = None
        for r in reg_list:
            mapped_regions.add(r)
            if r in curves:
                data = np.asarray(curves[r])
                if summed is None:
                    summed = np.zeros_like(data)
                summed += data
        if summed is not None:
            mapped_curves[roi] = summed
    # Unmapped warning
    unmapped = curve_regions - mapped_regions
    if unmapped:
        print(f"WARNING: {len(unmapped)} unmapped region(s): {unmapped}")
        if times is not None:
            print("Max sum unmapped regions:",
                  max(np.sum([curves[r] for r in unmapped], axis=0)))
    if times is not None:
        print("Max sum all regions (before mapping):",
              max(np.sum([curves[r] for r in curves.keys()], axis=0)))
        if mapped_curves:
            print("Max sum mapped regions (after mapping):",
                  max(np.sum([mapped_curves[roi] for roi in mapped_curves], axis=0)))
        print(f"Total number of original regions in mapping: {len(region_usage)}")
        print(f"Total number of output mapped groups: {len(mapped_curves)}")
    return mapped_curves

def plot_occupancy_timecourse(trackers, id_to_name, title, ax=None, mapping=None):
    if ax is None:
        ax = plt.gca()
    times, curves = build_occupancy_timecourse(trackers, id_to_name)
    print("[DEBUG] All present region keys:", curves.keys())
    if mapping is not None:
        curves = map_region_curves(curves, mapping, times=times)
    for region, counts in sorted(curves.items()):
        ax.step(times, counts, where='post', label=region)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cells in Region')
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

#for dataframes states tags positions and regions, create a representation of the ordered events that take place. Additionally to the timeline we could slide through the time and update positions like in the graph for it.
#This way one could color code what happens in the timeline and the states with a graph (ie height is state that can go 0 1 2 and color is tag)
#TODO #TODO
#TODO check if all cell_ids are matching after a simulation and stitching them together.





#TODO change to handle tag -1 for eliminated!
def plot_stacked_occupancy_timecourse(trackers, id_to_name, title, ax=None, mapping=None, colormap='tab20'):
    if ax is None:
        ax = plt.gca()
    times, curves = build_occupancy_timecourse(trackers, id_to_name)
    print("[DEBUG] All present region keys:", curves.keys())
    if not times or not curves:
        ax.set_title(title)
        return
    if mapping is not None:
        curves = map_region_curves(curves, mapping, times=times)

    # Gather keys for blood compartments
    region_keys_lower = {k.lower(): k for k in curves.keys()}
    venous_key = region_keys_lower.get('venous')
    arterial_key = region_keys_lower.get('arterial')
    heartchambers_key = region_keys_lower.get('heartchambers')

    blood_regions = []
    blood_counts = []
    if venous_key is not None:
        blood_regions.append("Blood (venous)")
        blood_counts.append(np.array(curves[venous_key]))
    if arterial_key is not None:
        blood_regions.append("Blood (arterial)")
        blood_counts.append(np.array(curves[arterial_key]))
    if heartchambers_key is not None:
        blood_regions.append("Heartchambers")
        blood_counts.append(np.array(curves[heartchambers_key]))

    # All other regions, sorted, excluding the special keys
    special_keys = {venous_key, arterial_key, heartchambers_key}
    non_blood_keys = [k for k in curves if k not in special_keys]
    other_regions = sorted(non_blood_keys)
    other_counts = [np.array(curves[r]) for r in other_regions]

    regions = blood_regions + other_regions
    counts = np.array(blood_counts + other_counts)

    # Use a qualitative colormap for distinct colors
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i) for i in range(len(regions))]

    ax.stackplot(times, counts, labels=regions, step='post', alpha=0.8, colors=colors)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cells in region')
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
# ----------------- DATA PREP -------------------


def add_list_entry_to_dict(combined_dict, key, entry, pos):
    if key in combined_dict:
        if len(combined_dict[key]) == pos:
            # append entry
            combined_dict[key].append(entry)
        else:
            if pos < len(combined_dict[key]):
                combined_dict[key][pos] += entry
            else:
                assert False
    else:
        # add key to dict and init list with element
        combined_dict[key] = [entry]


def update_countlog_using_loclist(combined_dict, loclist, interval=settings.INTERVAL):

    #  cell_locations = [(roi.name, roi.get_cells()) for roi in self.rois] entries interval times

    # print(loclist)
    # print(len(loclist))#1167 lists per call
    # print(len(loclist[0]))#7 rois
    # print(len(loclist[0][0]))# 2 entries name and count
    # assert False

    for iterated, cell_locations in enumerate(loclist):
        # extract values from list and map rois that hav ethe same name
        # [(roi.name, count),...]
        for name, count in cell_locations:
            add_list_entry_to_dict(combined_dict, name, count, iterated)


def create_simlog_from_loclist(merged_count, final_rois):
    # simlog holds all keys and a list of concentrations describing their behaviour over time
    sim_concentration = dict()
    #print(merged_count)
    for key, value in merged_count.items():
        if key != "time":
            # Find the ROI with the matching name
            rois = [roi for roi in final_rois if roi.name == key]

            if not rois:
                print(f"Warning: No ROI found with name '{key}'. Skipping.")
                sim_concentration[key] = None  # or some other appropriate value
            else:
                # Use the volume of the first matching ROI
                volume_ml = rois[0].volume_ml if not "blood" in rois[0].name else 5000
                print(rois[0].name, rois[0].volume_ml)
                sim_concentration[key] = np.asarray(merged_count[key]) / volume_ml
        else:
            sim_concentration[key] = merged_count[key]
    return sim_concentration



def calculate_sim_concentrations(cellcounts):
    sim_count = dict()
    # print("count", cellcounts)

    # print("cellcount",sum(cc))
    for counts in cellcounts:  # [(roi.name, count),...]
        # [process_comms[i][1].get() for i in range(process_count)]
        for name, count in counts:
            if name in sim_count:
                sim_count[name] += count
            else:
                sim_count[name] = count

    # print("globalcount", sim_count.values())
    sim_concentration = dict()
    for key, value in sim_count.items():
        sim_concentration[key] = (
            sim_count[key] / [roi.volume_ml for roi in final_rois if roi.name == key][0]
        )
    return sim_concentration, sim_count


def get_next_keytime(time):
    keytimes = sorted(settings.COLLECT_KEYFRAMES)
    for elem in keytimes:
        if elem > time:
            return elem  # return first (lowest) element thats higher than current time
    return -1


def curr_substate(substate, dic, time):
    sub = substate
    sorted_keys = sorted(dic.keys())
    most_recent_key = None
    for key in sorted_keys:
        if key < time:
            most_recent_key = key
        else:
            break

    if most_recent_key is not None:
        return dic[most_recent_key]
    else:
        return sub


def interpolate_points(points_dict, interval=.1, extrapolate=False):
    # get times and points
    times = sorted(points_dict.keys())
    points = [points_dict[time] for time in times]

    # transpose points to obtain separate lists
    x_points, y_points, z_points = zip(*points)

    # interpolation for axis
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

    # Create new dictionary with regular intervals
    interpolated_points_dict = {}
    start_time = min(times) if extrapolate else times[0]
    end_time = max(times)
    t = start_time
    while t <= end_time:
        interpolated_points_dict[t] = (f_x(t), f_y(t), f_z(t))
        t += interval

    return interpolated_points_dict

import os
from multiprocessing import Pool, cpu_count
# Move this OUTSIDE any other function
def process_cell(args):
    c, cellobject, directory, optionaladdition = args
    ts, path_points, comprs = cellobject  # (cell.times,cell.path,cell.comp_change_times)
    cellsubstate = 0
    next_keytime = get_next_keytime(0)
    points_dict = dict()
    ctime = 0
    for i, t in enumerate(ts):
        ctime += t
        points_dict[ctime] = tuple(path_points[i])
    interpolated_points_dict = interpolate_points(
        points_dict,
        interval=.1,  # settings.MINIMAL_TIME_DIFFERENCE_OUTPUT,
        extrapolate=False,
    )

    text_lines = []
    key_lines = []
    for t, point in interpolated_points_dict.items():
        cellsubstate = curr_substate(cellsubstate, dic=comprs, time=t)
        text_lines.append(f"{point[0]} {point[1]} {point[2]} {t} {cellsubstate}\n")
        if next_keytime != -1 and t >= next_keytime:
            key_lines.append(f"{point[0]} {point[1]} {point[2]} {t}\n")
            next_keytime = get_next_keytime(next_keytime)

    textfile = os.path.join(directory, f"cell{c}_{optionaladdition}.txt")
    keyfile = os.path.join(directory, f"KEYTIMES_cell{c}_{optionaladdition}.txt")

    with open(textfile, "w") as f:
        f.writelines(text_lines)
    with open(keyfile, "w") as f:
        f.writelines(key_lines)

    # Only for debugging or if needed
    # with open("./testime.pickle", "wb") as handle:
    #     pickle.dump(points_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return len(interpolated_points_dict), len(points_dict)

def create_GATE_output(result, directory, optionaladdition="_", num_workers=None):
    import os
    from multiprocessing import Pool, cpu_count

    parent_dir = "."
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)

    # Prepare argument tuples for each cell
    call_args = [(c, cellobject, path, optionaladdition) for c, cellobject in enumerate(result)]
    if num_workers is None:
        num_workers = max(min(cpu_count(), len(result)),2)
        print("-------------->",cpu_count(),num_workers)#TODO this seems to fail with low number of tracked cells
    with Pool(num_workers) as pool:
        stats = pool.map(process_cell, call_args)
    total_interpolated = sum(s[0] for s in stats)
    total_points = sum(s[1] for s in stats)
    print(f"Total: {total_interpolated} interpolated, {total_points} original points")
    
# ------------------------ PLOTTING -------------------------

def save_roi_data(directory, roidata):
    """Saves ROI data to a pickle file."""
    filehandler = open(f"{directory}/roidata.pickle", "wb")
    pickle.dump(roidata, filehandler)
    filehandler.close()
def process_simulation_results(result, directory, final_rois, roi_mapping_str, interval):
    #check which tag is used (objects use tuple (res,0), vector uses (res,1))
    for i, tw in enumerate(result):
            if tw[1]==0:#indicates object
                #change the result to be in old format (ie strip the second tuple part)
                #TODO
                process_simulation_results_object_based(result, directory, final_rois, roi_mapping_str, interval)
            else:
                process_simulation_results_vector_based(result, directory, final_rois, roi_mapping_str, interval)
            break

from CeFloPS.simulation.vectorized_cells_simulation import plot_bound_fractions,plot_occupancy_stack,plot_occupancy_stack_alive_only
def offset_cell_ids(entries, offset):
    return [(e[0]+offset, *e[1:]) for e in entries]
def process_simulation_results_vector_based(result, directory, final_rois, roi_mapping_str, interval):
    tag_int_to_str = []
    tag_str_to_int = {}

    for tag in ["VB", "B",'beating_heart.systole.pericardium']  + [r.name for r in final_rois[:-1]]:
        if tag not in tag_str_to_int:
            code = len(tag_str_to_int)
            tag_str_to_int[tag] = code
            tag_int_to_str.append(tag)
    tag_log_paths = []
    position_log_paths = []
    region_log_paths = []
    state_log_paths = []

    for result, result_type in result:
        if not result or result_type != 1:
            continue
        if result["tag_log"]:
            tag_log_paths.append(result["tag_log"])
        if result["position_log"]:
            position_log_paths.append(result["position_log"])
        if result["region_log"]:
            region_log_paths.append(result["region_log"])
        if result["state_log"]:
            state_log_paths.append(result["state_log"])
    all_tag_df = rebase_cell_ids(tag_log_paths)
    region_df = rebase_cell_ids(region_log_paths)
    position_df = rebase_cell_ids(position_log_paths)
    state_df = rebase_cell_ids(state_log_paths)

    mergedfile = os.path.join(directory, "simulation_pos_log.brotli")
    position_df.to_parquet(mergedfile,compression="brotli", engine="pyarrow", index=False)
    mergedfile = os.path.join(directory, "simulation_tag_log.brotli")
    all_tag_df.to_parquet(mergedfile,compression="brotli", engine="pyarrow", index=False)
    mergedfile = os.path.join(directory, "simulation_region_log.brotli")
    region_df.to_parquet(mergedfile,compression="brotli", engine="pyarrow", index=False)
    mergedfile = os.path.join(directory, "simulation_states_log.brotli")
    state_df.to_parquet(mergedfile,compression="brotli", engine="pyarrow", index=False)
    #mergedfile = os.path.join(directory, "simulation_merged_log.unc")
    #merged_df.to_parquet(mergedfile,compression=None, engine="pyarrow", index=False)
    print(f"Saved merged DataFrames to {mergedfile}")
    for f in tag_log_paths + region_log_paths + position_log_paths + state_log_paths:
        if os.path.exists(f):
            os.remove(f)
        print("Deleted compressed files")
    n_organs=14#TODO adaptive
    c1_indices = set([3 + 2*i for i in range(n_organs)])
    c2_indices = set([4 + 2*i for i in range(n_organs)])
    #plot_bound_fractions(merged_df, c1_indices, c2_indices,output_dir=directory)
    plot_occupancy_stack(all_tag_df,tag_int_to_str,output_dir=directory)
    plot_occupancy_stack_alive_only(all_tag_df,tag_int_to_str,output_dir=directory)



def rebase_cell_ids(file_list):
    dfs = []
    max_cell_id = 0
    for f in file_list:
        if not f: continue
        df = pd.read_parquet(f)
        if 'cell_id' not in df or len(df) == 0:
            continue
        min_id = df['cell_id'].min()
        df['cell_id'] = df['cell_id'] - min_id + max_cell_id
        max_cell_id = df['cell_id'].max() + 1
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def save_data(directory, simlog, countlog, roi_mapping_str):
    """Saves simulation data to pickle files."""
    with open(f"{directory}/simlog.pickle", "wb") as handle:
        pickle.dump(simlog, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{directory}/countlog.pickle", "wb") as handle:
        pickle.dump(countlog, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{directory}/roi_mapping_str.pickle", "wb") as handle:
        pickle.dump(roi_mapping_str, handle, protocol=pickle.HIGHEST_PROTOCOL)

def settings_store_sim_info(directory):
    """Stores simulation information (e.g., time taken)."""
    settings.store_sim_info(directory)

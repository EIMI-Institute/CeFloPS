import random
from CeFloPS.simulation.common.functions import (
    normalize,
)
from CeFloPS.simulation.common.vessel_functions import (
    standard_speed_function,
) 
import CeFloPS.simulation.settings as settings
import sympy
from CeFloPS.simulation.common.vessel2 import TissueLink, Link
import numpy as np
from .three_dim_connection import travel_to
import inspect

 


def check_if_all_options_have_volumes(selection_to_choose_from):
    q_chances = []
    no_vols = False  # set to True if not all have vols
    for (
        target_vesselq,
        target_indexq,
        source_indexq,
    ) in selection_to_choose_from:
        try:
            if (
                target_vesselq.get_volume_by_index(target_indexq) != None
                and type(
                    target_vesselq.get_volume_by_index(target_indexq).get_symval(
                        target_vesselq.get_volume_by_index(target_indexq).Q_1
                    )
                )
                != sympy.Symbol
            ):
                q_chances.append(
                    target_vesselq.get_volume_by_index(target_indexq).get_symval(
                        target_vesselq.get_volume_by_index(target_indexq).Q_1
                    )
                )
            else:
                no_vols = True
        except:
            no_vols = True
            break
    return no_vols, q_chances


def p_link_chances_q(links, simulation=None):
    # define tlink chances if tlinks are included
    # chances=[link.target_vessel.get_volume_by_index(link.target_index) for link in links if type(link)==Link]
    normal_links = [link for link in links if type(link) == Link]
    no_vols, q_chances = check_if_all_options_have_volumes(
        [
            (link.target_vessel, link.target_index, link.source_index)
            for link in normal_links
        ]
    )
    chances = normalize(q_chances)
    # tlinks:
    tlinks = [link for link in links if type(link) != Link]
    t_chances = [
        (
            settings.TCHANCE
            if l.source_index != len(l.source_vessel.path) - 1
            else l.target_tissue.volume_ml
        )
        for l in tlinks
    ]
    t_chances = normalize(t_chances)
    #print(f"[PROBABILITY LINKS] from {[link.source_vessel.associated_vesselname[-32:] for link in normal_links][0]} --> {normal_links + tlinks, chances + t_chances}")
    #print(f"[PROBABILITY LINKS - TARGETS] {[link.target_vessel.associated_vesselname[-32:] for link in normal_links] + [l.target_tissue.name for l in tlinks]}")
    return normal_links + tlinks, chances + t_chances


def p_link_chances_q_heartvess_reduce(links, simulation=None):
    # define tlink chances if tlinks are included
    # chances=[link.target_vessel.get_volume_by_index(link.target_index) for link in links if type(link)==Link]
    normal_links = [link for link in links if type(link) == Link]
    no_vols, q_chances = check_if_all_options_have_volumes(
        [
            (link.target_vessel, link.target_index, link.source_index)
            for link in normal_links
        ]
    )
    for i, link in enumerate(normal_links):
        if (
            "heart" in link.target_vessel.associated_vesselname
            and not "aorta" in link.target_vessel.associated_vesselname
            and "aorta" in link.source_vessel.associated_vesselname
        ):
            q_chances[i] *= 0.1
    chances = normalize(q_chances)
    # tlinks:
    tlinks = [link for link in links if type(link) != Link]
    t_chances = [
        (
            settings.TCHANCE
            if l.source_index != len(l.source_vessel.path) - 1
            else l.target_tissue.volume_ml
        )
        for l in tlinks
    ]
    t_chances = normalize(t_chances)
    return normal_links + tlinks, chances + t_chances


def p_link_chances_r(links, simulation=None):
    # define tlink chances if tlinks are included
    normal_links = [link for link in links if type(link) == Link]
    chances = [l.target_vessel.diameters[l.target_index] for l in normal_links]
    chances = normalize(chances)
    # tlinks:
    tlinks = [link for link in links if type(link) != Link]
    t_chances = [
        (
            settings.TCHANCE
            if len(normal_links) > l.source_index != len(l.source_vessel.path) - 1
            else l.target_tissue.volume_ml
        )
        for l in tlinks
    ]
    t_chances = normalize(t_chances)
    return normal_links + tlinks, chances + t_chances


def p_link_chances_attract(links, simulation):
    # possible for both tlinks and vessellinks
    selection_to_choose_from = [
        [target_vessel, target_index, source_index]
        for target_vessel, target_index, source_index in links
    ]
    chances = [
        sum([r.get_attraction(simulation) for r in rois])
        for rois in [
            vessel.get_rois(n_index)
            for vessel, n_index, index in selection_to_choose_from
        ]
    ]
    return links, chances


def check_if_all_options_have_volumes(selection_to_choose_from):
    q_chances = []
    no_vols = False  # set to True if not all have vols
    for (
        target_vesselq,
        target_indexq,
        source_indexq,
    ) in selection_to_choose_from:
        try:
            if (
                target_vesselq.get_volume_by_index(target_indexq) != None
                and type(
                    target_vesselq.get_volume_by_index(target_indexq).get_symval(
                        target_vesselq.get_volume_by_index(target_indexq).Q_1
                    )
                )
                != sympy.Symbol
            ):
                q_chances.append(
                    target_vesselq.get_volume_by_index(target_indexq).get_symval(
                        target_vesselq.get_volume_by_index(target_indexq).Q_1
                    )
                )
            else:
                no_vols = True
        except:
            no_vols = True
            break
    return no_vols, q_chances

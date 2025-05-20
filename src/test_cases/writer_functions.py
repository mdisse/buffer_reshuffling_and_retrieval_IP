import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(wd, '../..'))

import json


def save_results(filename: str, test_case):
    """
    Saves the results of the experiment as a json file.
    """
    data = dict()
    data['layout_file'] = test_case.instance.get_layout_file()
    data['fill_level'] = test_case.instance.get_fill_level()
    data['max_priority'] = test_case.instance.get_max_p()
    data['height'] = test_case.instance.get_height()
    data['seed'] = test_case.instance.get_seed()
    data['sink'] = test_case.instance.has_sink()
    data['source'] = test_case.instance.has_source()

    data['h_initial'] = int(test_case.h_initial)
    data['runtime'] = test_case.runtime
    data['mipgap'] = test_case.mipgap
    data['solution_found'] = test_case.solution_found

    if not test_case.solution_found:
        data['bay_info'] = dict()
        data['sink_info'] = dict()
        data['source_info'] = dict()
        data['initial_state'] = dict()
        for bay in test_case.instance.wh_initial.bays:
            data['bay_info'][bay.get_id()] = bay.to_data_dict()
            data['initial_state'][bay.get_id()] = bay.state.tolist()
        for sink in test_case.instance.wh_initial.sinks: 
            data['sink_info'][sink.get_id()] = sink.to_data_dict()
        for source in test_case.instance.wh_initial.sources: 
            data['source_info'][source.get_id()] = source.to_data_dict()

        data['access_points'] = []
        for point in test_case.instance.wh_initial.all_access_points:
            data['access_points'].append(point.to_data_dict())

        f = open(filename, 'w')
        json.dump(data, f, indent=4)
        f.close()
        return

    data['upper_bound'] = int(test_case.ub)
    data['f_final'] = int(test_case.solved_node.f)
    data['total_distance'] = float(test_case.solved_node.total_distance)
    data['node_counter'] = test_case.node_counter

    data['moves_history'] = []
    for move in test_case.solved_node.get_moves_history():
        data['moves_history'].append(list(move))
    data['h_history'] = test_case.solved_node.get_h_history()

    data['bay_info'] = dict()
    data['sink_info'] = dict()
    data['source_info'] = dict()
    data['initial_state'] = dict()
    data['final_state'] = dict()
    for bay in test_case.instance.wh_initial.bays:
        data['bay_info'][bay.get_id()] = bay.to_data_dict()
        data['initial_state'][bay.get_id()] = bay.state.tolist()
    for bay in test_case.instance.wh_reshuffled.bays:
        data['bay_info'][bay.get_id()] = bay.to_data_dict()
        data['final_state'][bay.get_id()] = bay.state.tolist()
    for sink in test_case.instance.wh_initial.sinks: 
        data['sink_info'][sink.get_id()] = sink.to_data_dict()
    for source in test_case.instance.wh_initial.sources: 
        data['source_info'][source.get_id()] = source.to_data_dict()

    data['access_points'] = []
    for point in test_case.instance.wh_initial.all_access_points:
        data['access_points'].append(point.to_data_dict())

    data['virtual_lanes'] = []
    for lane in test_case.instance.wh_initial.virtual_lanes:
        data['virtual_lanes'].append(lane.to_data_dict())

    f = open(filename, 'w')
    json.dump(data, f, indent=4)
    f.close()

def save_resultsCP(filename: str, test_case):
    """
    Saves the results of the experiment as a json file.
    """
    data = dict()
    data['layout_file'] = test_case.instance.get_layout_file()
    data['fill_level'] = test_case.instance.get_fill_level()
    data['max_priority'] = test_case.instance.get_max_p()
    data['height'] = test_case.instance.get_height()
    data['seed'] = test_case.instance.get_seed()
    data['sink'] = test_case.instance.has_sink()
    data['source'] = test_case.instance.has_source()

    data['h_initial'] = int(test_case.h_initial)
    data['runtime'] = test_case.runtime
    data['cp_runtime'] = test_case.cp_runtime
    data['cp_buildtime'] = test_case.cp_buildtime
    data['worker_number'] = test_case.worker_number
    data['solution_found'] = test_case.solution_found
    data['solution_optimal'] = test_case.solution_optimal
    data["virtual_lanes_clear"] = [[int(x) for x in sublist] for sublist in test_case.virtual_lanes]
    data["access_point_number"] = test_case.access_point_number

    if not test_case.solution_found:

        data['bay_info'] = dict()
        data['sink_info'] = dict()
        data['source_info'] = dict()
        data['initial_state'] = dict()
        for bay in test_case.instance.wh_initial.bays:
            data['bay_info'][bay.get_id()] = bay.to_data_dict()
            data['initial_state'][bay.get_id()] = bay.state.tolist()
        for sink in test_case.instance.wh_initial.sinks: 
            data['sink_info'][sink.get_id()] = sink.to_data_dict()
        for source in test_case.instance.wh_initial.sources: 
            data['source_info'][source.get_id()] = source.to_data_dict()

        data['access_points'] = []
        for point in test_case.instance.wh_initial.all_access_points:
            data['access_points'].append(point.to_data_dict())

        f = open(filename, 'w')
        json.dump(data, f, indent=4)
        f.close()
        return

    data['upper_bound'] = None  # int(test_case.ub)
    data['f_final'] = None  # int(test_case.solved_node.f)
    data['total_distance'] = test_case.solved_node.total_distance  # float(test_case.solved_node.total_distance)
    data['node_counter'] = None  # test_case.node_counter

    data['moves_history'] = test_case.solved_node.history
    data["dist_history"] = test_case.distance_history

    data['h_history'] = None  # test_case.solved_node.get_h_history()

    data['bay_info'] = dict()
    data['sink_info'] = dict()
    data['source_info'] = dict()
    data['initial_state'] = dict()
    data['final_state'] = dict()
    for bay in test_case.instance.wh_initial.bays:
        data['bay_info'][bay.get_id()] = bay.to_data_dict()
        data['initial_state'][bay.get_id()] = bay.state.tolist()
    for bay in test_case.instance.wh_reshuffled.bays:
        data['bay_info'][bay.get_id()] = bay.to_data_dict()
        data['final_state'][bay.get_id()] = bay.state.tolist()
    for sink in test_case.instance.wh_initial.sinks: 
        data['sink_info'][sink.get_id()] = sink.to_data_dict()
    for source in test_case.instance.wh_initial.sources: 
        data['source_info'][source.get_id()] = source.to_data_dict()

    data['access_points'] = []
    for point in test_case.instance.wh_initial.all_access_points:
        data['access_points'].append(point.to_data_dict())

    data["access_directions"] = test_case.access_directions

    data['virtual_lanes'] = []
    for lane in test_case.instance.wh_initial.virtual_lanes:
        data['virtual_lanes'].append(lane.to_data_dict())

    f = open(filename, 'w')
    json.dump(data, f, indent=4)
    f.close()

def save_resultsBrr(filename: str, test_case):
    """
    Saves the results of the experiment as a json file.
    """
    data = dict()
    data['layout_file'] = test_case.instance.get_layout_file()
    data["access_directions"] = test_case.instance.access_directions
    data['fill_level'] = round(test_case.instance.get_fill_level(), 2)
    data['height'] = test_case.instance.get_height()
    data['seed'] = test_case.instance.get_seed()
    data['fleet_size'] = test_case.instance.get_fleet_size()
    data['vehicle_speed'] = test_case.instance.get_vehicle_speed()
    data['handling_time'] = test_case.instance.get_handling_time()
    data['unit_loads'] = []
    for ul in test_case.instance.unit_loads:
        data['unit_loads'].append(ul.to_data_dict())

    data['results'] = test_case.results

    data['bay_info'] = dict()
    data['sink_info'] = dict()
    data['source_info'] = dict()
    data['initial_state'] = dict()
    for source in test_case.instance.wh_initial.sources: 
        data['source_info'][source.get_id()] = source.to_data_dict()
    for bay in test_case.instance.wh_initial.bays:
        data['bay_info'][bay.get_id()] = bay.to_data_dict()
        data['initial_state'][bay.get_id()] = bay.state.tolist()
    for sink in test_case.instance.wh_initial.sinks: 
        data['sink_info'][sink.get_id()] = sink.to_data_dict()

    data['access_points'] = []
    for point in test_case.instance.wh_initial.all_access_points:
        data['access_points'].append(point.to_data_dict())

    data['virtual_lanes'] = []
    for lane in test_case.instance.wh_initial.virtual_lanes:
        data['virtual_lanes'].append(lane.to_data_dict())


    f = open(filename, 'w')
    json.dump(data, f, indent=4)
    f.close()
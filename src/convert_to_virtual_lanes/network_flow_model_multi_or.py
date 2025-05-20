import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from tupledict import TupleDict

from bay.virtual_lane import VirtualLane


class SolutionPrinterNFM(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, flow, arc, solution_limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._solution_count = 0
        self._flow = flow
        self._used_arc = arc
        self._v_flow = []
        self._v_used_arc = []
        self._solution_limit = solution_limit

    def on_solution_callback(self):

        self._solution_count += 1
        # print('Solution %i' % self._solution_count)

        used_arc_dict = dict()
        for arc in self._used_arc:
            used_arc_dict[arc] = self.Value(self._used_arc[arc])
        # print(used_arc_dict)

        flow_list_dict = dict()
        for flow in self._flow:
            flow_list_dict[flow] = self.Value(self._flow[flow])
        # print(flow_list_dict)

        self._v_flow.append(flow_list_dict)
        self._v_used_arc.append(used_arc_dict)

        # if self._solution_limit == 10:
        # print('Solution %i' % self._solution_count)
        # print(flow_list_dict)
        # print(used_arc_dict)

        if self._solution_count >= self._solution_limit:
            # print('Stop search after {} solutions for NFM solutions.'.format(self._solution_limit))
            self.StopSearch()


def get_position_last_misplaced(sequence_lane):
    """Count items above zero until the last misplaced item per lane"""
    counter1 = -1  # counter for tier level; Set 0 to get correctly placed
    counter2 = 0  # counter for stack number
    max_tier = sequence_lane.shape[1]  # maximum height of a stack
    # print("tier_max", max_tier)
    value = 999  # import sys and sys.maxsize or float('inf')
    found = False  # Variable shows, if misplaced position has been found
    sequence_lane_flip = np.flip(sequence_lane, axis=0)
    for stack in sequence_lane_flip:
        stack = stack[::-1]  # reverse stack [first level, second level], because second level access first
        for h in stack:
            if h <= value:  # If h is greater or equal than previous h (value)
                if found is False:  # If misplaced item has not been found yet
                    if h != 0:
                        value = h  # Set value to current priority h
                else:  # If misplaced item has been found, count + 1
                    counter1 += 1
                    if counter1 == max_tier:
                        counter2 += 1
                        counter1 = 0
            else:
                if h > value:  # Misplaced item detected
                    found = True
                    counter1 += 1
                    if counter1 == max_tier:
                        counter2 += 1
                        counter1 = 0
    if counter1 == -1:  # No misplaced items, returns None
        return
    return counter2, counter1


def get_position_last_leading_zero(sequence_lane):
    """Count leading zeros until first non-zero. Counter starts at -1 to get the position of the last empty spot/zero"""
    counter1 = -1  # counter for tier level
    counter2 = 0  # counter for stack number
    max_tier = sequence_lane.shape[1]  # maximum height of a stack
    for stack in sequence_lane:
        for h in stack:
            if counter1 == max_tier:
                counter2 += 1
                counter1 = 0
            if h == 0:
                counter1 += 1
            else:
                if counter1 == -1:  # No zeros in front, returns None
                    return
                return counter2, counter1
    return counter2, counter1


# Network flow model to get virtual lanes
class NetworkFlowModelMultiOR():
    def __init__(self, bay, optimal_objective_value=None):
        super().__init__()
        # Bay config infos

        self.state = bay.state
        self.access_points = bay.access_points
        self.width = np.shape(bay.state)[0]
        self.length = np.shape(bay.state)[1]
        self.tiers = np.shape(bay.state)[2]
        self.access = self.get_access_directions(bay.access_points)
        # print(f'l{self.length} w{self.width} h{self.tiers}')

        # Network flow model
        self.m = cp_model.CpModel()
        # self.m.reset()
        self.v_used_arc = TupleDict()
        self.arcs = None
        self.cost = None
        self.cost_restricted = None
        self.demand = None
        self.v_flow = TupleDict()
        self.north_row = None
        self.south_row = None
        self.east_column = None
        self.west_column = None
        self.inner_nodes = None
        self.get_sequence_arc = None  # Index to get the sequence to calculate the costs of each arc; For each stack [column_index, row_index]
        self.access_directions_list = []
        self.generate_model(optimal_objective_value)

    def get_access_directions(self, access_points):
        access_direction = [False, False, False, False]
        for a in access_points:
            # print(a)
            if a.direction == "north":
                # print(access_direction[0])
                access_direction[0] = True
            if a.direction == "south":
                access_direction[1] = True
            if a.direction == "west":
                access_direction[2] = True
            if a.direction == "east":
                access_direction[3] = True
        # print(access_direction)
        return access_direction

    def get_solutions(self):
        """
        Returns:
        List of solutions. Solution is a list of virtual lanes.
        """
        solution, optimal_objective_value = self.run_model()

        solutions = []

        for i in range(self.n_solutions):

            if optimal_objective_value == 0:
                stack_indices = self.get_stack_indices_for_each_lane(solution, i)
                virtual_lanes_output = self.derive_virtual_lanes(stack_indices, solution)

                solutions.append(virtual_lanes_output)

        # print(virtual_lanes_output)
        return solutions, optimal_objective_value

    def derive_virtual_lanes(self, stack_indices, solution):
        virtual_lanes_list = []
        new_ap_id = None  # sets ID for access point
        for index, lane in enumerate(stack_indices):
            virtual_lane_state = []  # Stack object --> Save index and value
            for index2, stack_index in enumerate(lane):
                column_index = (stack_index - 1) // self.length
                row_index = (stack_index - 1) % self.length
                virtual_lane_state.append(self.state[column_index][row_index])
                if index2 == 0:
                    for i in self.access_points:
                        if column_index == i.stack_x and row_index == i.stack_y:
                            new_ap_id = i.ap_id
                        else:
                            continue
            vl = VirtualLane()
            vl.stacks = virtual_lane_state
            vl.ap_id = new_ap_id
            # Get ap_id via stack_x and stack_y
            virtual_lanes_list.append(vl)
        # for x in virtual_lanes_list:
        #     print(f'stakcks {x.stacks},ap_id {x.ap_id}')
        return virtual_lanes_list  # np.array(bay_state, dtype="object")  # , None , bay_state

    # objs = [MyClass() for i in range(10)]
    # for obj in objs:
    #     other_object.add(obj)
    # objs[0].do_sth()

    def get_stack_indices_for_each_lane(self, solution, i):
        stacks = {}  # dict with from to location
        duplicate_stacks = {}  # dict for two duplicates if two edges from one source node
        directions_edges = []  # entry stack for each lane

        for edge in self.cost:
            if solution._v_used_arc[i][edge] >= (1 - 1e-09) and solution._v_flow[i][edge] >= (1 - 1e-09) and edge[
                0] != "o":  # Get only used arcs
                if edge[0] == "north" or edge[0] == "south" or edge[0] == "west" or edge[0] == "east":
                    directions_edges.append(edge)
                else:
                    if edge[
                        0] not in stacks.keys():  # and edge[1] not in stacks.values():  # edge[0] is from the same stack but different direction, egde[1] is to the stack but from different dirction
                        stacks[edge[0]] = edge[1]
                    else:
                        duplicate_stacks[edge[0]] = edge[1]
        lanes = []
        visited_stacks = []  # prevent loops
        for index, lane in enumerate(directions_edges):
            lanes.append([int(lane[1][1:])])  # entry stack; Debug point: index == 33
            visited_stacks.append(lane[1])
            previous = lane[1]
            while previous is not None:
                try:
                    if previous in duplicate_stacks.keys():  # Postprocessing for two successors from a stack (flow going back)
                        successor_stacks = [int(duplicate_stacks[previous][1:]),
                                            int(stacks[previous][1:])]  # both successor stacks
                        difference = int(lanes[index][0]) - int(
                            lanes[index][1])  # Get difference to see if successor has a bigger or smaller ID
                        # print(difference)
                        if difference > 0:
                            stack_id = min(successor_stacks)
                        else:
                            stack_id = max(successor_stacks)
                        lanes[index].append(int(stack_id))
                        previous = f"s{stack_id}"
                        if previous not in visited_stacks:
                            visited_stacks.append(previous)
                    elif previous in duplicate_stacks.values() and previous in stacks.values():  # Postprocessing for two incoming edges
                        predecessor_stacks = [
                            list(duplicate_stacks.keys())[list(duplicate_stacks.values()).index(previous)][1:],
                            stacks[previous][1:]]  # both stacks inbound edges
                        difference = int(lanes[index][0]) - int(
                            lanes[index][1])  # Get difference to see if successor has a bigger or smaller ID
                        if difference < 0:
                            stack_id = min(predecessor_stacks)
                        else:
                            stack_id = max(predecessor_stacks)
                        lanes[index].append(int(stack_id))
                        previous = f"s{stack_id}"
                        if previous not in visited_stacks:
                            visited_stacks.append(previous)
                    elif stacks[previous] not in visited_stacks:  # Check if successor has not been visited
                        lanes[index].append(int(stacks[previous][1:]))
                        if previous not in visited_stacks:
                            visited_stacks.append(previous)
                        previous = stacks[previous]
                    else:
                        previous = None
                except KeyError:
                    previous = None
        # Change access for outer stacks to the respective direction # Better way would be to do it in pre-processing and build a smaller model!
        directions_edges_stack_ids = []
        for edge in directions_edges:
            directions_edges_stack_ids.append(int(edge[1][1:]))
        # print("directions_edges_stack_ids", directions_edges_stack_ids)
        unvisited_directions_edges_stack_ids = []
        for edge in self.cost:
            if edge[0] == "north" or edge[0] == "south" or edge[0] == "west" or edge[0] == "east":
                if int(edge[1][1:]) in directions_edges_stack_ids:
                    continue
                else:
                    unvisited_directions_edges_stack_ids.append(int(edge[1][1:]))
        new_lanes = []
        additional_lanes = []
        for idx, lane in enumerate(lanes):
            new_lanes.append([])
            for stack_id in lane:
                if stack_id not in unvisited_directions_edges_stack_ids:
                    new_lanes[idx].append(stack_id)
                else:
                    additional_lanes.append([stack_id])
        new_lanes.extend(additional_lanes)
        # print(new_lanes)
        return new_lanes

    def run_model(self):
        # Creates the solver and solve.
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        # solver.parameters.linearization_level = 0
        # solver.parameters.max_time_in_seconds = 999999
        solver.parameters.enumerate_all_solutions = True

        solver.Solve(self.m, self.solution_printer)

        # self.m.ExportToFile('model.pb.txt')

        self.n_solutions = self.solution_printer._solution_count

        return self.solution_printer, solver.ObjectiveValue()

    def generate_model(self, optimal_objective_value=None):
        self.create_cost_dict_from_scratch()
        self.create_demand_dict()

        for arc in self.arcs:
            # print('used_arc_{}_{}'.format(arc[0], arc[1]))
            self.v_flow[arc[0], arc[1]] = self.m.NewIntVar(0, self.width * self.length,
                                                           'flow_{}_{}'.format(arc[0], arc[1]))
            self.v_used_arc[arc[0], arc[1]] = self.m.NewBoolVar('used_arc_{}_{}'.format(arc[0], arc[1]))

        self.m.v_flow = self.v_flow
        self.m.v_used_arc = self.v_used_arc

        objective_terms = []
        for arc in self.arcs:
            objective_terms.append(self.m.v_used_arc[(arc[0], arc[1])] * (self.cost[arc]))

        if optimal_objective_value == None:
            self.m.Minimize(sum(objective_terms))
            self.solution_printer = SolutionPrinterNFM(self.m.v_flow, self.m.v_used_arc, solution_limit=9999)
        else:
            # self.m.Minimize(sum(objective_terms))
            # print("New Constraint")
            self.m.Add(
                sum(self.m.v_used_arc[(arc[0], arc[1])] * int(self.cost[arc] * 10000) for arc in self.arcs) <= int(
                    round(optimal_objective_value * 10000)))
            self.solution_printer = SolutionPrinterNFM(self.m.v_flow, self.m.v_used_arc, solution_limit=10)

        big_m = self.width * self.length
        for arc in self.arcs:
            self.m.Add(self.v_flow[(arc[0], arc[1])] <= self.m.v_used_arc[(arc[0], arc[1])] * big_m)

        nodes_without_origin = list(self.demand.keys())
        nodes_without_origin.remove("o")

        for i in nodes_without_origin:
            # print(self.m.v_flow[None, i])
            try:
                self.m.Add(sum(self.m.v_flow[None, i]) == sum(self.m.v_flow[i, None]) + self.demand[i])
            except KeyError:
                self.m.Add(sum(self.m.v_flow[None, i]) == self.demand[i])

        self.m.Add(sum(self.m.v_flow["o", None]) == self.demand["o"])

        # Avoid turnings:
        for i in self.inner_nodes:
            # Surrounding neighbors: -1 above, 1 bellow, length to the right, -length to the left
            north_neighbor = "s{}".format(i - 1)
            south_neighbor = "s{}".format(i + 1)
            west_neighbor = "s{}".format(i - self.length)
            east_neighbor = "s{}".format(i + self.length)
            stack = "s{}".format(i)
            if i not in self.north_row and i not in self.west_column:  # in case a stack is in the back (2 access directions and turnings should be avoided)
                self.m.Add(self.m.v_used_arc[(north_neighbor, stack)] + self.m.v_used_arc[(stack, west_neighbor)] <= 1)
                self.m.Add(self.m.v_used_arc[(west_neighbor, stack)] + self.m.v_used_arc[(stack, north_neighbor)] <= 1)

            if i not in self.north_row and i not in self.east_column:
                self.m.Add(self.m.v_used_arc[(north_neighbor, stack)] + self.m.v_used_arc[(stack, east_neighbor)] <= 1)
                self.m.Add(self.m.v_used_arc[(east_neighbor, stack)] + self.m.v_used_arc[(stack, north_neighbor)] <= 1)

            if i not in self.south_row and i not in self.west_column:
                self.m.Add(self.m.v_used_arc[(south_neighbor, stack)] + self.m.v_used_arc[(stack, west_neighbor)] <= 1)
                self.m.Add(self.m.v_used_arc[(west_neighbor, stack)] + self.m.v_used_arc[(stack, south_neighbor)] <= 1)

            if i not in self.south_row and i not in self.east_column:
                self.m.Add(self.m.v_used_arc[(south_neighbor, stack)] + self.m.v_used_arc[(stack, east_neighbor)] <= 1)
                self.m.Add(self.m.v_used_arc[(east_neighbor, stack)] + self.m.v_used_arc[(stack, south_neighbor)] <= 1)

    def create_demand_dict(self):
        # creates the supply, through and demand dictionaries
        # Each node and supply/demand; transshipment with 0; Capacities not required
        self.demand = dict({'o': self.length * self.width})  # Supply equals amount of stacks
        if self.access[0]:
            self.demand['north'] = 0
        if self.access[1]:
            self.demand['south'] = 0
        if self.access[2]:
            self.demand['west'] = 0
        if self.access[3]:
            self.demand['east'] = 0
        # position number for each stack (s1, s2,...) starting with 1
        position_number = 1
        for width in range(self.width):
            for length in range(self.length):
                dictionary_string = "s{}".format(position_number)
                self.demand[dictionary_string] = 1  # Each stack has to be visited once {'s1': 1, 's2': 1, 's3': 1 ...}
                position_number += 1

    def create_cost_dict_from_scratch(self):
        # create a list containing all nodes as string ( 'o', 'north' ...) or integers (1, 2, 3 ...) in a list
        nodes_to_loop_through = [*range(1, self.length * self.width + 1)]
        self.north_row = set(np.arange(1, self.length * self.width, self.length))
        self.south_row = set(np.arange(self.length, self.length * self.width + 1, self.length))
        self.west_column = set(np.arange(1, self.length + 1, 1))
        self.east_column = set(np.arange(self.length * (self.width - 1) + 1, self.length * self.width + 1, 1))
        self.inner_nodes = set(nodes_to_loop_through)
        nodes_to_loop_through.extend(['o'])
        if self.access[0]:
            self.access_directions_list.extend(['north'])
            self.inner_nodes = self.inner_nodes - self.north_row
        if self.access[1]:
            self.access_directions_list.extend(['south'])
            self.inner_nodes = self.inner_nodes - self.south_row
        if self.access[2]:
            self.access_directions_list.extend(['west'])
            self.inner_nodes = self.inner_nodes - self.west_column
        if self.access[3]:
            self.access_directions_list.extend(['east'])
            self.inner_nodes = self.inner_nodes - self.east_column
        nodes_to_loop_through.extend(self.access_directions_list)
        self.create_network(nodes_to_loop_through)
        self.create_get_sequence_lane()
        for arc in self.arcs:
            self.cost[arc] = self.update_costs_in_cost_dict(arc)

    def create_network(self, nodes_to_loop_through):
        cost_dict_input = {}
        get_sequence_arc_input = []
        # now loop through the pair of node_1 and node_2. In two for-loops every possible node combination is selected.
        for node_1 in nodes_to_loop_through:
            for node_2 in nodes_to_loop_through:
                if node_1 != 'o' and node_2 in ['o', 'north', 'east', 'south',
                                                'west']:  # no edge is pointing from inside the grid to the outside nodes( 'o', 'north' ...)
                    continue
                if node_1 == 'o' and node_2 not in ['north', 'east', 'south',
                                                    'west']:  # the origin node is only allowed to point towards the compass direction nodes
                    continue
                if node_1 == node_2:
                    continue
                if self.neighbour_check(node_1,
                                        node_2) is False:  # Now skip the node pair if the nodes are not each others neighbours
                    continue
                # Now we can insert the calculated cost value for the selected node pair into the dictionary
                # create the node name for example node 1 becomes  "S1", if the node is an integer...
                if type(node_1) == int:
                    dict_node_1 = 's{}'.format(node_1)
                else:  # if the node is a compass direction, just leave it as for example 'north'
                    dict_node_1 = node_1
                if type(node_2) == int:
                    dict_node_2 = 's{}'.format(node_2)
                    # Add all arcs to get sequence arc, which have a stack as second node (arcs with costs)
                    get_sequence_arc_input.append((node_1, node_2))
                else:
                    dict_node_2 = node_2
                # add the cost value for the selected nodes into the dictionary
                cost_dict_input[dict_node_1, dict_node_2] = None
        self.get_sequence_arc = get_sequence_arc_input
        self.arcs, self.cost = gp.multidict(cost_dict_input)

    def create_get_sequence_lane(self):
        get_sequence_arc_dict_input = {}  # All stack positions in state to calculate sequence & costs for arcs (keys)
        for arc in self.get_sequence_arc:  # loop through the list of arcs (node_1, node_2) without origin
            node_1 = arc[0]
            node_2 = arc[1]
            dict_node_2 = 's{}'.format(node_2)
            try:  # as long as node_2 is not a string (cardinal direction), then except
                dict_node_1 = 's{}'.format(node_1)
                difference = node_1 - node_2  # get the difference to know the direction of the arc
                if difference == -1:
                    # the indices from the upper border of the warehouse until node_2(the destination node)
                    if node_2 % self.length == 0:
                        # special case if the destination node is on the lower border, np.arange.stop has to be adjusted
                        sequence_lane_indices = np.arange(node_2, (node_2 // self.length - 1) * self.length, step=-1)
                    else:
                        sequence_lane_indices = np.arange(node_2, (node_2 // self.length) * self.length, step=-1)
                if difference == 1:
                    # the indices from node_1 until the lower border of the warehouse
                    sequence_lane_indices = np.arange(node_2, ((node_2 // self.length) + 1) * self.length + 1)
                if difference == self.length:
                    sequence_lane_indices = np.arange(node_2, self.length * self.width + 1, step=self.length)
                if difference == -self.length:
                    sequence_lane_indices = np.arange(node_2, 0, step=-self.length)
                # indices are generated now extract the bay elements of the given indices
                list_stack_indices = []
                for i in reversed(sequence_lane_indices):
                    column_index = (i - 1) // self.length  # The bay state is referenced with bay[column, row]
                    row_index = (
                                        i - 1) % self.length  # the referenced elements in the bay start from zero on -> S1 = element zero
                    # if sequence_lane is bigger than only two elements, then change the way you concatenate the elements
                    if column_index == self.width:
                        column_index = self.width - 1
                        row_index = self.length - 1
                    list_stack_indices.append([column_index, row_index])
                get_sequence_arc_dict_input[dict_node_1, dict_node_2] = list_stack_indices
            except TypeError:  # if node_1 is not int:  # If node_1 is a cardinal direction
                dict_node_1 = node_1
                column_index = (node_2 - 1) // self.length
                row_index = (
                                    node_2 - 1) % self.length  # the referenced elements in the bay start from zero on -> S1 = element zero
                if column_index == self.width:
                    column_index = self.width - 1
                    row_index = self.length - 1
                get_sequence_arc_dict_input[dict_node_1, dict_node_2] = [[column_index, row_index]]
        self.get_sequence_arc = get_sequence_arc_dict_input  # Overwrite list of arcs with dict of arcs and indices

    def update_costs_in_cost_dict(self, arc):
        try:
            sequence_indices = self.get_sequence_arc[arc]
            sequence_arc = []  # sequence for each arc
            for index in sequence_indices:
                sequence_arc.append(self.state[index[0], index[1]])
            sequence_arc = np.vstack(sequence_arc)  # [[0 2] [4 2]]
            cost_increment = self.get_cost_increment_per_arc(sequence_arc)
        except KeyError:  # if node_2 is not int (no stack), the costs are 0
            cost_increment = 0.0001  # cost_increment = 0  # 0.0001
        if arc[0] is 'o':  # cost increment zero for access directions! Otherwise they might be avoided
            cost_increment = 0
        return cost_increment

    def neighbour_check(self, node_1, node_2):
        if self.access[0]:
            # No edges between nodes of the north row (if access north is true)
            if node_1 in self.north_row and node_2 in self.north_row:
                return False
            # The origin is connected to the compass directions
            if node_1 == 'o' and node_2 == "north":
                return True
            if node_1 == 'north':  # the north node has edges to the most north row of the warehouse
                if node_2 in self.north_row:
                    return True  # node_2 is in most north row of warehouse -> neighbour_check = True!
                else:
                    return False
        if self.access[1]:
            # nodes on the south side
            if node_1 in self.south_row and node_2 in self.south_row:
                return False
            if node_1 == 'o' and node_2 == "south":
                return True
            if node_1 == 'south':
                if node_2 in self.south_row:
                    return True
                else:
                    return False
        if self.access[2]:
            # nodes on the west side
            if node_1 in self.west_column and node_2 in self.west_column:
                return False
            if node_1 == 'o' and node_2 == "west":
                return True
            if node_1 == 'west':
                if node_2 in self.west_column:
                    return True
                else:
                    return False
        if self.access[3]:
            # nodes on the east side
            if node_1 in self.east_column and node_2 in self.east_column:
                return False
            if node_1 == 'o' and node_2 == "east":
                return True
            if node_1 == 'east':
                if node_2 in self.east_column:
                    return True
                else:
                    return False
        # if the nodes are neither the compass directions nor the origin 'o' calculate the surrounding stack numbers of node_1 and check whether node_2 is within them
        # Surrounding neighbors: -1 above, 1 bellow, length to the right, -length to the left
        for i in [-1, 1, self.length,
                  -self.length]:  # loop through the elements in the grid: -1 -> the previous element, length the element next to the right ...
            if i == -1 and ((node_1 - 1) % self.length) != 0:
                if node_1 + i == node_2:
                    return True
            if i == 1 and node_1 % self.length != 0:
                if node_1 + i == node_2:
                    return True
            if i == self.length and ((node_1 + self.length) <= (self.length * self.width)):
                if node_1 + i == node_2:
                    return True
            if i == -self.length and (node_1 - self.length > 0):
                if node_1 + i == node_2:
                    return True
        return False

    def get_cost_increment_per_arc(self, sequence):
        # the cost increment for the edge of two neighbors node_1 to node_2 is calculated
        cost_value_current = self.cost_calculation(sequence)
        if len(sequence) > 1:
            sequence2 = sequence[:-1]  # Remove the last/target stack
            cost_value_previous = self.cost_calculation(sequence2)
            cost_value_increment = cost_value_current - cost_value_previous
        else:  # if we were at the border of the warehouse just return the current cost_value
            cost_value_increment = cost_value_current
        if cost_value_increment == 0:  # Change to small costs to avoid turnings
            cost_value_increment = 0.0001  # 0.0001
        return cost_value_increment

    def cost_calculation(self, sequence):
        # The cost of a node pair is calculated here
        if get_position_last_misplaced(sequence) is None:
            cost_value = 0  # if no misplaced item is found the cost for this node pair is zero
        else:
            cost_tuple = get_position_last_misplaced(sequence)
            cost_value = cost_tuple[0] * self.tiers + cost_tuple[1] + 1
            zeros_tuple = get_position_last_leading_zero(sequence)
            if zeros_tuple is None:
                zeros_counter = 0
            else:
                zeros_counter = zeros_tuple[0] * self.tiers + zeros_tuple[1] + 1
            cost_value = cost_value - zeros_counter
        # add costs of 1, if the sequence has any holes / Swiss Cheese
        previous_item = -1
        for stack in sequence:
            for current_item in stack:
                if previous_item > current_item == 0:
                    cost_value += 999  # Adds 999 to costs for each hole/Swiss Cheese slot
                previous_item = current_item
        return cost_value

    def print_model_results(self, obj):
        # Activate to see flow and used arcs in output console
        product_flow = pd.DataFrame(columns=["From", "To", "Flow"])
        pd.set_option('display.max_rows', None)  # To see all lines
        for arc in self.arcs:
            # if flow[arc].x > 1e-6:
            product_flow = product_flow.append(
                {"From": arc[0], "To": arc[1], "Flow": self.v_flow[arc].x, "Used Arc": self.v_used_arc[arc].x},
                ignore_index=True)
        product_flow.index = [''] * len(product_flow)
        print(product_flow)
        print("Objective value:", int(obj.getValue()))

    def generate_graph(self):
        # Generate visualization via networkX
        G = nx.DiGraph()
        # G.clear()
        # Simple Graph without position
        # for k in all_nodes:
        #     G.add_node(k)
        # Grid-graph: Calculate position for each node
        x = 0
        start = 0
        for s in range(start, self.length * self.width, self.length):
            y = 0
            for j in range(s, s + self.length):
                node_value = j + 1
                node = "s{}".format(node_value)
                G.add_node(node, pos=(x, y))
                y -= 10
            x += 10
        start += self.length
        x -= 10  # Remove added parameter of last loop
        y += 10  # Remove added parameter of last loop
        # Add origin node to graph
        # G.add_node("o", pos=(-20, y / 4))
        # Add directions node
        if self.access[0]:
            G.add_node("north", pos=(x / 2, 10))

        if self.access[1]:
            G.add_node("south", pos=(x / 2, y - 10))

        if self.access[2]:
            G.add_node("west", pos=(-10, y / 2))

        if self.access[3]:
            G.add_node("east", pos=(x + 10, y / 2))

        label_gen = {}
        for edge in self.cost:
            # print("edge", edge)
            if self.v_used_arc[edge].x >= (1 - 1e-09) and self.v_flow[edge].x >= (1 - 1e-09) and edge[
                0] != "o":  # Show only used arcs
                label_gen[edge[0], edge[1]] = "{}|{}".format(int(self.v_flow[edge].x), int(self.cost[edge[0], edge[1]]))
                G.add_edge(edge[0], edge[
                    1])  # , weight=cost[edge[0], edge[1]])  # flow[edge].x)  # weight_graph)  # cost[x[0], x[1]])
        # Default spring layout instead of grid
        # pos = nx.spring_layout(G)
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, font_weight='bold')
        # labels = nx.get_edge_attributes(G, 'weight')
        labels = label_gen
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        number_int = 1
        number = str(number_int)
        plt.savefig('network_flow_chart' + number + '.png', dpi=200)
        plt.clf()
        number_int += 1

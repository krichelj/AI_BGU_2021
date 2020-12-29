import pandas as pd
from typing import Any, Dict
from copy import deepcopy
from collections import OrderedDict
from numpy import prod
from treelib import Tree

from Graph import Graph, Vertex, Edge, DirectedEdge

LEAKAGE_PROBABILITY = 0.001

config_from_site = '''
#N 4                 ; number of vertices n in graph (from 1 to n)

#V1 F 0              ; Vertex 1, no evacuees for sure
#V2 F 0.4            ; Vertex 2, probability of evacuees 0.4
#V3 F 0              ; Either assume evacuees probability 0 by default,
#V4 F 0              ; or make sure to specify this probability for all vertices.

#E1 1 2 W1           ; Edge1 between vertices 1 and 2, weight 1
#E2 2 3 W3           ; Edge2 between vertices 2 and 3, weight 3
#E3 3 4 W3           ; Edge3 between vertices 3 and 4, weight 3
#E4 2 4 W4           ; Edge4 between vertices 2 and 4, weight 4
                     ; Either assume blocking probability 0 by default,
                     ; or make sure to specify this probability for all edges.
#Ppersistence 0.9    ; Set persistence probability to 0.9
'''

config_from_notes = '''
#N 3

#V1 F 0.3
#V2 F 0.1
#V3 F 0

#E1 1 2 W1
#E2 1 3 W3

#Ppersistence 0.9
'''

config_from_video_lecture = '''
#N 3

#V1 F 0.3
#V2 F 0.1
#V3 F 0

#E1 1 2 W1
#E2 1 3 W3

#Ppersistence 0.9
'''


def parse_config_string(config_string):
    N = 0
    persistence = 0.0
    vertices_config = {}
    edges_config = {}

    for line in config_string.split('\n'):
        if len(line) and line[0] == '#':
            identifier = line[1]

            if identifier == 'N':
                N = int(line[3:].split(' ')[0])
            elif identifier == 'V':
                splitted = line[1:].split(' ')[0:3]
                v = splitted[0]
                p = splitted[2]
                vertices_config[v] = float(p)
            elif identifier == 'P':
                persistence = float(line[1:].split(' ')[1])
            else:
                splitted = line[1:].replace('W', '').split(' ')[0:4]
                e, v1_num, v2_num, w = splitted
                v1v2 = sorted([v1_num, v2_num])
                v1_id = 'V' + v1v2[0]
                v2_id = 'V' + v1v2[1]
                edges_config[v1_id + v2_id] = (v1_id, v2_id, int(w))

    return N, persistence, vertices_config, edges_config


def get_binary_array(n: int, j: int):
    B = 0
    res = []

    for i in range(1, 2 ** n + 1):
        res.append(B)
        if i % ((2 ** n) / (2 ** j)) == 0:
            B = 1 - B

    return res


class RandomVariable(Vertex):
    def __init__(self, rv_id: str, parents: OrderedDict, p: float = None, t: int = None, w: int = None,
                 persistence: float = None, x: float = None, y: float = None):
        self.parents: OrderedDict[str, RandomVariable] = parents
        self.sons: Dict[str, RandomVariable] = OrderedDict()
        super().__init__(v_id=rv_id, p_people=p, x=x, y=y)
        self.t = t
        self.w = w
        self.persistence = persistence
        self.p = 0.0 if len(parents) else p
        self.initial_probability_values = self.construct_initial_table() if p is None else p
        self.evidence_probability_value = 0.0

        for parent in self.parents.values():
            parent.sons[self.v_id] = self

    def construct_initial_table(self):
        n = len(self.parents)
        data = {parent_id: get_binary_array(n, j + 1) for j, parent_id in enumerate(self.parents.keys())}

        probability_column_name = 'P(' + self.v_id + ')'

        if self.t == 0:
            p = 0.6 * 1 / self.w
            data[probability_column_name] = [0.001, p, p, 1 - (1 - p) ** 2]
        else:
            data[probability_column_name] = [LEAKAGE_PROBABILITY, self.persistence]

        probability_table = pd.DataFrame.from_dict(data)

        for row in probability_table.itertuples():
            row = list(row[1:])
            p_given = row[-1]
            assignment = row[:-1]

            for i, parent in enumerate(self.parents.values()):
                p_parent = parent.p

                if assignment[i] == 1:
                    p_given *= p_parent
                else:
                    p_given *= 1 - p_parent

            self.p += p_given

        return probability_table


class BayesianNetwork(Graph):
    def __init__(self, vertices: Dict[Any, Vertex], edges: Dict[Any, Edge], T: int, persistence: float):

        n = len(vertices)
        m = len(edges)
        random_variables = OrderedDict()
        random_variables.update({
            v_id: RandomVariable(rv_id=v_id, parents=OrderedDict(), p=v.p_people if v.p_people else 0, x=i / n, y=1)
            for i, (v_id, v) in enumerate(vertices.items())})
        bayesian_edges = {}

        for t in range(T + 1):
            for i, (e_id, e) in enumerate(edges.items()):
                if t == 0:
                    curr_parents = OrderedDict({v.v_id: random_variables[v.v_id] for v in e.Vs})
                else:
                    parent_id = e_id + '_' + str(t - 1)
                    curr_parents = OrderedDict({parent_id: random_variables[parent_id]})

                curr_rv = RandomVariable(rv_id=e_id + '_' + str(t), parents=curr_parents, t=t, w=e.w,
                                         persistence=persistence, x=i / m, y=1 - (t + 1) / T)

                bayesian_edges.update(
                    {parent.v_id + e_id: DirectedEdge(parent.v_id + e_id, curr_rv, parent, parent.v_id)
                     for parent_id, parent in curr_parents.items()})

                random_variables[e_id + '_' + str(t)] = curr_rv

        super().__init__(random_variables, bayesian_edges)
        self.evidence: OrderedDict[str, bool] = OrderedDict()
        self.vars: OrderedDict[str: RandomVariable] = random_variables
        self.evaluation_tree_counter = 1

    def set_one_evidence(self, X_id: str, e: bool):
        self.evidence[X_id] = e

    @staticmethod
    def print_evidence(evidence: Dict):
        output = ' | '

        for e, b in evidence.items():
            output += e + ' = ' + str(b) + ', '

        if len(evidence):
            output = output[:-2]
        else:
            output = ''

        return output

    def discard_barren_nodes(self, current_evidence: Dict[str, bool]) -> Dict[str, RandomVariable]:
        trimmed_vars = deepcopy(self.vars)

        exist_nodes_to_discard = any([var_id for var_id, var in trimmed_vars.items() if not len(var.sons)
                                      and var_id not in current_evidence.keys()])

        while exist_nodes_to_discard:
            vars_to_trim = [var_id for var_id, var in trimmed_vars.items() if not (len(var.sons)
                                                                                   or var_id in current_evidence.keys())]
            for var_id in vars_to_trim:
                for var_parent in trimmed_vars[var_id].parents.values():
                    del var_parent.sons[var_id]

            trimmed_vars = OrderedDict({var_id: var for var_id, var in trimmed_vars.items()
                                        if var_id not in vars_to_trim})

            exist_nodes_to_discard = any([var_id for var_id, var in trimmed_vars.items() if not len(var.sons)
                                          and var_id not in current_evidence.keys()])

        return trimmed_vars

    def enumeration_ask(self, X_id: str):

        if X_id in self.evidence.keys():
            return int(self.evidence[X_id])

        evaluation_tree = Tree()
        evaluation_tree.create_node(tag='P(' + X_id + BayesianNetwork.print_evidence(self.evidence) + ')',
                                    identifier=1)

        current_evidence = self.evidence.copy()
        current_evidence[X_id] = True
        trimmed_vars = self.discard_barren_nodes(current_evidence)

        alpha = self.normalize() if len(self.evidence) else 1
        if alpha == 0:
            return 'Not defined'
        print('alpha: ' + str(alpha))
        outcome = self.enumerate_all(trimmed_vars, current_evidence, evaluation_tree, 1)

        evaluation_tree.show()
        self.evaluation_tree_counter = 1

        return outcome / alpha

    def enumerate_all(self, trimmed_vars: Dict[str, RandomVariable], current_evidence: Dict[str, bool],
                      evaluation_tree: Tree, parent_node_id: int):

        if not len(trimmed_vars):
            return 1.0

        trimmed_vars = trimmed_vars.copy()
        Y_id, Y = trimmed_vars.popitem(last=False)

        Y_parents = Y.parents

        if len(Y_parents):
            Y_parents_assignments = {parent_id: int(current_evidence[parent_id]) for parent_id in Y_parents.keys()}
            Y_parents_assignments_string = ''.join([str(x) for x in Y_parents_assignments.values()])
            Y_parents_assignments_key = int(Y_parents_assignments_string, 2)
            # print(Y_parents_assignments_key)
            p_y_given_Y_parents = Y.initial_probability_values['P(' + Y_id + ')'].iloc[Y_parents_assignments_key]
        else:
            p_y_given_Y_parents = Y.initial_probability_values

        if Y_id in current_evidence.keys():
            y = current_evidence[Y_id]
            res = p_y_given_Y_parents if y else 1 - p_y_given_Y_parents

            parents_assignments = {p_id: current_evidence[p_id] for p_id in Y.parents.keys()}
            self.evaluation_tree_counter += 1
            evaluation_tree.create_node(tag=('not ' if not y else '') +
                                            'P(' + Y_id + BayesianNetwork.print_evidence(parents_assignments) + ') = '
                                            + str(res), identifier=self.evaluation_tree_counter,
                                        parent=parent_node_id)

            if res == 0:
                return 0

            output = res * self.enumerate_all(trimmed_vars, current_evidence, evaluation_tree,
                                              self.evaluation_tree_counter)
            # print(output)
        else:
            output = 0.0

            for y in [True, False]:
                curr_evidence = current_evidence.copy()
                curr_res = p_y_given_Y_parents if y else 1 - p_y_given_Y_parents

                if curr_res == 0:
                    continue

                curr_evidence[Y_id] = y

                parents_assignments = {p_id: current_evidence[p_id] for p_id in Y.parents.keys()}
                self.evaluation_tree_counter += 1
                evaluation_tree.create_node(tag=('not ' if not y else '') +
                                                'P(' + Y_id + BayesianNetwork.print_evidence(
                    parents_assignments) + ') = '
                                                + str(curr_res), identifier=self.evaluation_tree_counter,
                                            parent=parent_node_id)

                res = curr_res * self.enumerate_all(trimmed_vars, curr_evidence, evaluation_tree,
                                                    self.evaluation_tree_counter)
                # print(res)

                output += res

        return output

    def normalize(self):
        return prod([self.vars[E_id].p if self.evidence[E_id] else 1 - self.vars[E_id].p
                     for E_id in self.evidence.keys()])


def add_piece_of_evidence(BN: BayesianNetwork):
    available_vars = {i + 1: var_id for i, var_id in enumerate(BN.vars.keys()) if
                      var_id not in BN.evidence.keys()}
    available_vars_string = ''

    for i, var in available_vars.items():
        available_vars_string += '\t' + str(i) + '. ' + var + '\n'

    var_name = input('Choose the random variable name to assign evidence\n' + available_vars_string)
    e = input('Choose the evidence assignment\n0. False\n1. True\n')
    BN.set_one_evidence(available_vars[int(var_name)], bool(int(e)))


def get_vertices_probabilities(graph: Graph, BN: BayesianNetwork):
    vertices = graph.get_vertices()

    evidence_string = BayesianNetwork.print_evidence(BN.evidence)

    for v_id in vertices.keys():
        print('#' * 30 + ' v = ' + v_id + ' ' + '#' * 30)
        n = v_id[1:]
        p = BN.enumeration_ask(v_id)

        print('Vertex ' + n)
        print('\tP(Evacuees ' + n + evidence_string + ') = ' + str(p))
        print('\tP(not Evacuees ' + n + evidence_string + ') = ' +
              (str(1 - p) if p != 'Not defined' else p))
        print()


def get_edges_probabilities(graph: Graph, T: int, BN: BayesianNetwork):
    edges = graph.get_edges()
    evidence_string = BayesianNetwork.print_evidence(BN.evidence)

    for t in range(T + 1):
        print('#' * 60 + ' t = ' + str(t) + ' ' + '#' * 60)
        for e_id, e in edges.items():
            print('#' * 30 + ' e = ' + e_id + ' ' + '#' * 30)
            id_as_vertices = e.get_id_as_vertices()

            var_id = id_as_vertices + '_' + str(t)

            # if var_id in ['V1V2_1']:
            #     print(BN.vars[var_id].initial_probability_values)

            p = BN.enumeration_ask(var_id)

            print('Edge ' + id_as_vertices + ', time ' + str(t))
            print('\tP(Blockage ' + id_as_vertices + evidence_string + ') = ' + str(p))
            print('\tP(not Blockage ' + id_as_vertices + evidence_string + ') = ' +
                  (str(1 - p) if p != 'Not defined' else p))
            print()


def get_path_probability(graph: Graph, BN: BayesianNetwork, val: str):
    if val == '5':
        while True:
            t = input('Please specify a time to check for blockages in a path\n')

            if t.isdigit():
                break
            else:
                print('Please input a non-negative int')
    else:
        t = 0

    while True:
        edges = graph.get_edges()
        input_edges = input('Please specify a path by edge names separated by spaces\n').split(' ')
        print(input_edges)
        if graph.is_path(input_edges):

            total_path_probability = 1
            probabilities = {}
            initial_evidence = BN.evidence.copy()

            for e_id in input_edges:
                if e_id not in edges.keys():
                    e_id = e_id[2:4] + e_id[0:2]

                var_id = e_id + '_' + str(t)

                if var_id in BN.evidence.keys() and BN.evidence[var_id]:
                    probabilities[e_id] = 0

                p_blocked = BN.enumeration_ask(var_id)
                BN.set_one_evidence(var_id, False)

                if p_blocked != 'Not defined':
                    p_not_blocked = 1 - p_blocked
                    total_path_probability *= p_not_blocked
                    probabilities[e_id] = p_not_blocked
                else:
                    total_path_probability = -1
                    break

                if val == '6':
                    t += 1

            BN.evidence = initial_evidence

            print('The individual probabilities of the edges to be free are: ' + str(probabilities))
            print('The probability that the path (' + ' '.join(
                input_edges) + ') is free from blockages at time ' + str(t) + ' is ' +
                  (str(total_path_probability) if total_path_probability != -1 else ' not defined'))

            break

        else:
            print('The edges specified do not form a valid path in the graph')


def run_simulation(config):
    N, persistence, vertices_config, edges_config = parse_config_string(config)

    graph = Graph.from_config(vertices_config, edges_config)
    T = 2
    BN = BayesianNetwork(graph.get_vertices(), graph.get_edges(), T, persistence)

    menu = """
    1. Reset evidence list to empty
    2. Add piece of evidence to evidence list
    3. Infer the probability that each of the vertices contains evacuees
    4. Infer the probability that each of the edges is blocked
    5. Infer the probability that a certain path is free from blockages at a specified time
    6. Infer the probability that a certain path is free from blockages at progressing time steps
    7. Quit
    """
    graph.plot()

    while True:
        val = input(menu)

        if val == '1':
            BN.evidence = OrderedDict()
        elif val == '2':
            add_piece_of_evidence(BN)
        elif val == '3':
            get_vertices_probabilities(graph, BN)
        elif val == '4':
            get_edges_probabilities(graph, T, BN)
        elif val in ['5', '6']:
            get_path_probability(graph, BN, val)
        elif val == '7':
            break


if __name__ == '__main__':
    run_simulation(config_from_site)

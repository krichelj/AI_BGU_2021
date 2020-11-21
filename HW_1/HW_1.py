# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from math import sin, cos, pi
import heapq
from typing import List, Dict, Any
import sys
from abc import ABC, abstractmethod
import concurrent
import copy
from termcolor import colored
import textwrap
import itertools

config1 = '''
#N 6
#D 40            
#V1 P6
#V2
#V3 P2
#V4
#V5 P7
#V6 P1

#E1 1 2 W4
#E2 2 3 W5
#E3 3 6 W1
#E4 2 4 W3
#E5 4 5 W2
#E6 2 5 W1
'''

config2 = '''
#N 20
#D 4.5             
#V1
#V2 P1
#V3
#V4 P2
#V5
#V6 P1
#V7
#V8 P4
#V9
#V10 P2
#V11 P3
#V12
#V13 P5
#V14 P4
#V15
#V16 P7
#V17 P2
#V18
#V19 P4
#V20 P3

#E1 1 2 W8
#E2 3 4 W8
#E3 2 3 W6
#E4 1 3 W4
#E5 2 4 W5
#E6 3 5 W6
#E7 3 6 W4
#E8 3 9 W7
#E9 4 7 W3
#E10 4 11 W1
#E11 11 15 W5
#E12 7 14 W6
#E13 14 18 W2
#E14 18 19 W3
#E15 12 19 W4
#E16 6 19 W1
#E17 10 13 W5
#E18 10 17 W4
#E19 5 16 W2
#E20 16 17 W3
#E21 17 20 W5
#E22 1 20 W6
#E23 20 5 W2
#E23 20 7 W4
#E24 4 8 W3
#E25 11 7 W2
#E26 12 14 W5
#E27 2 13 W2
#E28 5 14 W4
#E29 8 5 W1
'''

config3 = '''
#N 7
#D 20 
#V1          
#V2 P1            
#V3 P2               
#V4 P4            
#V5 P5
#V6
#V7 P6

#E1 1 3 W2  
#E2 1 2 W3           
#E3 3 2 W4                
#E4 1 6 W1  
#E5 1 7 W5
#E6 3 6 W1  
#E7 3 4 W7           
#E8 4 5 W4                
#E9 7 4 W10  

'''


def parse_config_string(config_string):
    N = 0
    D = 0
    vertices_config = {}
    edges_config = {}

    for line in config_string.split('\n'):
        if len(line):
            identifier = line[1]

            if identifier == 'N':
                N = int(line[3:])
            elif identifier == 'D':
                D = float(line[3:])
            elif identifier == 'V':
                p = 0
                splitted = line[1:].replace('P', '').split(' ')
                if len(splitted) > 1 and len(splitted[1]):
                    v, p = splitted[0:2]
                else:
                    v = splitted[0]
                vertices_config[v] = int(p)
            else:
                splitted = line[1:].replace('W', '').split(' ')[0:4]
                e, v1_num, v2_num, w = splitted
                v1v2 = sorted([v1_num, v2_num])
                v1_id = 'V' + v1v2[0]
                v2_id = 'V' + v1v2[1]
                edges_config[v1_id + v2_id] = (v1_id, v2_id, int(w))

    return N, D, vertices_config, edges_config


class Vertex:
    def __init__(self, v_id, n_people: int, x: float = None, y: float = None):
        self.v_id = v_id
        self.n_people = n_people
        self.visited = False
        self.x = x
        self.y = y
        self.edges = {}
        self.dist = sys.maxsize
        self.init = False

    def __lt__(self, other):
        return (self.dist, self.v_id) < (other.dist, other.v_id)

    def __str__(self):
        return self.v_id


class Edge:
    def __init__(self, e_id, V1: Vertex, V2: Vertex, w: int):
        self.e_id = e_id
        self.Vs = {V1, V2}
        self.w = w
        self.blocked = False

    def __str__(self):
        return self.e_id

    def __eq__(self, other):
        return self.e_id == other.e_id

    def __hash__(self):
        return hash(self.e_id)

    def __lt__(self, other):
        return (self.w, self.e_id) < (other.w, other.e_id)

    def get_other_vertex(self, v):
        return (self.Vs - {v}).pop()

    def get_vertices_ids(self):
        v_id = self.e_id[0:2]
        u_id = self.e_id[2:4]

        return v_id, u_id


class Graph:
    def __init__(self, vertices: Dict[Any, Vertex], edges: Dict[Any, Edge]):
        self._vertices = vertices
        self._edges = edges

    def get_vertices(self):
        return self._vertices

    def get_edges(self):
        return self._edges

    @staticmethod
    def add_edges_to_vertices(edges):
        for e_id, e in edges.items():
            for v in e.Vs:
                v_edges = v.edges
                if e_id not in v_edges:
                    v_edges[e_id] = e

    @classmethod
    def from_config(cls, vertices_config: Dict[str, int], edges_config: Dict[str, tuple]):
        n = len(vertices_config)
        vertices = {v_id: Vertex(v_id, n_people, cos(2 * pi * i / n), sin(2 * pi * i / n))
                    for i, (v_id, n_people) in enumerate(vertices_config.items())}
        edges = {e_id: Edge(e_id, vertices[e_tup[0]], vertices[e_tup[1]], e_tup[2])
                 for e_id, e_tup in edges_config.items()}

        Graph.add_edges_to_vertices(edges)

        return cls(vertices, edges)

    def plot(self):
        V_x = []
        V_y = []
        V_x_people = []
        V_y_people = []
        V_x_init = []
        V_y_init = []

        fig, ax = plt.subplots(dpi=100)

        for v in self._vertices.values():
            if v.init:
                V_x_init.append(v.x)
                V_y_init.append(v.y)
            elif v.n_people > 0:
                V_x_people.append(v.x)
                V_y_people.append(v.y)
            else:
                V_x.append(v.x)
                V_y.append(v.y)

            ax.annotate(s=v.v_id, xy=(v.x, v.y))

        ax.scatter(V_x_init, V_y_init, color="g", label='init', s=200)
        ax.scatter(V_x, V_y, color="b", label='shapes', s=200)
        ax.scatter(V_x_people, V_y_people, color="r", label='people', s=200)

        for e in self._edges.values():
            V_x = []
            V_y = []

            w_x = 0
            w_y = 0

            for v in e.Vs:
                V_x.append(v.x)
                V_y.append(v.y)

                w_x += v.x
                w_y += v.y

            ax.plot(V_x, V_y, color="b", linewidth=0.3)
            plt.text(w_x / 2, w_y / 2, str(e.w))

        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       labelbottom=False, right=False, left=False,
                       labelleft=False)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def Dijkstra(self, v_id):
        vertices = self._vertices

        for v in vertices.values():
            v.dist = sys.maxsize

        vertices[v_id].dist = 0
        Q = list(vertices.values())

        while len(Q):
            heapq.heapify(Q)
            u = heapq.heappop(Q)

            for e in u.edges.values():
                v = e.get_other_vertex(u)

                if v in Q:
                    alt = u.dist + e.w
                    if alt < v.dist:
                        v.dist = alt

    def Kruskal(self) -> int:
        min_spanning_tree_cost = 0
        vertex_sets = {v_id: v_set for v_set, v_id in enumerate(self._vertices.keys())}
        ordered_edges = sorted(self._edges.values())

        for e in ordered_edges:
            v_id, u_id = e.get_vertices_ids()
            v_set = vertex_sets[v_id]
            u_set = vertex_sets[u_id]

            if v_set != u_set:
                min_spanning_tree_cost += e.w

                for k_id, k_set in vertex_sets.items():
                    if k_set == u_set:
                        vertex_sets[k_id] = v_set

        return min_spanning_tree_cost

    def Prim(self, init_id) -> int:
        vertices = self.get_vertices()
        n = len(vertices)
        init = vertices[init_id]
        forest_vertices_ids = {init_id}
        edges_heap = list(init.edges.values())
        min_spanning_tree_edges = []
        min_spanning_tree_cost = 0

        while len(forest_vertices_ids) < n:
            heapq.heapify(edges_heap)
            e = heapq.heappop(edges_heap)
            v_id, u_id = e.get_vertices_ids()

            if v_id not in forest_vertices_ids:
                k_id = v_id
            elif u_id not in forest_vertices_ids:
                k_id = u_id
            else:
                continue

            edges_heap += list(vertices[k_id].edges.values())

            min_spanning_tree_edges.append(e)
            forest_vertices_ids.add(k_id)
            min_spanning_tree_cost += e.w

        return min_spanning_tree_cost


class Problem:
    def __init__(self, D: float, vertices_config: Dict[str, int], edges_config: Dict[str, tuple]):
        self.deadline = D
        self.graph = Graph.from_config(vertices_config, edges_config)
        self.metric_closure_graph = self.create_metric_closure()

        for v in self.graph.get_vertices().values():
            v.dist = sys.maxsize
        self.graph.plot()

    def run_agents(self, agents_config: Dict[Any, List]):
        for agent_id, agent_config in agents_config.items():
            agent_type, V0_id = agent_config
            agent = (HumanAgent(agent_id, self) if agent_type == 'Human' else
                     (GreedyAgent(agent_id, self, self.graph.get_vertices()[V0_id])
                      if agent_type == 'Greedy' else
                      SaboteurAgent(agent_id, self, self.graph.get_vertices()[V0_id], agent_config[2])))

            agent.run()

    def run_AI_agents(self, agents_config: Dict[Any, List], heuristic: str):
        for agent_id, agent_config in agents_config.items():
            agent_type = agent_config[0]
            V0_id = agent_config[1]

            agent = (AIGreedyAgent(agent_type, agent_id, self,
                                   self.graph.get_vertices()[V0_id],
                                   heuristic) if agent_type == 'Greedy' else
                     (AIAStarAgent(agent_type, agent_id, self,
                                   self.graph.get_vertices()[V0_id], heuristic) if agent_type == 'AStar' else
                      AIAStarRealTimeAgent(agent_type, agent_id, self,
                                           self.graph.get_vertices()[V0_id], heuristic, agent_config[2])))
            agent.goal_test()

    def create_metric_closure(self):
        graph = self.graph
        vertices = graph.get_vertices()
        terminals = {v_id: Vertex(v_id, v.n_people, v.x, v.y) for v_id, v in
                     vertices.items()}
        metric_edges = {}

        for v_id, v in terminals.items():
            graph.Dijkstra(v_id)

            for u_id, u in terminals.items():
                if u_id != v_id:
                    uv = sorted([v_id[1], u_id[1]])
                    e_id = 'V' + uv[0] + 'V' + uv[1]

                    if e_id not in metric_edges:
                        potential_edge = Edge(e_id, u, v, vertices[u_id].dist)
                        terminals[v_id].edges[e_id] = potential_edge
                        terminals[u_id].edges[e_id] = potential_edge
                        metric_edges[e_id] = potential_edge

        metric_closure_graph = Graph(terminals, metric_edges)
        return metric_closure_graph


class State:
    def __init__(self, prb: Problem, current_vertex: Vertex,
                 people: Dict[Any, bool], current_path_cost: int,
                 current_path: List[str], real_time_cluster: int = None):
        self.current_path_cost = current_path_cost
        self.prb = prb
        self.current_vertex = current_vertex
        self.people = people.copy()
        self.is_terminal = False

        if self.current_vertex.v_id in self.people and \
                self.people[self.current_vertex.v_id]:
            self.people[self.current_vertex.v_id] = False
            self.is_terminal = True

        self.path = current_path + [self.current_vertex.v_id]
        self.current_metric_graph = self.get_current_metric_graph()
        self.g_value = -1
        self.h_value = -1
        self.f_value = self.get_f_value()
        self.real_time_cluster = real_time_cluster

        # state_string = 'Potential Vertex: ' + colored(self.current_vertex.v_id, 'blue') + ', people: ' + \
        #       str(self.people) + ', f:  ' + str(self.f_value)
        # people_string = 'Metric terminals: ' + \
        #       str({v_id for v_id in self.current_metric_graph.get_vertices().keys()}) + \
        #       '\nMetric edges: ' + \
        #       str({' '.join([v.v_id for v in e.Vs]): e.w for e in self.current_metric_graph.get_edges().values()})
        #       # '\nNum of edges: ' + str(len(self.current_metric_graph.get_edges()))
        # print(state_string)
        # print('\n'.join(textwrap.wrap(people_string, 120)))

    def get_heuristic_value(self, heuristic):

        # steiner_ratio = 2*(1-1/len(self.current_metric_graph.get_vertices()))
        steiner_ratio = 1

        if not any(self.people.values()):
            res = 0
        elif heuristic == 'Kruskal MST':
            mst = self.current_metric_graph.Kruskal()
            res = mst / steiner_ratio
        elif heuristic == 'Prim MST':
            mst = self.current_metric_graph.Prim(self.current_vertex.v_id)
            res = mst / steiner_ratio
        else:
            shortest_paths_to_terminals = self.perform_Dijkstra_to_terminals()
            res = max(shortest_paths_to_terminals)

        # print(res)
        return res

    def get_successors(self):

        cluster_counter = itertools.count()

        def raise_son(e: Edge):
            son = State(self.prb, e.get_other_vertex(self.current_vertex),
                        self.people, self.current_path_cost + e.w,
                        self.path, next(cluster_counter) if (self.real_time_cluster
                                                             and self.real_time_cluster == -1) else self.real_time_cluster)
            # fig1 = plt.gcf()
            # fig1.canvas.start_event_loop(sys.float_info.min)
            # son.current_metric_graph.plot()
            # fig1.clf()
            # plt.close()

            return son

        successors = list(map(raise_son, self.current_vertex.edges.values()))

        # with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        #     successors = list(executor.map(raise_son, self.current_vertex.edges.values()))

        return successors

    def get_f_value(self):
        pass

    def __lt__(self, other):
        return (self.f_value, self.h_value, self.current_vertex.v_id) < \
               (other.f_value, other.h_value, other.current_vertex.v_id)

    def perform_Dijkstra_to_terminals(self):
        self.prb.graph.Dijkstra(self.current_vertex.v_id)
        shortest_paths_to_terminals = [self.prb.graph.get_vertices()[v_id].dist for
                                       v_id, b in self.people.items() if b]
        return shortest_paths_to_terminals

    def get_current_metric_graph(self):

        def is_edge_valid(e_id, terminal_vals, u_id=None):
            v1_id = e_id[0:2]
            v2_id = e_id[2:4]
            res = {v1_id, v2_id}.issubset(terminal_vals)

            if u_id:
                res = res and u_id in [v1_id, v2_id]

            return res

        v_id = self.current_vertex.v_id
        metric_graph = self.prb.metric_closure_graph
        metric_vertices = metric_graph.get_vertices()
        metric_edges = metric_graph.get_edges()
        v = metric_vertices[v_id]

        current_terminals = {u_id: u for u_id, u in metric_vertices.items()
                             if u_id in self.people and self.people[u_id]}
        if v_id not in current_terminals:
            current_terminals[v_id] = v
        terminal_vals = set(current_terminals.keys())

        for u in current_terminals.values():
            u_id = u.v_id
            u.edges = {e_id: e for e_id, e in metric_edges.items() if
                       is_edge_valid(e_id, terminal_vals, u_id)}

        current_metric_edges = {e_id: e for e_id, e in
                                metric_edges.items()
                                if is_edge_valid(e_id, terminal_vals)}

        current_metric_graph = Graph(current_terminals, current_metric_edges)

        return current_metric_graph


# @title
class Agent(ABC):
    @abstractmethod
    def __init__(self, agent_id, prb: Problem, V0: Vertex = None):
        self.agent_id = agent_id
        self.prb = prb
        self.terminated = False
        self.current_vertex = V0

    @abstractmethod
    def run(self):
        pass

    def traverse(self, new_edge):
        self.current_vertex = (new_edge.Vs - {self.current_vertex}).pop()
        self.prb.deadline -= new_edge.w

    def terminate(self):
        self.terminated = True
        print('The agent ' + self.agent_id + ' has terminated')


class HumanAgent(Agent):
    def __init__(self, agent_id, prb: Problem):
        super().__init__(agent_id, prb)

    def run(self):
        self.prb.graph.plot()


class GreedyAgent(Agent):
    def __init__(self, agent_id, prb: Problem, V0: Vertex):
        super().__init__(agent_id, prb, V0)
        self.saved_people = V0.n_people
        self.current_vertex.n_people = 0
        self.current_vertex.visited = True

    def run(self):
        while not self.terminated:
            minimal_edge = None
            for e in self.current_vertex.edges:
                if not e.blocked and any([v.n_people > 0 for v in e.Vs]):
                    minimal_edge = e
                    break

            if minimal_edge and minimal_edge.w <= self.prb.deadline:
                self.traverse(minimal_edge)
            else:
                self.terminate()

    def traverse(self, new_edge):
        super().traverse(new_edge)
        self.saved_people += self.current_vertex.n_people
        self.current_vertex.n_people = 0
        self.current_vertex.visited = True


class SaboteurAgent(Agent):
    def __init__(self, agent_id, prb: Problem, V0: Vertex, V: int):
        super().__init__(agent_id, prb, V0)
        self.current_vertex = V0
        self.V = V

    def run(self):
        self.prb.deadline -= self.V
        while not self.terminated:
            minimal_edge = None
            if len(self.current_vertex.edges):
                self.current_vertex.edges[0].blocked = True
                self.prb.deadline -= 1

            for e in self.current_vertex.edges[1:]:
                if not e.blocked:
                    minimal_edge = e
                    break

            if minimal_edge and minimal_edge.w <= self.prb.deadline:
                self.traverse(minimal_edge)

            else:
                self.terminate()

    def traverse(self, new_edge):
        super().traverse(new_edge)


class AIAgent(ABC):
    LIMIT = 10000

    @abstractmethod
    def __init__(self, agent_type: str, agent_id, prb: Problem, V0: Vertex,
                 heuristic: str):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.heuristic = heuristic
        self.prb = prb
        V0.init = True
        init_people = {v.v_id: True for v in self.prb.graph.get_vertices().values() \
                       if v.n_people > 0}
        self.goal = {v_id: False for v_id in init_people.keys()}
        self.current_state = State(self.prb, V0, init_people, 0, [],
                                   -1 if self.agent_type == 'AStarRealTime' else None)
        self.min_heap = [self.current_state]
        self.expansions = 0
        self.terminated = False
        self.root_sons = {}

    def goal_test(self):
        while not self.terminated:

            heapq.heapify(self.min_heap)
            self.current_state = heapq.heappop(self.min_heap)
            s = self.current_state
            v = s.current_vertex
            people = s.people
            # if self.expansions == 0:
            # print('Current Vertex: ' + colored(v_id, 'red') + ', f value: ' + str(s.f_value))
            # print('------------------------------------------------------')
            if people == self.goal:
                self.terminate('success')
            elif self.agent_type == 'AStar' and self.expansions == AIAgent.LIMIT:
                self.terminate('limit_reached')
            elif self.agent_type == 'AStarRealTime' and self.expansions > 0 and self.expansions % self.L == 0:
                cont_state = self.root_sons[self.current_state.real_time_cluster]
                cont_state.real_time_cluster = -1
                self.root_sons = {}
                self.min_heap = [cont_state]
                self.expansions += 1
                continue

            # if self.expansions == 1:
            #   break
            # else:
            self.expand()
            self.expansions += 1

    def expand(self):

        potential_sons = self.current_state.get_successors()

        if self.agent_type == 'AStarRealTime' and self.current_state.real_time_cluster == -1:
            self.root_sons = {i: son for i, son in enumerate(potential_sons)}

        for son in potential_sons:
            heapq.heappush(self.min_heap, son)

    def terminate(self, outcome: str):
        self.terminated = True
        print('Agent ' + str(self.agent_id) + ' has terminated')

        path_cost = self.current_state.current_path_cost
        N = self.expansions
        performance_measures = 'T = 0 : ' + str(path_cost) + \
                               ', T = 0.01 : ' + str(path_cost + N * 0.01) + ', T = 0.000001 : ' \
                               + str(path_cost + N * 0.000001)

        if outcome == 'success':
            print('Verices visited: ' + ' '.join([v_id for v_id in self.current_state.path]) +
                  '\nOptimal path cost: ' + str(path_cost)
                  + '\nAgent Type: ' + self.agent_type +
                  '\nHeuristic: ' + self.heuristic +
                  '\nTotal Expansions: ' + str(self.expansions) +
                  '\nPerformance measures: ' + performance_measures
                  + '\n--------------------------------------------------')

        elif outcome == 'limit_reached':
            print('FAIL: Time limit exceeded')


class AIGreedyAgent(AIAgent):
    def __init__(self, agent_type: str, agent_id, prb: Problem, V0: Vertex,
                 heuristic: str):
        State.get_f_value = lambda self: self.get_heuristic_value(heuristic)
        super().__init__(agent_type, agent_id, prb, V0, heuristic)


class AIAStarAgent(AIAgent):
    def __init__(self, agent_type: str, agent_id, prb: Problem, V0: Vertex,
                 heuristic: str):
        def get_f_val(self):
            h = self.get_heuristic_value(heuristic)
            g = self.current_path_cost
            f = h + g

            self.g_value = g
            self.h_value = h
            # print('curr: ' + self.current_vertex.v_id + ', h: ' + str(h) + ', g: ' + str(g) + ', f:' + str(f))
            return f

        State.get_f_value = get_f_val
        super().__init__(agent_type, agent_id, prb, V0, heuristic)


class AIAStarRealTimeAgent(AIAgent):
    def __init__(self, agent_type: str, agent_id, env: Problem, V0: Vertex,
                 heuristic: str, L: int = 10):
        State.get_f_value = lambda self: self.get_heuristic_value(heuristic) + \
                                         self.current_path_cost
        super().__init__(agent_type, agent_id, env, V0, heuristic)
        self.L = L
        self.root_sons = {}


def run_simulation(config, L = 10):
    N, D, vertices_config, edges_config = parse_config_string(config)
    prb = Problem(D, vertices_config, edges_config)

    n = len(vertices_config)
    j = 1
    for i in range(1, n + 1):
        v = 'V' + str(i)
        print(colored('-' * 60 + ' V0 = V' + str(i) + " " + '-' * 60, 'green'))
        for agent_type in ['Greedy', 'AStar', 'AStarRealTime']:
            heuristic = 'Prim MST'
            # for heuristic in ['Prim MST', 'Kruskal MST']:
            prb.run_AI_agents({
                'A' + str(j): [agent_type, v, L],
            }, heuristic)
            j += 1


if __name__ == '__main__':
    run_simulation(config3)

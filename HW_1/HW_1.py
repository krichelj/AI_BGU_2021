import matplotlib.pyplot as plt
from math import sin, cos, pi
import bisect
import heapq
from typing import List, Dict, Any
import sys
from abc import ABC, abstractmethod
import concurrent.futures

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

                if len(splitted) > 1:
                    v, p = splitted
                else:
                    v = splitted[0]
                vertices_config[v] = int(p)
            else:
                e, v1, v2, w = line[1:].replace('W', '').split(' ')
                edges_config[e] = ('V' + v1, 'V' + v2, int(w))

    return N, D, vertices_config, edges_config


class Vertex:
    def __init__(self, v_id, n_people: int, x: float, y: float):
        self.v_id = v_id
        self.n_people = n_people
        self.visited = False
        self.x = x
        self.y = y
        self.edges = []
        self.dist = sys.maxsize

    def __lt__(self, other):
        return (self.dist, self.v_id) < (other.dist, other.v_id)


class Edge:
    def __init__(self, e_id, V1: Vertex, V2: Vertex, w: int):
        self.e_id = e_id
        self.Vs = {V1, V2}
        self.w = w
        self.blocked = False

        for v in self.Vs:
            bisect.insort(v.edges, self)

    def __lt__(self, other):
        return (self.w, self.e_id) < (other.w, other.e_id)

    def get_other_vertex(self, v):
        return (self.Vs - {v}).pop()


class Problem:
    def __init__(self, D: float, vertices_config: Dict[str, int], edges_config: Dict[str, tuple]):
        self.deadline = D
        self.graph = Graph(vertices_config, edges_config)

    def run_agents(self, agents_config: Dict[Any, List]):
        agents = [(HumanAgent(agent_id, self) if agent_config[0] == 'Human' else
                   (GreedyAgent(agent_id, self, self.graph.vertices[agent_config[1]])
                    if agent_config[0] == 'Greedy' else
                    SaboteurAgent(agent_id, self, self.graph.vertices[agent_config[1]], agent_config[2])))
                  for agent_id, agent_config in agents_config.items()]

        for agent in agents:
            agent.run()

    def run_AI_agents(self, agents_config: Dict[Any, List]):
        agents = [(AIGreedyAgent(agent_config[0], agent_id, self, self.graph.vertices[agent_config[1]])
                   if agent_config[0] == 'Greedy' else
                   AIAStarAgent(agent_config[0], agent_id, self, self.graph.vertices[agent_config[1]]))
                  for agent_id, agent_config in agents_config.items()]

        for agent in agents:
            agent.goal_test()


class Graph:
    def __init__(self, vertices_config: Dict[str, int], edges_config: Dict[str, tuple]):
        n = len(vertices_config)
        self.vertices = {v_id: Vertex(v_id, n_people, cos(2 * pi * i / n), sin(2 * pi * i / n))
                         for i, (v_id, n_people) in enumerate(vertices_config.items())}
        self.edges = {e_id: Edge(e_id, self.vertices[e_tup[0]], self.vertices[e_tup[1]], e_tup[2])
                      for e_id, e_tup in edges_config.items()}

    def plot(self):
        V_x = []
        V_y = []
        V_x_visited = []
        V_y_visited = []

        fig, ax = plt.subplots()

        for v in self.vertices.values():
            if v.visited:
                V_x_visited.append(v.x)
                V_y_visited.append(v.y)
            else:
                V_x.append(v.x)
                V_y.append(v.y)

            ax.annotate(s=v.v_id + ' ' + str(v.n_people), xy=(v.x, v.y))

        ax.scatter(V_x, V_y, color="b", label='shapes', s=200)
        ax.scatter(V_x_visited, V_y_visited, color="r", label='visited', s=200)

        for e in self.edges.values():
            V_x = []
            V_y = []

            for v in e.Vs:
                V_x.append(v.x)
                V_y.append(v.y)

            ax.plot(V_x, V_y, color="b", linewidth=0.3)

        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                       labelleft=False)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


class State:
    def __init__(self, prb: Problem, current_vertex: Vertex,
                 people: Dict[Any, bool], current_path_cost: int,
                 current_path: List[str]):
        self.current_path_cost = current_path_cost
        self.prb = prb
        self.current_vertex = current_vertex
        self.people = people
        self.f_value = self.get_f_value()
        self.path = current_path + [self.current_vertex.v_id]

    def get_heuristic_value(self):
        if not any(self.people.values()):
            return 0

        vertices = self.prb.graph.vertices
        vertices[self.current_vertex.v_id].dist = 0
        Q = list(vertices.values())

        while len(Q):
            heapq.heapify(Q)
            u = heapq.heappop(Q)

            for e in u.edges:
                v = e.get_other_vertex(u)

                if v in Q:
                    alt = u.dist + e.w
                    if alt < v.dist:
                        v.dist = alt

        res = max([vertices[v_id].dist for v_id, b in self.people.items() if b])

        for v in vertices.values():
            v.dist = sys.maxsize

        return res

    def get_successors(self):

        raise_son = lambda e: State(self.prb, e.get_other_vertex(self.current_vertex),
                                    self.people.copy(), self.current_path_cost + e.w,
                                    self.path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            potential_sons = list(executor.map(raise_son, self.current_vertex.edges))

        return potential_sons

    def get_f_value(self):
        pass

    def __lt__(self, other):
        return (self.get_f_value(), self.current_vertex.v_id) < (other.get_f_value(), other.current_vertex.v_id)


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
    expansions = 0

    @abstractmethod
    def __init__(self, agent_type: str, agent_id, prb: Problem, V0: Vertex):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.prb = prb
        init_people = {v.v_id: v != V0 for v in self.prb.graph.vertices.values() if v.n_people > 0}
        self.goal = {v_id: False for v_id in init_people.keys()}
        self.current_state = State(self.prb, V0, init_people, 0, [])
        self.min_heap = [self.current_state]
        self.terminated = False

    def goal_test(self):
        heapq.heapify(self.min_heap)
        self.current_state = heapq.heappop(self.min_heap)

        s = self.current_state
        v = s.current_vertex
        v_id = v.v_id
        people = s.people

        print('Current Vertex: ' + v_id + ', f value: ' + str(s.f_value))
        AIAgent.expansions += 1

        if v_id in people:
            people[v_id] = False

        if people == self.goal:
            self.terminate('success')
        elif self.agent_type == 'AStar' and AIAgent.expansions == AIAgent.LIMIT:
            self.terminate('limit_reached')

        if not self.terminated:
            self.expand()
            self.goal_test()

    def expand(self):
        potential_sons = self.current_state.get_successors()
        # print({son.current_vertex.v_id: son.f_value for son in potential_sons})

        for son in potential_sons:
            heapq.heappush(self.min_heap, son)

    def terminate(self, outcome: str):
        self.terminated = True
        print('Agent ' + str(self.agent_id) + ' has terminated')

        if outcome == 'success':
            print('Verices visited: ' + ' '.join([v_id for v_id in self.current_state.path]) +
                  '\nOptimal path cost: ' + str(self.current_state.current_path_cost)
                  + '\nExpansions: ' + str(AIAgent.expansions))
        elif outcome == 'limit_reached':
            print('FAIL: Time limit exceeded')


class AIGreedyAgent(AIAgent):
    def __init__(self, agent_type: str, agent_id, prb: Problem, V0: Vertex):
        State.get_f_value = lambda self: self.get_heuristic_value()
        super().__init__(agent_type, agent_id, prb, V0)


class AIAStarAgent(AIAgent):
    def __init__(self, agent_type: str, agent_id, prb: Problem, V0: Vertex):
        State.get_f_value = lambda self: self.get_heuristic_value() + self.current_path_cost
        super().__init__(agent_type, agent_id, prb, V0)


class AIAStarRealTimeAgent(AIAgent):
    def __init__(self, agent_type: str, agent_id, env: Problem, V0: Vertex):
        super().__init__(agent_type, agent_id, env, V0)


if __name__ == '__main__':
    N, D, vertices_config, edges_config = parse_config_string(config1)
    prb = Problem(D, vertices_config, edges_config)

    # prb.run_agents({
    #     'A1': ['Saboteur', 'V2', 2],
    #     'A2': ['Greedy', 'V1'],
    #     'A3': ['Human'],
    #     })
    prb.run_AI_agents({
        'A1': ['AStar', 'V5'],
    })
    prb.run_agents({
        'A3': ['Human'],
    })

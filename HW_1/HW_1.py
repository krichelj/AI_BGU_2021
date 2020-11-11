import yaml
import matplotlib.pyplot as plt
from math import sin, cos, pi
import bisect
import heapq
from typing import Dict
import sys

config = '''
General:
  N: 6
  D: 40
Vertices:
  V1: 6
  V2: 0
  V3: 2
  V4: 0
  V5: 7
  V6: 1
Edges:
  E1:
    - V1
    - V2
    - 10
  E2:
    - V2
    - V3
    - 100
  E3:
    - V3
    - V6
    - 1
  E4:
    - V2
    - V4
    - 3
  E5:
    - V4
    - V5
    - 2
  E6:
    - V2
    - V5
    - 1
'''

config_file = 'config.yaml'
with open(config_file, "w") as text_file:
    text_file.write(config)


def parse_config(config_filename):
    with open(config_filename) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    general_config = config['General']
    N = general_config['N']
    D = general_config['D']

    vertices_config = config['Vertices']
    edges_config = config['Edges']

    if N != len(vertices_config):
        raise ValueError('The number of vertices should match their description')

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

    def __eq__(self, other):
        return self.v_id == other.v_id

    def __hash__(self):
        return hash(self.v_id)

    def __lt__(self, other):
        return self.dist < other.dist


class Edge:
    def __init__(self, e_id, V1: Vertex, V2: Vertex, w: int):
        self.e_id = e_id
        self.Vs = {V1, V2}
        self.w = w
        self.blocked = False

        for v in self.Vs:
            bisect.insort(v.edges, self)

    def __lt__(self, other):
        return self.w < other.w if self.w == other.w else self.e_id < other.e_id


class Environment:
    def __init__(self, D: float, vertices_config: dict, edges_config: dict):
        n = len(vertices_config)
        self.vertices = {v_id: Vertex(v_id, n_people, cos(2 * pi * i / n), sin(2 * pi * i / n))
                         for i, (v_id, n_people) in enumerate(vertices_config.items())}
        self.edges = {e_id: Edge(e_id, self.vertices[e_tup[0]], self.vertices[e_tup[1]], e_tup[2])
                      for e_id, e_tup in edges_config.items()}
        self.deadline = D
        self.actions_performed = 0
        self.rescued_people = 0
        self.agents = {}

        # AI addition
        self.people = {v: True for v in self.vertices.values() if v.n_people > 0}
        self.search_tree = SearchTree()
        self.shortest_paths = {}

    def run_agents(self, agents_config: dict):

        self.agents = {agent_id: (HumanAgent(agent_id, self) if agent_config[0] == 'Human' else
                                  (GreedyAgent(agent_id, self, self.vertices[agent_config[1]])
                                   if agent_config[0] == 'Greedy' else
                                   SaboteurAgent(agent_id, self, self.vertices[agent_config[1]], agent_config[2])))
                       for agent_id, agent_config in agents_config.items()}

        for agent in self.agents.values():
            agent.run()

    def run_AI_agents(self, agents_config: dict):

        self.agents = {agent_id: AIGreedyAgent(agent_id, self, self.vertices[agent_config[1]])
                       for agent_id, agent_config in agents_config.items()}

        for agent in self.agents.values():
            agent.goal_test()

    def plot_state(self):
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
    def __init__(self, env: Environment, current_vertex: Vertex, people: Dict[Vertex, bool]):
        self.env = env
        self.current_vertex = current_vertex
        self.people = people
        self.sons = []

    def __eq__(self, other):
        return self.current_vertex == other.current_vertex and self.people == other.people

    def __hash__(self):
        return self.get_hash()

    def get_hash(self):
        return hash((self.current_vertex,
                     ''.join(['1' if x else '0' for x in self.people.values()])))

    def get_heuristic_value(self):
        if self.current_vertex in self.env.shortest_paths:
            res = self.env.shortest_paths[self.current_vertex]
        else:
            if not any(self.people.values()):
                return 0
            vertices = self.env.vertices.copy()
            vertices[self.current_vertex.v_id].dist = 0
            dists = {}
            Q = [(v, v_id) for v_id, v in vertices.items()]
            heapq.heapify(Q)

            while len(Q):
                u, u_id = heapq.heappop(Q)
                dists[u_id] = u.dist
                del vertices[u_id]

                for e in u.edges:
                    v = (e.Vs - {u}).pop()

                    if v.v_id in vertices.keys():
                        alt = u.dist + e.w
                        if alt < v.dist:
                            u.dist = alt

            res = [dists[v.v_id] for v, b in self.people.items() if b]
            print(res)
            self.env.shortest_paths[self.current_vertex] = res

        return max(res)


class SearchTree:
    def __init__(self):
        self.states = {}


class Agent:
    def __init__(self, agent_id, env: Environment, V0: Vertex = None):
        self.agent_id = agent_id
        self.env = env
        self.terminated = False
        self.current_vertex = V0

    def run(self):
        pass

    def traverse(self, new_edge):
        self.current_vertex = (new_edge.Vs - {self.current_vertex}).pop()
        self.env.deadline -= new_edge.w

    def terminate(self):
        self.terminated = True
        print('The agent ' + self.agent_id + ' has terminated')


class HumanAgent(Agent):
    def __init__(self, agent_id, env: Environment):
        super().__init__(agent_id, env)

    def run(self):
        self.env.plot_state()


class GreedyAgent(Agent):
    def __init__(self, agent_id, env: Environment, V0: Vertex):
        super().__init__(agent_id, env, V0)
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

            if minimal_edge and minimal_edge.w <= self.env.deadline:
                self.traverse(minimal_edge)
            else:
                self.terminate()

    def traverse(self, new_edge):
        super().traverse(new_edge)
        self.saved_people += self.current_vertex.n_people
        self.current_vertex.n_people = 0
        self.current_vertex.visited = True


class SaboteurAgent(Agent):
    def __init__(self, agent_id, env: Environment, V0: Vertex, V: int):
        super().__init__(agent_id, env, V0)
        self.current_vertex = V0
        self.V = V

    def run(self):
        self.env.deadline -= self.V
        while not self.terminated:
            minimal_edge = None
            if len(self.current_vertex.edges):
                self.current_vertex.edges[0].blocked = True
                self.env.deadline -= 1

            for e in self.current_vertex.edges[1:]:
                if not e.blocked:
                    minimal_edge = e
                    break

            if minimal_edge and minimal_edge.w <= self.env.deadline:
                self.traverse(minimal_edge)

            else:
                self.terminate()

    def traverse(self, new_edge):
        super().traverse(new_edge)


class AIAgent:
    def __init__(self, agent_id, env: Environment, V0: Vertex):
        self.agent_id = agent_id
        self.env = env
        init_people = {v: v != V0 for v in self.env.vertices.values() if v.n_people > 0}
        self.current_state = State(env, V0, init_people)
        self.env.search_tree.states[self.current_state.get_hash()] = self.current_state
        self.path = []
        self.min_heap = []

    def goal_test(self):
        minimal_state = heapq.heappop(self.min_heap)
        if minimal_state.current_vertex in minimal_state.people:
            minimal_state.people[minimal_state.current_vertex] = False

        self.path.append(minimal_state)
        self.current_state = minimal_state
        people = list(minimal_state.people.values())
        goal = [False] * len(people)

        print(people)

        if people == goal:
            print('Agent ' + str(self.agent_id) + ' has terminated')
            print('Verices visited: ' + ' '.join([s.current_vertex.v_id for s in self.path]))
            return

        self.expand()
        self.goal_test()

    def expand(self):
        potential_sons = []

        for e in self.current_state.current_vertex.edges:
            v = (e.Vs - {self.current_state.current_vertex}).pop()
            son = State(self.current_state.env, v, self.current_state.people.copy())

            if self.current_state.current_vertex in son.people:
                son.people[self.current_state.current_vertex] = False
            potential_sons.append(son)

        for son in potential_sons:
            # son_hash = son.get_hash()
            # if son_hash in self.env.search_tree.states:
            #     son = self.env.search_tree.states[son_hash]
            # else:
            #   self.env.search_tree.states[son_hash] = son
            self.current_state.sons.append(son)

        for son in self.current_state.sons:
            heapq.heappush(self.min_heap, son)

    def get_path(self):
        return self.path


class AIGreedyAgent(AIAgent):
    def __init__(self, agent_id, env: Environment, V0: Vertex):
        super().__init__(agent_id, env, V0)
        State.__lt__ = lambda myself, other: myself.get_heuristic_value() < other.get_heuristic_value()
        self.min_heap = [self.current_state]
        heapq.heapify(self.min_heap)


# class AIAStarAgent(AIAgent):

# class AIAStarRealTimeAgent(AIAgent):

if __name__ == '__main__':
    N, D, vertices_config, edges_config = parse_config(config_file)
    env = Environment(D, vertices_config, edges_config)

    # env.run_agents({
    #     'A1': ['Saboteur', 'V2', 2],
    #     'A2': ['Greedy', 'V1'], 
    #     'A3': ['Human'],
    #     })
    env.run_AI_agents({
        'A1': ['AI', 'V1', 2],
    })
    # heuristic_evaluation_function(env)

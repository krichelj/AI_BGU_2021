import yaml
from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt


class Vertex:
    def __init__(self, v_id, n_people: int, x: float, y: float):
        self.v_id = v_id
        self.n_people = n_people
        self.x = x
        self.y = y
        self.edges = set()


class Edge:
    def __init__(self, e_id, V1: Vertex, V2: Vertex, w: int):
        self.e_id = e_id
        self.Vs = {V1, V2}
        self.w = w
        self.blocked = False

        for v in self.Vs:
            v.edges.add(self)

    def __lt__(self, other):
        return self.w < other.w


class Simulation:
    def __init__(self, D: float, vertices_config: dict, edges_config: dict):
        self.vertices = {v_id: Vertex(v_id, n_people, np.random.random(), np.random.random())
                         for v_id, n_people in vertices_config.items()}
        self.edges = {e_id: Edge(e_id, self.vertices[e_tup[0]], self.vertices[e_tup[1]], e_tup[2])
                      for e_id, e_tup in edges_config.items()}
        self.deadline = D
        self.actions_performed = 0
        self.rescued_people = 0

    def run_agents(self, program, initial_positions: List[Any]):
        if any([pos not in self.vertices.keys() for pos in initial_positions]):
            raise ValueError('Invalid initial positions')

        agents = [HumanAgent(self, self.vertices[V0]) if program == 'Human' else
                  (GreedyAgent(self, self.vertices[V0]) if program == 'Greedy' else
                   SaboteurAgent(self, self.vertices[V0])) for V0 in initial_positions]

        for agent in agents:
            agent.run()

    def plot_state(self):
        V_x = []
        V_y = []

        for v in self.vertices.values():
            V_x.append(v.x)
            V_y.append(v.y)
            plt.annotate(text=v.n_people, xy=(v.x, v.y))

        plt.scatter(V_x, V_y, color="r", label='shapes', s=200)

        for e in self.edges.values():
            V_x = []
            V_y = []

            for v in e.Vs:
                V_x.append(v.x)
                V_y.append(v.y)

            plt.plot(V_x, V_y, color="b", linewidth=0.3)

        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                        labelleft=False)
        plt.show()


class Agent:
    def __init__(self, sim_state, V0: Vertex):
        self.sim_state = sim_state
        V0.n_people = 0
        self.current_vertex = V0

    def run(self):
        pass

    def traverse(self, new_edge):
        pass

    @staticmethod
    def terminate():
        print('The simulation has ended')


class HumanAgent(Agent):
    def __init__(self, sim_state: Simulation, V0: Vertex):
        super().__init__(sim_state, V0)

    def run(self):
        self.sim_state.plot_state()


class GreedyAgent(Agent):
    def __init__(self, sim_state, V0: Vertex):
        super().__init__(sim_state, V0)

    def run(self):
        # print([v.n_people for e in self.current_vertex.edges for v in e.Vs])
        relevant_edges = [e for e in self.current_vertex.edges
                          if not e.blocked and any([v.n_people > 0 for v in e.Vs])]
        minimal_edge = min(relevant_edges) if len(relevant_edges) else 0
        if minimal_edge != 0 and minimal_edge.w < self.sim_state.deadline:
            self.traverse(minimal_edge)
        else:
            Agent.terminate()

    def traverse(self, new_edge):
        self.current_vertex = (new_edge.Vs - {self.current_vertex}).pop()
        self.current_vertex.n_people = 0
        self.sim_state.deadline -= new_edge.w


class SaboteurAgent(Agent):
    def __init__(self, sim_state, V0: Vertex):
        super().__init__(sim_state, V0)


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


if __name__ == '__main__':
    N, D, vertices_config, edges_config = parse_config(r'config.yaml')
    sim = Simulation(D, vertices_config, edges_config)
    sim.run_agents('Greedy', ['V1'])

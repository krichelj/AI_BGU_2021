import yaml
import matplotlib.pyplot as plt
from math import sin, cos, pi
import bisect


class Vertex:
    def __init__(self, v_id, n_people: int, x: float, y: float):
        self.v_id = v_id
        self.n_people = n_people
        self.visited = False
        self.x = x
        self.y = y
        self.edges = []


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
        self.agents = []

    def run_agents(self, agents_config: dict):
        self.agents = {agent_id: (HumanAgent(agent_id, self) if agent_config[0] == 'Human' else
                                  (GreedyAgent(agent_id, self, self.vertices[agent_config[1]])
                                   if agent_config[0] == 'Greedy' else
                                   SaboteurAgent(agent_id, self, self.vertices[agent_config[1]])))
                       for agent_id, agent_config in agents_config.items()}

        for agent in self.agents.values():
            agent.run()

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

            ax.annotate(text=v.v_id + ' ' + str(v.n_people), xy=(v.x, v.y))

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


class Agent:
    def __init__(self, agent_id, env_state: Environment, V0: Vertex = None):
        self.agent_id = agent_id
        self.env_state = env_state
        self.terminated = False
        self.current_vertex = V0

    def run(self):
        pass

    def traverse(self, new_edge):
        self.current_vertex = (new_edge.Vs - {self.current_vertex}).pop()
        self.env_state.deadline -= new_edge.w

    def terminate(self):
        self.terminated = True
        print('The agent ' + self.agent_id + ' has terminated')


class HumanAgent(Agent):
    def __init__(self, agent_id, env_state: Environment):
        super().__init__(agent_id, env_state)

    def run(self):
        self.env_state.plot_state()


class GreedyAgent(Agent):
    def __init__(self, agent_id, env_state: Environment, V0: Vertex):
        super().__init__(agent_id, env_state, V0)
        self.saved_people = V0.n_people
        V0.n_people = 0
        self.current_vertex = V0
        self.current_vertex.visited = True

    def run(self):
        while not self.terminated:
            minimal_edge = None
            for e in self.current_vertex.edges:
                if not e.blocked and any([v.n_people > 0 for v in e.Vs]):
                    minimal_edge = e
                    break

            if minimal_edge and minimal_edge.w <= self.env_state.deadline:
                self.traverse(minimal_edge)
            else:
                self.terminate()

    def traverse(self, new_edge):
        super().traverse(new_edge)
        self.saved_people += self.current_vertex.n_people
        self.current_vertex.n_people = 0
        self.current_vertex.visited = True


class SaboteurAgent(Agent):
    def __init__(self, agent_id, env_state: Environment, V0: Vertex):
        super().__init__(agent_id, env_state, V0)
        self.current_vertex = V0

    def run(self):
        while not self.terminated:
            minimal_edge = None
            if len(self.current_vertex.edges):
                self.current_vertex.edges[0].blocked = True
                self.env_state.deadline -= 1

            for e in self.current_vertex.edges[1:]:
                if not e.blocked:
                    minimal_edge = e
                    break
            if minimal_edge and minimal_edge.w <= self.env_state.deadline:
                self.traverse(minimal_edge)
            else:
                self.terminate()

    def traverse(self, new_edge):
        super().traverse(new_edge)


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
    env = Environment(D, vertices_config, edges_config)
    env.run_agents({'A1': ['Saboteur', 'V2'], 'A2': ['Greedy', 'V1'],
                    'A3': ['Human']})

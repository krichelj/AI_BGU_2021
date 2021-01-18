import sys
import heapq
from typing import Dict, Union, List
from math import sin, cos, pi
import matplotlib.pyplot as plt

MAX_SIZE = sys.maxsize


class Vertex:
    def __init__(self, v_id, p_people: Union[int, float] = None, x: float = None, y: float = None):
        self.v_id = v_id
        self.p_people = p_people
        self.x = x
        self.y = y
        self.edges = {}
        self.dist = MAX_SIZE
        self.init = ''

    def __lt__(self, other):
        return (self.dist, self.v_id) < (other.dist, other.v_id)

    def __str__(self):
        return self.v_id


class Edge:
    def __init__(self, e_id, V1: Vertex, V2: Vertex, w: int = 1, prob: float = 0):
        self.e_id = e_id
        self.Vs = {V1, V2}
        self.w = w
        self.blocked = False
        self.prob = prob

    def __str__(self):
        return self.e_id

    def __eq__(self, other):
        return self.e_id == other.e_id

    def __hash__(self):
        return hash(self.e_id)

    def __lt__(self, other):
        return (self.w, self.e_id) < (other.w, other.e_id)

    def get_other_vertex(self, v: Vertex) -> Vertex:
        return (self.Vs - {v}).pop()

    def get_vertices_ids(self):
        v_id = self.e_id[0:2]
        u_id = self.e_id[2:4]

        return v_id, u_id

    def get_id_as_vertices(self) -> str:
        v_id, u_id = self.get_vertices_ids()

        return v_id + u_id


class DirectedEdge(Edge):
    def __init__(self, e_id, V1: Vertex, V2: Vertex, origin_id):
        self.origin_id = origin_id
        super().__init__(e_id, V1, V2)


class EdgeLocation:
    def __init__(self, e_id, origin_v_id, destination_v_id, units: int):
        self.e_id = e_id
        self.origin_v_id = origin_v_id
        self.destination_v_id = destination_v_id
        self.units = units

    def __str__(self):
        return 'Walking on ' + self.e_id + ' to ' + self.destination_v_id \
               + ' with ' + str(self.units) + ' units left'


class Graph:
    def __init__(self, vertices: Dict[str, Vertex], edges: Dict[str, Edge]):
        self._vertices = vertices
        self._edges = edges

    def get_vertices(self) -> Dict[str, Vertex]:
        return self._vertices

    def get_edges(self) -> Dict[str, Edge]:
        return self._edges

    def get_weight(self, v1_id: str, v2_id: str) -> int:
        e_id = v1_id + v2_id
        if e_id not in self._edges:
            e_id = v2_id + v1_id

        return self._edges[e_id].w

    @staticmethod
    def add_edges_to_vertices(edges: Dict[str, Edge]):
        for e_id, e in edges.items():
            for v in e.Vs:
                v_edges = v.edges
                if e_id not in v_edges:
                    v_edges[e_id] = e

    @classmethod
    def from_config(cls, vertices_config: Dict[str, float], edges_config: Dict[str, tuple]):
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
            if v.p_people and v.p_people > 0:
                V_x_people.append(v.x)
                V_y_people.append(v.y)
            else:
                V_x.append(v.x)
                V_y.append(v.y)

            ax.annotate(str(v.v_id) + (', p:' + str(v.p_people) if v.p_people and v.p_people > 0 else '') +
                        (':' + str(v.init) if v.init else ''), xy=(v.x, v.y))

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

    def Dijkstra(self, v_id: str):
        vertices = self._vertices

        for v in vertices.values():
            v.dist = MAX_SIZE

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

    def Prim(self, init_id: str) -> int:
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

    def is_path(self, path_edges: List[str]) -> bool:
        result = True
        edges = self.get_edges()

        for i in range(len(path_edges) - 1):

            e_id_1 = path_edges[i]
            e_id_2 = path_edges[i + 1]

            e_ids = [e_id_1, e_id_2]

            for j in range(len(e_ids)):
                if e_ids[j] not in edges:
                    e_ids[j] = e_ids[j][2:4] + e_ids[j][0:2]
                    if e_ids[j] not in edges:
                        return False

            result = result and e_id_1.split('V')[2] == e_id_2.split('V')[1]

        return result

    def get_all_paths(self, origin_id: str, target_id: str) -> List[List[str]]:

        connection_path = []
        paths = []

        def find_paths(origin_id: str, target_id: str):

            origin = self._vertices[origin_id]

            for e_id, e in origin.edges.items():
                next_vertex = e.get_other_vertex(origin)
                next_id = next_vertex.v_id

                if next_id == target_id:
                    current_path = connection_path.copy()
                    current_path.append(e_id)
                    paths.append(current_path)
                elif e_id not in connection_path:
                    connection_path.append(e_id)
                    find_paths(next_id, target_id)
                    connection_path.pop()

        find_paths(origin_id, target_id)

        return paths

import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast
# networkx 是一个用于创建、操作和研究 复杂网络（图）的 Python 库。它支持 有向图、无向图、多重图 等多种图结构，并提供了丰富的算法，例如 图遍历、最短路径、社区检测、图匹配、中心性计算 等。
import networkx as nx
import numpy as np

from .._utils import logger
from ..base import (
    BaseGraphStorage,
    SingleCommunitySchema,
)
from ..prompt import GRAPH_FIELD_SEP


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    # 静态方法 load_nx_graph：如果 file_name 存在，则加载 GraphML 格式的图数据，否则返回 None。
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    # 静态方法 write_nx_graph：将 networkx 图数据保存为 GraphML 格式，并记录图的节点数和边数。
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    # 该方法用于提取图的最大连通子图，并确保节点和边的顺序稳定。
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component
# 
        graph = graph.copy()# 拷贝一份图，以避免修改原始图
        graph = cast(nx.Graph, largest_connected_component(graph))# 获取最大连通子图
        # 规范化节点名称（去除HTML转义字符并转换为大写），然后调用 _stabilize_graph 进行标准化处理。
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore # 统一格式
        graph = nx.relabel_nodes(graph, node_mapping) # 重新命名节点
        return NetworkXStorage._stabilize_graph(graph)# 确保图结构稳定

    @staticmethod
    # 该方法确保无向图的读取顺序始终一致，即使存储和读取发生变化。
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        # 如果原始图是有向图，则创建 DiGraph，否则创建无向 Graph。
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
        
        sorted_nodes = graph.nodes(data=True)
        # 以节点 ID 进行排序，并添加到新的 fixed_graph 中。
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])
        fixed_graph.add_nodes_from(sorted_nodes)
        # 以固定顺序存储边，确保无向边 (A, B) 和 (B, A) 一致
        edges = list(graph.edges(data=True))#(source, target, edge_data)

        if not graph.is_directed():

            def _sort_source_target(edge):
                """对无向边进行排序，确保存储顺序一致。"""
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            """生成唯一的边标识，确保排序稳定。"""
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        """初始化类时，加载已存储的图，如果不存在，则创建一个空图。"""
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,# 定义聚类算法
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,# 定义节点嵌入算法
        }

    async def index_done_callback(self):
        """索引完成后，将当前图存储到 GraphML 文件中。"""
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        """检查图中是否存在指定的节点。"""
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """检查图中是否存在指定的边。"""
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """获取指定节点的数据字典。"""
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        """返回节点的度数，如果节点不存在，则返回 0。"""
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """计算两个节点的度数之和。"""
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """获取指定边的数据字典。"""
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        """获取与指定节点相连的所有边。"""
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """插入或更新节点数据。"""
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """插入或更新边数据。"""
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def clustering(self, algorithm: str):
        """执行指定的聚类算法。"""
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """计算社区结构，并返回社区信息。"""
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                results[cluster_key]["chunk_ids"].update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        """将聚类结果存储到图的节点属性中。"""
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        """执行 Leiden 聚类算法，并更新图的社区结构。"""
        from graspologic.partition import hierarchical_leiden

        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        """执行指定的节点嵌入算法。"""
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        """执行 Node2Vec 节点嵌入。"""
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

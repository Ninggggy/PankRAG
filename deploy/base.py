from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar

import numpy as np

from ._utils import EmbeddingFunc


@dataclass
class QueryParam:
    mode: Literal["local", "global", "naive"] ="global"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    # naive search
    naive_max_token_for_text_unit = 12000
    # local search
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33
    local_max_token_for_local_context: int = 4800  # 12000 * 0.4
    local_max_token_for_community_report: int = 3200  # 12000 * 0.27
    local_community_single_one: bool = False
    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

SingleCommunitySchema = TypedDict(
    "SingleCommunitySchema",
    {
        "level": int,
        "title": str,
        "edges": list[list[str, str]],
        "nodes": list[str],
        "chunk_ids": list[str],
        "occurrence": float,
        "sub_communities": list[str],
    },
)


class CommunitySchema(SingleCommunitySchema):
    report_string: str
    report_json: dict


T = TypeVar("T")


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_start_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass  # 使用 dataclass 自动生成 __init__ 等方法
class BaseKVStorage(Generic[T], StorageNameSpace):
    """
    通用键值存储抽象基类，支持异步操作，定义存储的基础接口。
    该类必须由子类实现具体存储逻辑（如数据库、缓存、文件存储）。
    """

    async def all_keys(self) -> list[str]:
        """
        获取存储中的所有键（keys）。
        返回:
            list[str]: 存储中的所有键的列表。
        抛出:
            NotImplementedError: 该方法需要在子类中实现。
        """
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        """
        根据唯一 ID 获取存储的值。
        
        参数:
            id (str): 唯一标识符。
        
        返回:
            Union[T, None]: 如果 ID 存在，则返回存储的值，否则返回 None。
        """
        raise NotImplementedError


    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        """
        批量获取多个 ID 对应的存储值。
        
        参数:
            ids (list[str]): 需要查询的 ID 列表。
            fields (Union[set[str], None]): 可选，指定查询的字段（如果存储支持）。
        
        返回:
            list[Union[T, None]]: 返回 ID 对应的数据列表，若 ID 不存在，则返回 None。
        """
        raise NotImplementedError


    async def filter_keys(self, data: list[str]) -> set[str]:
        """
        过滤出存储中 **不存在** 的键（keys）。
        
        参数:
            data (list[str]): 需要检查的键列表。
        
        返回:
            set[str]: 返回存储中 **未找到** 的键集合。
        """
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        """
        插入或更新存储中的数据（存在则更新，不存在则插入）。
        
        参数:
            data (dict[str, T]): 需要存储的数据，键为 ID，值为存储的对象。
        """
        raise NotImplementedError


    async def drop(self):
        """
        清空存储中的所有数据。
        """
        raise NotImplementedError



@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Return the community representation with report and nodes"""
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in nano-graphrag.")

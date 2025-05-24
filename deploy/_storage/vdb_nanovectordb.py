import asyncio
import os
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB

from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2# 设置默认的相似度阈值（余弦相似度）

    def __post_init__(self):
        # 初始化数据库存储路径、参数和 NanoVectorDB 客户端
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        # 从 global_config 获取最大批处理大小（防止计算资源溢出）
        self._max_batch_size = self.global_config["embedding_batch_num"]
        # 初始化 NanoVectorDB 客户端：设定 嵌入向量维度 (embedding_dim)。设定 存储文件路径 (storage_file)
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        # 从 global_config 读取查询相似度阈值，如果不存在则使用默认值 0.2。
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )
# 向数据库插入或更新向量
    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        # 提取 meta_fields（元数据字段） 并 添加 __id__，用于后续存储。
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        # 提取所有文本 content，用于计算 向量嵌入。
        contents = [v["content"] for v in data.values()]
        # 将 contents 划分为多个批次，便于 分批计算向量嵌入
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        # 异步并行计算多个文本的向量嵌入
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        # 合并所有批次的向量，得到完整的 numpy 向量矩阵。
        embeddings = np.concatenate(embeddings_list)
        # 把计算出的向量嵌入 存入 list_data
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        # 把计算出的向量嵌入 存入 list_data
        results = self._client.upsert(datas=list_data)
        return results
# 查询最相似的向量
    async def query(self, query: str, top_k=5):
        # 将 query 进行向量化，用于检索数据库中的相似向量。
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        # 通过 NanoVectorDB.query() 执行向量检索：top_k: 只返回 最相似的 k 个 向量。cosine_better_than_threshold: 只返回相似度 高于阈值 的向量。
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        # 格式化查询结果
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()

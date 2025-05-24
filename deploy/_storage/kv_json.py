import os
from dataclasses import dataclass

from .._utils import load_json, logger, write_json
from ..base import (
    BaseKVStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        """初始化 JSON 存储，加载已有数据"""
        working_dir = self.global_config["working_dir"] # 获取工作目录
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")# 生成存储文件路径
        self._data = load_json(self._file_name) or {}# 加载 JSON 数据，若文件不存在则初始化为空字典
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")# 记录日志，输出已加载的键值对数量

    async def all_keys(self) -> list[str]:
        """返回所有存储的键"""
        return list(self._data.keys())

    async def index_done_callback(self):
        """在索引完成后，将数据写入 JSON 文件"""
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        """根据 ID 获取存储的数据，若不存在返回 None"""
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        """根据多个 ID 获取数据，可选返回特定字段"""
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        """过滤出 data 列表中未存储的键"""
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        """批量插入或更新数据"""
        self._data.update(data)

    async def drop(self):
        """清空存储数据"""
        self._data = {}

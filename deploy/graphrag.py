import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
import re
import tiktoken

import json
from dotenv import load_dotenv
# 从同级文件夹种加载函数用于调用
from ._llm import (
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)
from ._op import (
    chunking_by_token_size,
    extract_entities,
    generate_community_report,
    get_chunks,
    local_query,
    local_query_with_data,
    global_query,
    global_query_with_data,
    naive_query,
    query_dag_decompose,
    gather_zero_answer,
    local_query_with_data_content,
    global_query_with_data_content,
    local_query_content,
    global_query_content
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

import re

def extract_json(raw_str):
    """
    从含有多余字符的字符串中提取最外层{}包裹的JSON
    
    参数：
    raw_str (str): 可能包含其他字符的原始字符串
    
    返回：
    str: 纯净的JSON字符串
    
    示例：
    >>> extract_json('```json\n{"intents_number":1,...}\n```')
    '{"intents_number":1,"intents":[{"id":0,"intent":"content summary","if_abstract":true}]}'
    """

    processed = raw_str.replace('\n', '').strip()
    
    # 方案一：正则匹配（优先使用）
    json_match = re.search(r'\{.*\}', processed, re.DOTALL)
    if json_match:
        return json_match.group()
    
    # 方案二：字符定位（备用方案）
    start = processed.find('{')
    end = processed.rfind('}')
    if start != -1 and end != -1 and start < end:
        return processed[start:end+1]
    

    return '{}'


import os
import json
import time
from pathlib import Path

class CheckpointManager:
    def __init__(self, working_dir: str):
        self.checkpoint_file = Path(working_dir) / "insert_checkpoint.json"
    
    def load_checkpoint(self) -> dict:
        """返回包含doc_index和pending_chunks的字典，无断点时返回空字典"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_checkpoint(self, data: dict):
        """保存包含文档位置和待处理chunks的字典"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"保存断点失败: {str(e)}")

    def clear(self):
        """清除断点文件"""
        try:
            if self.checkpoint_file.exists():
                os.remove(self.checkpoint_file)
        except Exception as e:
            print(f"清除断点失败: {str(e)}")

@dataclass
class GraphRAG:

    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    enable_local: bool = True
    enable_naive_rag: bool = False
    

    chunk_func: Callable[
        [
            list[list[int]],# 分词后的 token 列表
            List[str],# 文档键名
            tiktoken.Encoding,# tiktoken 编码模型
            Optional[int], # 可选的重叠 token 大小
            Optional[int],# 可选的最大 token 大小
        ],
        List[Dict[str, Union[str, int]]],# 返回的分块结果
    ] = chunking_by_token_size

    chunk_token_size: int = 1200 # 每个文本块的最大 token 数
    chunk_overlap_token_size: int = 100 # 相邻块之间的重叠 token 数
    tiktoken_model_name: str = "gpt-4o" # 使用的 tiktoken 模型名称

    # entity extraction
    entity_extract_max_gleaning: int = 1 # 进行实体抽取的最大轮数
    entity_summary_to_max_tokens: int = 500 # 实体摘要的最大 token 数

    # graph clustering
    graph_cluster_algorithm: str = "leiden" # 选择的聚类算法（Leiden 算法）
    max_graph_cluster_size: int = 10 # 每个聚类的最大节点数
    graph_cluster_seed: int = 0xDEADBEEF # 随机种子，确保聚类结果可复现

    # node embedding
    node_embedding_algorithm: str = "node2vec" # 选择的节点嵌入算法（node2vec）
    node2vec_params: dict = field(
        default_factory=lambda: {  # 使用默认参数初始化字典
            "dimensions": 1536, # 生成的节点嵌入向量维度
            "num_walks": 10, # 每个节点执行的随机游走次数
            "walk_length": 40,  # 每次随机游走的步长
            "num_walks": 10, # 每次随机游走的分支数
            "window_size": 2,  # Skip-gram 训练时的窗口大小
            "iterations": 3, # 训练迭代次数
            "random_seed": 3, # 随机种子，确保结果可复现
        }
    )

    # community reports 特殊社区（special community）报告的 LLM 相关参数
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}} # 指定 LLM 输出格式为 JSON 对象
    )

    # 文本嵌入（text embedding）相关参数
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)# 使用 OpenAI 的嵌入函数进行文本向量化
    embedding_batch_num: int = 32 # 每批次嵌入处理的文本数量
    embedding_func_max_async: int = 16 # 最大异步并发处理的嵌入请求数
    query_better_than_threshold: float = 0.2 # 查询匹配的最小阈值，确保检索结果质量

    # 大语言模型（LLM）相关参数
    using_azure_openai: bool = False # 是否使用 Azure OpenAI 服务（默认为 False）
    using_amazon_bedrock: bool = False # 是否使用 Amazon Bedrock 服务（默认为 False）
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0" # 最高性能的 LLM 模型 ID（Claude 3 Sonnet）
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0" # 经济型 LLM 模型 ID（Claude 3 Haiku）
    best_model_func: callable = gpt_4o_complete# 最高性能模型的调用函数
    best_model_max_token_size: int = 32768# 最高性能模型的最大 token 处理能力
    best_model_max_async: int = 16# 最高性能模型的最大并发请求数
    cheap_model_func: callable = gpt_4o_mini_complete# 经济型模型的调用函数
    cheap_model_max_token_size: int = 32768# 经济型模型的最大 token 处理能力
    cheap_model_max_async: int = 16# 经济型模型的最大并发请求数

    # entity extraction
    entity_extraction_func: callable = extract_entities # 实体抽取的核心函数

    # 存储（storage）相关参数
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage # 键-值对存储的实现类，使用 JSON 作为存储格式
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage # 向量数据库的存储实现，使用 NanoVectorDB 进行向量存储
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict) # 向量数据库存储的额外参数，默认为空字典
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage # 图数据存储的实现，使用 NetworkX 作为存储后端
    enable_llm_cache: bool = True # 是否启用 LLM 缓存，提高重复查询的响应速度

    # extension
    always_create_working_dir: bool = True # 是否始终创建工作目录，确保运行环境稳定
    addon_params: dict = field(default_factory=dict) # 额外的附加参数，默认为空字典
    convert_response_to_json_func: callable = convert_response_to_json # 将 LLM 响应转换为 JSON 格式的函数
    final:bool=False
    def __post_init__(self):
        # 记录 GraphRAG 初始化时的所有参数信息，方便调试
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")
         # 如果启用了 Azure OpenAI，则将默认的 OpenAI 相关函数替换为 Azure 版本
        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )
         # 如果启用了 Amazon Bedrock，则将默认的 OpenAI 相关函数替换为 Amazon Bedrock 版本
        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(self.best_model_id)
            self.cheap_model_func = create_amazon_bedrock_complete_function(self.cheap_model_id)
            self.embedding_func = amazon_bedrock_embedding
            logger.info(
                "Switched the default openai funcs to Amazon Bedrock"
            )
        # 如果工作目录不存在且允许创建，则创建工作目录
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        # 初始化 JSON 存储：用于存储完整文档
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )
        # 初始化 JSON 存储：用于存储文本分块
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )
        # 如果启用了 LLM 缓存，则初始化 JSON 存储：用于存储 LLM 的响应缓存
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )
        # 初始化 JSON 存储：用于存储社区报告数据
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        # 初始化图数据库存储：用于存储文本块的实体关系图
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )
        # 限制嵌入函数的最大异步调用数，以优化并发执行
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        # 初始化向量数据库：用于存储实体的向量嵌入，仅在启用本地存储时生效
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
         # 初始化向量数据库：用于存储文本块的向量嵌入，仅在启用 Naïve RAG 时生效
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )
        # 限制 LLM 模型的最大异步调用数，并绑定 LLM 响应缓存，以减少重复计算
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )

        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )
    
        self.processed_hops = set()  # 记录已处理的hop_id
        self.results = {}  # 存储处理结果

    async def process_hop(self, hop):
        param= QueryParam()
        # 检查依赖是否满足
        for dep in hop.get('dependencies', []):
            if dep not in self.processed_hops:
                await self.wait_for_dependency(dep)
            
        sub_query = hop['sub_query']
        # 收集依赖项的结果
        subq_data = [self.results[dep] for dep in hop.get('dependencies', [])]
        # 模拟处理逻辑（替换为实际处理代码）
        print(f"Processing {hop['hop_id']}: {hop['sub_query']}")
        param.mode ='local'
        answer = await self.aquery_with_data(sub_query, self.final,subq_data, param)
        
        # 记录处理完成的hop
        self.processed_hops.add(hop['hop_id'])
        self.results[hop['hop_id']] = answer
        print(f"Processed {hop['hop_id']} with answer: {answer}")

    async def wait_for_dependency(self, dep_id):
        while dep_id not in self.processed_hops:
            await asyncio.sleep(0.01)

    async def process_sequence(self, sequence):
        if sequence['sequence_id'] == "0":
            return
        
        # 修改点：顺序处理每个hop（原并行逻辑改为顺序）
        for hop in sequence['hop']:  # 直接遍历hop列表
            await self.process_hop(hop)  # 串行执行

    async def process_zero_sequence(self, sequences):
        zero_seq = next((s for s in sequences if s['sequence_id'] == "0"), None)
        if zero_seq:
            # 修改点：顺序处理0号序列的hop
            for hop in zero_seq['hop']:
                await self.process_hop(hop)

    async def process_final_query_noseq0(self, final_queries,zero_sequence):
        param= QueryParam()
        for fq in final_queries:
            # 检查最终依赖
            for dep in fq['dependencies']:
                if dep not in self.processed_hops:
                    await self.wait_for_dependency(dep)
            data = [self.results[dep] for dep in fq.get('dependencies', [])]

            param.mode ='local'
            self.final=True
            answer,context= await self.aquery_with_data(fq['final_query'], self.final,data, param)
            self.final=False

            # answer = await gather_zero_answer(fq['final_query'], data, param, global_config=asdict(self))


            return answer,context
    async def process_final_query(self, final_queries):
        for fq in final_queries:
            # 检查最终依赖
            for dep in fq['dependencies']:
                if dep not in self.processed_hops:
                    await self.wait_for_dependency(dep)
            
            print(f"\nFinalizing: {fq['final_query']}")
            # 组合最终结果
            answer = "\n".join([
                self.results[dep]
                for dep in fq['dependencies']
            ])
            print(f"Final Answer:\n{answer}")
            return answer

    async def execute(self, data):
        # Step 1: 处理非0序列（并行）
        main_sequences = [
            seq for seq in data['path']['sequences']
            if seq['sequence_id'] != "0"
        ]
        zero_sequence = [
            seq for seq in data['path']['sequences']
            if seq['sequence_id'] == "0"
        ]
        await asyncio.gather(*[
            self.process_sequence(seq) for seq in main_sequences
        ])

        # Step 2: 处理0序列
        await self.process_zero_sequence(data['path']['sequences'])

        # Step 3: 处理最终查询
        # response=await self.process_final_query_noseq0(data['path']['final_query'],zero_sequence)
        response,context=await self.process_final_query_noseq0(data['path']['final_query'],zero_sequence)
        return response,context        


    def insert(self, string_or_strings):
        # 获取当前或创建新的事件循环
        loop = always_get_an_event_loop()
        # 在事件循环中运行异步查询操作
        return loop.run_until_complete(self.ainsert(string_or_strings))



 

    async def  query(self, query: str, param: QueryParam = QueryParam()):

        data2=await query_dag_decompose(query,global_config=asdict(self))
        print(f"jsonloads前: {data2}")
        data=convert_response_to_json(data2)
        # if data["path"] not in None:
        print(f"分解路径 {data}")
        # processor = RealTimePathProcessor()
        # result = await processor.process(data)
        # 访问主查询
        if data["path"] is not None:
            # answer=await self.execute(data)
            answer,context=await self.execute(data)
        else:
            self.final=True
            param.mode = 'local'
            answer,context=self.aquery(query,self.final,param)
            # answer=self.aquery(query,self.final,param)
            self.final=False
        
        return answer,data2,context


    async def aquery_with_data(self, query: str, final, data:dict,param: QueryParam = QueryParam()):
        print(f"模式: {param.mode}")
        # 若查询模式为 "local"，但 enable_local 未启用，则抛出错误
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        # 若查询模式为 "naive"，但 enable_naive_rag 未启用，则抛出错误
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if final:
            # 处理 "local" 模式的查询
            if param.mode == "local":
                response, context = await local_query_with_data_content(
                    query,# 查询语句
                    data,
                    self.chunk_entity_relation_graph, # 文档间的实体关系图
                    self.entities_vdb, # 本地实体数据库
                    self.community_reports,  # 社区报告存储
                    self.text_chunks, # 文本块存储
                    param, # 查询参数
                    asdict(self),  # 当前实例的配置信息
                )
            # 处理 "global" 模式的查询
            elif param.mode == "global":
                response, context = await global_query_with_data_content(
                    query,
                    data,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.community_reports,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
            # 处理 "naive" 模式的查询
            elif param.mode == "naive":
                response,context = await naive_query(
                    query,
                    self.chunks_vdb, # 朴素 RAG 查询使用的文本向量数据库
                    self.text_chunks,
                    param,
                    asdict(self),
                )
                # 若查询模式未知，则抛出错误
            else:
                raise ValueError(f"Unknown mode {param.mode}")
                        # 查询完成后执行收尾操作（可能是缓存管理、日志记录等）
            await self._query_done()
            # 返回查询结果
            return response,context
        else:
            # 处理 "local" 模式的查询
            if param.mode == "local":
                response = await local_query_with_data(
                    query,# 查询语句
                    data,
                    self.chunk_entity_relation_graph, # 文档间的实体关系图
                    self.entities_vdb, # 本地实体数据库
                    self.community_reports,  # 社区报告存储
                    self.text_chunks, # 文本块存储
                    param, # 查询参数
                    asdict(self),  # 当前实例的配置信息
                )
            # 处理 "global" 模式的查询
            elif param.mode == "global":
                response = await global_query_with_data(
                    query,
                    data,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.community_reports,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
            # 处理 "naive" 模式的查询
            elif param.mode == "naive":
                response = await naive_query(
                    query,
                    self.chunks_vdb, # 朴素 RAG 查询使用的文本向量数据库
                    self.text_chunks,
                    param,
                    asdict(self),
                )
                # 若查询模式未知，则抛出错误
            else:
                raise ValueError(f"Unknown mode {param.mode}")
            # 查询完成后执行收尾操作（可能是缓存管理、日志记录等）
            await self._query_done()
            # 返回查询结果
            return response

    async def aquery(self, query: str,final,param: QueryParam = QueryParam()):
        print(f"模式: {param.mode}")
        # 若查询模式为 "local"，但 enable_local 未启用，则抛出错误
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        # 若查询模式为 "naive"，但 enable_naive_rag 未启用，则抛出错误
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        # 处理 "local" 模式的查询
        if final:
            if param.mode == "local":
                response,context = await local_query_content(
                    query,# 查询语句
                    self.chunk_entity_relation_graph, # 文档间的实体关系图
                    self.entities_vdb, # 本地实体数据库
                    self.community_reports,  # 社区报告存储
                    self.text_chunks, # 文本块存储
                    param, # 查询参数
                    asdict(self),  # 当前实例的配置信息
                )
            # 处理 "global" 模式的查询
            elif param.mode == "global":
                response,context = await global_query_content(
                    query,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.community_reports,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
            # 处理 "naive" 模式的查询
            elif param.mode == "naive":
                response,context = await naive_query(
                    query,
                    self.chunks_vdb, # 朴素 RAG 查询使用的文本向量数据库
                    self.text_chunks,
                    param,
                    asdict(self),
                )
                # 若查询模式未知，则抛出错误
            else:
                raise ValueError(f"Unknown mode {param.mode}")
            # 查询完成后执行收尾操作（可能是缓存管理、日志记录等）
            await self._query_done()
            # 返回查询结果
            return response,context
        else:
            if param.mode == "local":
                response = await local_query(
                    query,# 查询语句
                    self.chunk_entity_relation_graph, # 文档间的实体关系图
                    self.entities_vdb, # 本地实体数据库
                    self.community_reports,  # 社区报告存储
                    self.text_chunks, # 文本块存储
                    param, # 查询参数
                    asdict(self),  # 当前实例的配置信息
                )
            # 处理 "global" 模式的查询
            elif param.mode == "global":
                response = await global_query(
                    query,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.community_reports,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
            # 处理 "naive" 模式的查询
            elif param.mode == "naive":
                response = await naive_query(
                    query,
                    self.chunks_vdb, # 朴素 RAG 查询使用的文本向量数据库
                    self.text_chunks,
                    param,
                    asdict(self),
                )
                # 若查询模式未知，则抛出错误
            else:
                raise ValueError(f"Unknown mode {param.mode}")
            # 查询完成后执行收尾操作（可能是缓存管理、日志记录等）
            await self._query_done()
            # 返回查询结果
            return response


    async def ainsert(self, string_or_strings):
        # 标记插入操作开始（可能用于形成知识图谱日志、索引准备等）
        await self._insert_start()
        try:
            # 确保 string_or_strings 是一个字符串列表
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            # 计算文档的哈希 ID，并去除首尾空格
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            # 过滤出存储中**尚未存在**的文档
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            # 如果所有文档已存在，则直接返回
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking

            inserting_chunks = get_chunks(
                new_docs=new_docs,# 传入待处理的文档
                chunk_func=self.chunk_func,# 分块方法
                overlap_token_size=self.chunk_overlap_token_size,# 允许的分块重叠大小
                max_token_size=self.chunk_token_size,# 最大 token 数量
            )
            # 过滤出存储中**尚未存在**的文本块
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            # 如果所有文本块已存在，则直接返回
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            # 如果启用了 Naive RAG，则将文本块存入向量数据库
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: no incremental update for communities now, so just drop all 目前不支持社区数据的增量更新，因此直接清空
            await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph 提取或者总结实体并加到图里面
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,# 传入待处理的文本块
                knwoledge_graph_inst=self.chunk_entity_relation_graph, # 关系图存储
                entity_vdb=self.entities_vdb,# 实体向量数据库
                global_config=asdict(self),# 全局配置
                using_amazon_bedrock=self.using_amazon_bedrock,  # 是否使用 Amazon Bedrock
            )
             # 如果未提取到新实体，则直接返回
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            # 更新实体关系图
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- update clusterings of graph 进行图聚类，并生成社区报告 
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )

            # ---------- commit upsertings and indexing 提交数据存储
            await self.full_docs.upsert(new_docs) # 存储新文档
            await self.text_chunks.upsert(inserting_chunks) # 存储新的文本块
        finally:
            # 标记插入操作完成（可能用于索引更新、日志等）
            await self._insert_done()

    async def _insert_start(self):
        """
    插入操作开始前的回调，可能用于索引准备或预处理
        """
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,# 目前仅处理关系图存储
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks) # 并行执行所有回调

    async def _insert_done(self):
        """
    插入操作完成后的回调，确保所有存储系统同步更新索引
        """
        tasks = []
        for storage_inst in [
            self.full_docs, # 完整文档存储
            self.text_chunks, # 文本块存储
            self.llm_response_cache, # LLM 响应缓存
            self.community_reports, # 社区报告存储
            self.entities_vdb, # 实体向量数据库
            self.chunks_vdb, # 文本块向量数据库
            self.chunk_entity_relation_graph, # 实体关系图存储
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks) # 并行执行所有回调

    async def _query_done(self):
        """
    查询操作完成后的回调，确保查询缓存同步
        """
        tasks = []
        for storage_inst in [self.llm_response_cache]:# 仅处理 LLM 缓存
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks) # 并行执行所有回调

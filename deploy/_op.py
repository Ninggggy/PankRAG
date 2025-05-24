import re
import json
import asyncio
import tiktoken
from typing import Union
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter

from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS

import re



def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results




def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=16)
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )



async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )

a

from collections import defaultdict

def filter_and_deduplicate_entities(entities):
    # 打印输入数据
    # print("Input entities:", entities)
    
    # 过滤无效实体
    valid_entities = []
    for entity_list in entities.values():  # 遍历 defaultdict 的值（列表）
        for entity in entity_list:  # 遍历每个实体
            if isinstance(entity, dict):  # 检查是否为字典
                if "entity_name" in entity and entity["entity_name"].lower().strip('"\'') not in ["many", "some", "few"]:
                    valid_entities.append(entity)
                else:
                    print(f"Skipping entity due to invalid name: {entity}")
            else:
                print(f"Warning: Skipping invalid entity (not a dictionary): {entity}")
    
    # print("Valid entities after filtering:", valid_entities)

    # 去重实体
    unique_entities = {}
    for entity in valid_entities:
        normalized_name = entity["entity_name"].lower().strip('"\'')  # 标准化 entity_name
        if normalized_name not in unique_entities:
            unique_entities[normalized_name] = entity
        else:
            print(f"Duplicate entity found: {entity['entity_name']}")
    
    # print("Unique entities after deduplication:", unique_entities)

    return unique_entities


def filter_relations(relations, valid_entities):
    # 提取所有有效实体的 entity_name
    valid_entity_names = {entity["entity_name"] for entity in valid_entities.values()}

    filtered_relations=[]
    # 过滤关系，确保 src_id 和 tgt_id 都在有效实体中
    for relation_list in relations.values():  # 遍历 defaultdict 的值（列表）
        for rel in relation_list:  # 遍历每个关系
            # print("relation:", rel)
            if rel["src_id"] in valid_entity_names and rel["tgt_id"] in valid_entity_names:
                filtered_relations.append(rel)
            else:
                print(f"Skipping relation due to invalid src_id or tgt_id: {rel}")
    # print("filtered relations:", filtered_relations)
    after_relations = {}
    for rel in filtered_relations:
        normalized_name1 = rel["src_id"].lower().strip('"\'')  # 标准化 entity_name
        normalized_name2= rel["tgt_id"].lower().strip('"\'')
        str='('+normalized_name1+','+normalized_name2+')'
        after_relations[str] = rel
    # print("after_relations:", after_relations)
    return after_relations

def prepare_reasoner_message(prompt, caption, question, answer_choices, sub_questions, sub_answers):
    answer_prompt = ''
    for ans_id, ans_str in enumerate(answer_choices):
        answer_prompt = answer_prompt + 'Answer {}: {}\n'.format(str(ans_id+1), ans_str)

    sub_answer_prompt = ''
    flat_sub_questions = []
    for sub_questions_i in sub_questions:
        flat_sub_questions.extend(sub_questions_i)
    flat_sub_answers = []
    for sub_answers_i in sub_answers:
        flat_sub_answers.extend(sub_answers_i)

    assert len(flat_sub_questions) == len(flat_sub_answers)
    for ans_id, ans_str in enumerate(flat_sub_answers):
        sub_answer_prompt = sub_answer_prompt + 'Sub-question: {} Answer: {}\n'.format(flat_sub_questions[ans_id], ans_str)
        
    input_prompt = 'Imperfect Caption: {}\nMain Question: {}\nFour choices: \n{} Existing Sub-questions and answers: \n{}'.format(
        caption, question, answer_prompt, sub_answer_prompt)


    input_prompt = prompt.replace('[placeholder]', input_prompt)
    
    messages = {'role': 'user', 'content': input_prompt}
    
    return messages

async def extract_entities(
    chunks: dict[str, TextChunkSchema],  # 输入：包含文本数据片段的字典，键为文本ID，值为对应的文本片段对象
    knwoledge_graph_inst: BaseGraphStorage,  # 输入：知识图谱实例，用于存储和操作提取的实体和关系
    entity_vdb: BaseVectorStorage,  # 输入：存储实体向量的数据库，用于实体的相似度搜索或存储
    global_config: dict,  # 输入：全局配置字典，包含一些全局参数
    using_amazon_bedrock: bool=False,  # 可选：是否使用 Amazon Bedrock 作为模型平台
) -> Union[BaseGraphStorage, None]:  # 返回：提取后的知识图谱实例或 None（如果没有生成结果）
    # 从 global_config 获取 LLM 使用的函数和其他配置参数
    use_llm_func: callable = global_config["best_model_func"]  # 获取用于提取实体的最优模型函数
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]  # 获取最大提取范围配置

    # 将输入的文本片段按项转换成一个列表，方便后续处理
    ordered_chunks = list(chunks.items())

    # 获取用于实体提取的提示模板
    entity_extract_prompt = PROMPTS["entity_extraction"]  # 获取用于实体提取的提示模板
    context_base = dict(  # 定义用于构建上下文的基本信息
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],  # 元组分隔符，用于提示生成中分割实体对
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],  # 记录分隔符，用于生成记录的分隔符
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],  # 完成分隔符，用于标记模型生成的回答结束
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),  # 将默认的实体类型拼接成一个字符串，作为一个提示参数
    )

    # 获取继续提取的提示
    continue_prompt = PROMPTS["entiti_continue_extraction"]  # 获取继续提取实体的提示模板
    # 获取循环提取的提示
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]  # 获取循环提取实体的提示模板

    # 初始化一些计数变量，用于追踪处理状态
    already_processed = 0  # 已经处理的文本片段数
    already_entities = 0  # 已经提取的实体数
    already_relations = 0  # 已经提取的关系数

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        
        # 初始化最终结果为空字符串
        final_result = ""
        
        try:
            # 首次提取
            first_result = await use_llm_func(hint_prompt)
            if isinstance(first_result, list):
                first_result = first_result[0].get("text", "")  # 安全获取文本
            elif first_result is None:
                first_result = ""
                
            final_result = str(first_result)  # 强制转换为字符串
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result, using_amazon_bedrock)

            # 多次迭代提取
            for now_glean_index in range(entity_extract_max_gleaning):
                try:
                    glean_result = await use_llm_func(continue_prompt, history_messages=history)
                    # 空值安全处理
                    if glean_result is None:
                        print(f"第 {now_glean_index+1} 次提取无结果，跳过")
                        continue
                        
                    # 类型安全转换
                    if isinstance(glean_result, list):
                        glean_result = glean_result[0].get("text", "")
                    clean_glean = str(glean_result).strip()
                    
                    # 更新历史和结果
                    history += pack_user_ass_to_openai_messages(continue_prompt, clean_glean, using_amazon_bedrock)
                    final_result += "\n" + clean_glean  # 添加分隔符

                    # 检查是否继续
                    if_loop_result = await use_llm_func(if_loop_prompt, history_messages=history)
                    if_loop_result = str(if_loop_result).strip().lower()[:3]  # 截取前3字符兼容 "yes"/"y"
                    if if_loop_result not in ("yes", "y"):
                        break
                        
                except Exception as e:
                    print(f"第 {now_glean_index+1} 次提取失败: {str(e)}")
                    break

            # 记录处理
            records = split_string_by_multi_markers(
                final_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]]
            )
            # 后处理过滤空记录
            records = [r.strip() for r in records if r.strip()]
            
            # ========== 增强错误处理 ==========
            maybe_nodes = defaultdict(list)
            maybe_edges = defaultdict(list)
            valid_records = 0
            
            for idx, record in enumerate(records):
                try:
                    # 提取括号内容并验证
                    match = re.search(r"\((.*?)\)", record)
                    if not match:
                        print(f"记录 {idx} 格式错误: 未找到括号内容")
                        continue
                    
                    record_content = match.group(1).strip()
                    if not record_content:
                        print(f"记录 {idx} 内容为空")
                        continue
                    
                    # 分割属性并验证
                    attrs = split_string_by_multi_markers(
                        record_content,
                        [context_base["tuple_delimiter"]]
                    )
                    # if len(attrs) < 3:
                    #     print(f"记录 {idx} 属性不足: {attrs}")
                    #     continue
                    
                    # 实体处理
                    entity = await _handle_single_entity_extraction(attrs, chunk_key)
                    if entity:
                        maybe_nodes[entity["entity_name"]].append(entity)
                        valid_records += 1
                        continue
                        
                    # 关系处理
                    relation = await _handle_single_relationship_extraction(attrs, chunk_key)
                    if relation:
                        key = (relation["src_id"], relation["tgt_id"])
                        maybe_edges[key].append(relation)
                        valid_records += 1
                        
                except Exception as e:
                    print(f"处理记录失败: {record[:50]}... 错误: {str(e)}")
                    
            # 更新计数器
            already_processed += 1
            already_entities += len(maybe_nodes)
            already_relations += len(maybe_edges)
            
            # 进度显示
            progress = already_processed * 100 // len(ordered_chunks)
            now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
            print(
                f"{now_ticks} Processed {already_processed}({progress}%) chunks | "
                f"Valid records: {valid_records} | "
                f"Entities: {len(maybe_nodes)} | "
                f"Relations: {len(maybe_edges)}\r", 
                end="", 
                flush=True
            )
            
            return dict(maybe_nodes), dict(maybe_edges)
            
        except Exception as e:
            print(f"\n处理区块 {chunk_key} 时发生严重错误: {str(e)}")
            return {}, {}  # 返回空结果避免级联错误

    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]  # 展开列表为参数
    )
    print()  # 打印空行（可能用于清除进度条显示）

    # 初始化临时存储容器
    maybe_nodes = defaultdict(list)  # 节点暂存字典，格式 {节点类型: [节点数据列表]}
    maybe_edges = defaultdict(list)   # 边暂存字典，格式 {(源节点,目标节点): [边数据列表]}

    # 合并所有处理结果
    for m_nodes, m_edges in results:  # 遍历每个内容块的处理结果
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)  # 合并节点数据
        for k, v in m_edges.items():
            # 将边转换为排序后的元组，确保无向图特性（A-B 和 B-A 视为同一条边）
            maybe_edges[tuple(sorted(k))].extend(v)

    # 异步合并并插入节点到知识图谱
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(  # 节点合并与插入函数
                k,  # 节点类型（如 Person/Organization）
                v,  # 该类型下的节点列表
                knwoledge_graph_inst,  # 知识图谱实例（注意拼写错误，应为 knowledge_graph_inst）
                global_config  # 全局配置
            )
            for k, v in maybe_nodes.items()
        ]
    )

    # 异步合并并插入边到知识图谱
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(  # 边合并与插入函数
                k[0],  # 源节点
                k[1],  # 目标节点
                v,  # 边数据列表
                knwoledge_graph_inst,  # 知识图谱实例
                global_config  # 全局配置
            )
            for k, v in maybe_edges.items()
        ]
    )

    # 检查实体提取结果
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None

    # 如果有向量数据库实例，更新实体向量数据
    if entity_vdb is not None:
        # 构建向量数据库数据格式
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {  # 生成唯一ID
                "content": dp["entity_name"] + dp["description"],  # 向量化内容
                "entity_name": dp["entity_name"],  # 实体名称
            }
            for dp in all_entities_data  # 遍历所有实体
        }
        await entity_vdb.upsert(data_for_vdb)  # 批量更新向量数据库

    return knwoledge_graph_inst  # 返回更新后的知识图谱实例

def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
) -> tuple[str, int]:
    # TODO
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    def _safe_get(data, keys, default=""):
        """多层安全访问字典值"""
        if isinstance(data, dict):
            for key in keys:
                if key in data:
                    return data[key]
        return default

    # 安全获取基础字段
    title = _safe_get(parsed_output, ["title"], "Report")
    summary = _safe_get(parsed_output, ["summary"])
    
    # 获取findings时添加类型校验
    raw_findings = parsed_output.get("findings", [])
    valid_findings = [
        f for f in raw_findings 
        if isinstance(f, (dict, str))  # 允许字符串或字典
    ]
    
    # 自动转换纯字符串的finding
    processed_findings = []
    for f in valid_findings:
        if isinstance(f, str):
            processed_findings.append({"summary": f, "explanation": ""})
        else:
            processed_findings.append(f)
    
    def finding_summary(finding: dict) -> str:
        return _safe_get(finding, ["summary", "title", 0], "Untitled Finding")
    
    def finding_explanation(finding: dict) -> str:
        return _safe_get(finding, ["explanation", "details", "reason"], "")
    
    # 生成报告内容时添加空值保护
    report_sections = []
    for f in processed_findings:
        section_title = finding_summary(f)
        explanation = finding_explanation(f)
        if section_title or explanation:  # 忽略空内容项
            report_sections.append(
                f"## {section_title}\n\n{explanation}".strip()
            )
    
    # 组装最终报告
    report_content = []
    if title:
        report_content.append(f"# {title}")
    if summary:
        report_content.append(summary)
    if report_sections:
        report_content.append("\n\n".join(report_sections))
    
    return "\n\n".join(report_content) or "No valid report generated"

async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    community_report_prompt = PROMPTS["community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = community_report_prompt.format(input_text=describe)
        response = await use_llm_func(prompt, **llm_extra_kwargs)

        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
):
    # 定义一个异步函数，用于从实体数据中筛选出最相关的社区报告
    # 输入参数：
    # - node_datas: 实体数据列表（每个实体包含 clusters 字段）
    # - query_param: 查询参数对象（控制筛选条件）
    # - community_reports: 社区报告存储实例
    related_communities = []# 存储所有实体关联的社区原始数据
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue# 跳过没有 clusters 字段的实体
        # 将 clusters 字段（JSON字符串）解析为列表，并合并到总列表
        related_communities.extend(json.loads(node_d["clusters"]))
        # 筛选出符合层级要求的社区键，并记录重复次数
    related_community_dup_keys = [
        str(dp["cluster"])# 转换为字符串确保键类型一致
        for dp in related_communities
        if dp["level"] <= query_param.level# 根据查询参数筛选层级
    ]
    # 统计社区键的出现频率（Counter 会自动统计重复次数）
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    # 异步批量获取社区详细信息
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
     # 构建有效社区数据字典（过滤掉获取失败的数据）
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None# 排除空值
    }
        # 对社区键进行复合排序：
    # 1. 主排序：出现频率（降序）
    # 2. 次排序：报告评分（降序，默认-1表示无评分）
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],# 出现次数
            related_community_datas[k]["report_json"].get("rating", -1),# 评分
        ),
        reverse=True,# 降序排列
    )
     # 根据排序结果重组社区数据
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]
# 按 token 数量截断列表（控制上下文长度）
    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],# 计算 token 的内容字段
        max_token_size=query_param.local_max_token_for_community_report,# 最大允许 token 数
    )
    # 如果启用单社区模式，只保留最相关的一个
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports# 返回最终筛选的社区报告列


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    
    all_edges = []
    seen = set()
    
    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge) 
                
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
    )
    return all_edges_data


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # 定义一个异步函数 `_build_local_query_context`，用于构建本地查询上下文。
    # 参数：
    # - `query`: 查询内容。
    # - `knowledge_graph_inst`: 知识图谱存储实例。
    # - `entities_vdb`: 实体向量数据库实例。
    # - `community_reports`: 社区报告存储实例。
    # - `text_chunks_db`: 文本块存储实例。
    # - `query_param`: 查询参数对象。
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    # 使用实体向量数据库 `entities_vdb` 查询与 `query` 最相关的实体，返回前 `top_k` 个结果。
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
     # 使用 `asyncio.gather` 并发获取每个实体的节点数据：
    # - 遍历 `results`，获取每个实体的名称 `entity_name`。
    # - 调用 `knowledge_graph_inst.get_node` 获取节点数据。
    if not all([n is not None for n in node_datas]):
        # 检查是否有节点数据缺失，如果有，记录警告日志。
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    # 使用 `asyncio.gather` 并发获取每个实体的节点度数（即节点的连接数）。
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
        # 构建一个新的节点数据列表 `node_datas`：
    # - 将节点数据 `n`、实体名称 `k["entity_name"]` 和节点度数 `d` 合并为一个字典。
    # - 仅包含节点数据不为 `None` 的实体。
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    # 调用 `_find_most_related_community_from_entities` 函数，查找与实体最相关的社区报告。
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # 调用 `_find_most_related_text_unit_from_entities` 函数，查找与实体最相关的文本单元。
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    # 调用 `_find_most_related_edges_from_entities` 函数，查找与实体最相关的关系边。
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    # 记录日志，显示使用的实体、社区、关系和文本单元的数量。
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    # 初始化实体部分的 CSV 表头。
    for i, n in enumerate(node_datas):
        # 遍历 `node_datas`，将每个实体的信息添加到 `entites_section_list` 中。
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    # 将实体信息列表转换为 CSV 格式的字符串。
    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
        # 初始化关系部分的 CSV 表头。
    for i, e in enumerate(use_relations):
                # 遍历 `use_relations`，将每条关系的信息添加到 `relations_section_list` 中。
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)
    # 将关系信息列表转换为 CSV 格式的字符串。
    communities_section_list = [["id", "content"]]
        # 初始化社区部分的 CSV 表头。
    for i, c in enumerate(use_communities):
                # 遍历 `use_communities`，将每个社区的信息添加到 `communities_section_list` 中。
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)
    # 将社区信息列表转换为 CSV 格式的字符串。
    text_units_section_list = [["id", "content"]]
        # 初始化文本单元部分的 CSV 表头。
    for i, t in enumerate(use_text_units):
                # 遍历 `use_text_units`，将每个文本单元的信息添加到 `text_units_section_list` 中。
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
        # 将文本单元信息列表转换为 CSV 格式的字符串。
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""

async def local_query_with_data_content(
    query,  # 用户的查询字符串
    data,
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱存储实例
    entities_vdb: BaseVectorStorage,  # 实体向量数据库实例
    community_reports: BaseKVStorage[CommunitySchema],  # 社区报告存储实例
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本块存储实例
    query_param: QueryParam,  # 查询参数，包含查询类型和其他选项
    global_config: dict,  # 全局配置字典，包含配置项
) -> str:  # 返回查询结果，字符串类型
    # 从全局配置中获取最佳模型函数（可能是一个调用模型的函数）
    use_model_func = global_config["best_model_func"]
    # 构建查询上下文，可能会涉及多个数据源的检索和处理
    context = await _build_local_query_context(
        query,  # 用户查询
        knowledge_graph_inst,  # 知识图谱实例
        entities_vdb,  # 实体向量数据库实例
        community_reports,  # 社区报告实例
        text_chunks_db,  # 文本块数据库实例
        query_param,  # 查询参数
    )
     # 如果只需要上下文数据，不需要进一步的模型响应，则直接返回上下文
    if query_param.only_need_context:
        return context
    # 如果构建的上下文为空，则返回预定义的失败响应
    if context is None:
        return PROMPTS["fail_response"]
    # 获取系统提示模板
    sys_prompt_temp = PROMPTS["hop_local_answer"]
    # 格式化系统提示，将上下文数据和响应类型插入到模板中
    sys_prompt = sys_prompt_temp.format(
        context_data=context,# 上下文数据
        dependencies=data,# 依赖数据
        response_type=query_param.response_type# 查询响应类型（例如文本或数据）
    )
    response = await use_model_func(
        query,# 用户查询
        system_prompt=sys_prompt,# 格式化后的系统提示
    )
    relevant_context=context+str(data)
    return response,relevant_context

async def local_query_with_data(
    query,  # 用户的查询字符串
    data,
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱存储实例
    entities_vdb: BaseVectorStorage,  # 实体向量数据库实例
    community_reports: BaseKVStorage[CommunitySchema],  # 社区报告存储实例
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本块存储实例
    query_param: QueryParam,  # 查询参数，包含查询类型和其他选项
    global_config: dict,  # 全局配置字典，包含配置项
) -> str:  # 返回查询结果，字符串类型
    # 从全局配置中获取最佳模型函数（可能是一个调用模型的函数）
    use_model_func = global_config["best_model_func"]
    # 构建查询上下文，可能会涉及多个数据源的检索和处理
    context = await _build_local_query_context(
        query,  # 用户查询
        knowledge_graph_inst,  # 知识图谱实例
        entities_vdb,  # 实体向量数据库实例
        community_reports,  # 社区报告实例
        text_chunks_db,  # 文本块数据库实例
        query_param,  # 查询参数
    )
     # 如果只需要上下文数据，不需要进一步的模型响应，则直接返回上下文
    if query_param.only_need_context:
        return context
    # 如果构建的上下文为空，则返回预定义的失败响应
    if context is None:
        return PROMPTS["fail_response"]
    # 获取系统提示模板
    sys_prompt_temp = PROMPTS["hop_local_answer"]
    # 格式化系统提示，将上下文数据和响应类型插入到模板中
    sys_prompt = sys_prompt_temp.format(
        context_data=context,# 上下文数据
        dependencies=data,# 依赖数据
        response_type=query_param.response_type# 查询响应类型（例如文本或数据）
    )
    response = await use_model_func(
        query,# 用户查询
        system_prompt=sys_prompt,# 格式化后的系统提示
    )
    return response
async def local_query_content(
    query,  # 用户的查询字符串
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱存储实例
    entities_vdb: BaseVectorStorage,  # 实体向量数据库实例
    community_reports: BaseKVStorage[CommunitySchema],  # 社区报告存储实例
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本块存储实例
    query_param: QueryParam,  # 查询参数，包含查询类型和其他选项
    global_config: dict,  # 全局配置字典，包含配置项
) -> str:  # 返回查询结果，字符串类型
    # 从全局配置中获取最佳模型函数（可能是一个调用模型的函数）
    use_model_func = global_config["best_model_func"]
    # 构建查询上下文，可能会涉及多个数据源的检索和处理
    context = await _build_local_query_context(
        query,  # 用户查询
        knowledge_graph_inst,  # 知识图谱实例
        entities_vdb,  # 实体向量数据库实例
        community_reports,  # 社区报告实例
        text_chunks_db,  # 文本块数据库实例
        query_param,  # 查询参数
    )
     # 如果只需要上下文数据，不需要进一步的模型响应，则直接返回上下文
    if query_param.only_need_context:
        return context
    # 如果构建的上下文为空，则返回预定义的失败响应
    if context is None:
        return PROMPTS["fail_response"]
    # 获取系统提示模板
    sys_prompt_temp = PROMPTS["local_rag_response"]
    # 格式化系统提示，将上下文数据和响应类型插入到模板中
    sys_prompt = sys_prompt_temp.format(
        context_data=context,# 上下文数据
        response_type=query_param.response_type# 查询响应类型（例如文本或数据）
    )
    response = await use_model_func(
        query,# 用户查询
        system_prompt=sys_prompt,# 格式化后的系统提示
    )
    return response,context
async def local_query(
    query,  # 用户的查询字符串
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱存储实例
    entities_vdb: BaseVectorStorage,  # 实体向量数据库实例
    community_reports: BaseKVStorage[CommunitySchema],  # 社区报告存储实例
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本块存储实例
    query_param: QueryParam,  # 查询参数，包含查询类型和其他选项
    global_config: dict,  # 全局配置字典，包含配置项
) -> str:  # 返回查询结果，字符串类型
    # 从全局配置中获取最佳模型函数（可能是一个调用模型的函数）
    use_model_func = global_config["best_model_func"]
    # 构建查询上下文，可能会涉及多个数据源的检索和处理
    context = await _build_local_query_context(
        query,  # 用户查询
        knowledge_graph_inst,  # 知识图谱实例
        entities_vdb,  # 实体向量数据库实例
        community_reports,  # 社区报告实例
        text_chunks_db,  # 文本块数据库实例
        query_param,  # 查询参数
    )
     # 如果只需要上下文数据，不需要进一步的模型响应，则直接返回上下文
    if query_param.only_need_context:
        return context
    # 如果构建的上下文为空，则返回预定义的失败响应
    if context is None:
        return PROMPTS["fail_response"]
    # 获取系统提示模板
    sys_prompt_temp = PROMPTS["local_rag_response"]
    # 格式化系统提示，将上下文数据和响应类型插入到模板中
    sys_prompt = sys_prompt_temp.format(
        context_data=context,# 上下文数据
        response_type=query_param.response_type# 查询响应类型（例如文本或数据）
    )
    response = await use_model_func(
        query,# 用户查询
        system_prompt=sys_prompt,# 格式化后的系统提示
    )
    return response


async def _map_global_communities(
    query: str,
    communities_data: list[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
):
    """异步映射全局社区数据到查询相关的支持点"""

    # 获取全局配置中的JSON转换函数和最佳模型函数
    use_string_json_convert_func = global_config["convert_response_to_json_func"]  # 用于将文本响应转换为JSON的函数
    use_model_func = global_config["best_model_func"]  # 系统配置的最佳模型调用函数
    
    community_groups = []  # 初始化社区分组列表
    
    # 将社区数据按token限制分组（防止单次处理内容过长）
    while len(communities_data):
        # 截取当前组的数据（基于token数量限制）
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],  # 计算token的字段
            max_token_size=query_param.global_max_token_for_community_report,  # 最大token限制
        )
        community_groups.append(this_group)  # 将当前组添加到分组列表
        communities_data = communities_data[len(this_group) :]  # 移除已处理的数据

    # 定义异步处理单个社区组的内部函数
    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        """处理单个社区数据组的核心逻辑"""
        
        # 构建CSV格式的社区上下文数据
        communities_section_list = [["id", "content", "rating", "importance"]]  # CSV表头
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,  # 序号
                    c["report_string"],  # 社区报告内容
                    c["report_json"].get("rating", 0),  # 社区评分（默认0）
                    c["occurrence"],  # 社区出现频次
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)  # 转换为CSV字符串
        
        # 构建系统提示词
        sys_prompt_temp = PROMPTS["global_map_rag_points1"]  # 获取提示词模板
        sys_prompt = sys_prompt_temp.format(context_data=community_context)  # 插入上下文数据
        
        # 调用模型处理请求
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,  # 完整的系统提示
            **query_param.global_special_community_map_llm_kwargs,  # 模型特殊参数
        )
        
        # 解析模型响应
        data = use_string_json_convert_func(response)  # 将响应文本转换为JSON
        return data.get("points", [])  # 返回解析后的支持点列表

    # 记录分组处理信息
    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    
    # 并行处理所有分组（利用异步并发提升效率）
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    
    return responses  # 返回所有分组的处理结果




async def query_dag_decompose(query:str,global_config:dict)->dict:

    use_model_func = global_config["best_model_func"]
    sys_prompt_temp = PROMPTS["2_query_decom_dag_noseq0"]
    response = await use_model_func(
        query,
        system_prompt=sys_prompt_temp.format(query=query)
        # **query_param.global_special_community_map_llm_kwargs
    )
    return response


async def global_query_with_data(
    query,
    data,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["hop_global_answer"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, dependencies=data,response_type=query_param.response_type
        ),
    )
    return response

async def global_query_with_data_content(
    query,
    data,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["hop_global_answer"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, dependencies=data,response_type=query_param.response_type
        ),
    )
    relevant_context=points_context+str(data)
    return response,relevant_context

async def global_query_content(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response,points_context

async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response

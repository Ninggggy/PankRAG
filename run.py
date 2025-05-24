import os
import logging
import ollama
import numpy as np
from openai import AsyncOpenAI
import asyncio
from PankRAG import GraphRAG, QueryParam

from dotenv import load_dotenv

from PankRAG import GraphRAG
from PankRAG.base import BaseKVStorage
from PankRAG._utils import compute_args_hash, wrap_embedding_func_with_attrs,EmbeddingFunc
from PankRAG._llm import openai_embedding

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)
load_dotenv()

LLM_BASE_URL = ""
LLM_API_KEY = ""
MODEL= "gpt-4o-mini"

EMBEDDING_MODEL = ""
EMBEDDING_MODEL_DIM = 1024 
EMBEDDING_MODEL_MAX_TOKENS = 8192


from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def ollama_model_if_cache(prompt, system_prompt=None, history_messages=[], **kwargs):
    try:
        openai_async_client = AsyncOpenAI(
            api_key=LLM_API_KEY, base_url=LLM_BASE_URL
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Get the cached response if having-------------------
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        if hashing_kv is not None:
            args_hash = compute_args_hash(MODEL, messages)
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]
        # -----------------------------------------------------

        response = await openai_async_client.chat.completions.create(
            model=MODEL, messages=messages, **kwargs
        )

        # Cache the response if having-------------------
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
            )
        # 关键检查点
        if not response.choices:
            print("API 返回空响应，尝试重试...")
            raise IndexError("Empty choices list")
            
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"API 调用失败: {str(e)}")
        return "[ERROR] 模型响应无效"
    
def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)



WORKING_DIR= ""
import asyncio


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR, 
        best_model_func=ollama_model_if_cache,   
        cheap_model_func=ollama_model_if_cache,
        embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_MODEL_DIM, max_token_size=EMBEDDING_MODEL_MAX_TOKENS, func=embedding_func
            ),
    )
    print( 
        asyncio.run(rag.query(
            "What are the key components of programming language structure, and how do they contribute to language expressiveness?"
        ))
    )
  
import json
from pathlib import Path
from typing import List, Dict

def process_questions(input_path: str, output_path: str):

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_MODEL_DIM, 
            max_token_size=EMBEDDING_MODEL_MAX_TOKENS, 
            func=embedding_func
        ),
    )


    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            questions_data: List[Dict] = json.load(f)


    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"输入文件解析失败: {str(e)}")


    results = []
    for idx, item in enumerate(questions_data, 1):
        if "question" not in item:
            print(f"警告：跳过第 {idx} 条数据，缺少question字段")
            continue
        
        question = item["question"]
        try:
            print(f"正在处理问题 {idx}/{len(questions_data)}: {question[:50]}...")
            

            loop = asyncio.get_event_loop()
            # answer, querydec= loop.run_until_complete(rag.query(
            #     question
            # ))
            answer, querydec,context = loop.run_until_complete(rag.query(
                question,param=QueryParam(mode="local")
            ))
            # print("回答:", answer)
            print("context:",context)

            results.append({
                "question": question,
                "answer": answer,
                "querydec": querydec,
                "context":context,
                "status": "success"
            })
        except Exception as e:
            print(f"处理问题失败: {str(e)}")
            results.append({
                "question": question,
                "error": str(e),
                "status": "failed"
            })


    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存至 {output_path}")
    except IOError as e:
        print(f"结果保存失败: {str(e)}")

def insert():
    from time import time

    with open("", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    # remove_if_exist(f"{WORKING_DIR}/vdb_entities.json") 
    # remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    # remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    # remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    # remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_MODEL_DIM, max_token_size=EMBEDDING_MODEL_MAX_TOKENS, func=embedding_func
            ),
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)




# We're using Ollama to generate embeddings for the BGE model 
@wrap_embedding_func_with_attrs(
    embedding_dim= EMBEDDING_MODEL_DIM,
    max_token_size= EMBEDDING_MODEL_MAX_TOKENS,
)


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="",
        api_key="",
        base_url="",
    )

if __name__ == "__main__":
    # insert()
    # query()
    process_questions("","")

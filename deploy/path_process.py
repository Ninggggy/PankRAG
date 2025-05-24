import asyncio
from graphlib import TopologicalSorter
from collections import defaultdict
from typing import Dict, Set
import asyncio
from .base import (
    QueryParam
)
class RealTimePathProcessor:
    def __init__(self):
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
        if not hasattr(self, 'graph_rag'):
            from .graphrag import GraphRAG  # 延迟导入防循环引用
            self.graph_rag = GraphRAG()
            
        # 模拟处理逻辑（替换为实际处理代码）
        print(f"Processing {hop['hop_id']}: {hop['sub_query']}")
        if_abstract = hop.get('if_abstract', False)
        param.mode = 'global' if if_abstract else 'local'
        answer = await self.graph_rag.aquery_with_data(sub_query, subq_data, param)
        
        # 记录处理完成的hop
        self.processed_hops.add(hop['hop_id'])
        self.results[hop['hop_id']] = answer

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

    async def execute(self, data):
        # Step 1: 处理非0序列（并行）
        main_sequences = [
            seq for seq in data['path']['sequences']
            if seq['sequence_id'] != "0"
        ]
        await asyncio.gather(*[
            self.process_sequence(seq) for seq in main_sequences
        ])

        # Step 2: 处理0序列
        await self.process_zero_sequence(data['path']['sequences'])

        # Step 3: 处理最终查询
        await self.process_final_query(data['path']['final_query'])
       
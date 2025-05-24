import torch
from transformers import (
    AlbertTokenizer,  # 修改1：替换为Albert专用Tokenizer
    AlbertForSequenceClassification,  # 修改2：替换为Albert模型类
    pipeline
)
from sklearn.metrics import f1_score, accuracy_score, recall_score
import json
from pathlib import Path
import os

# ----------------------------------------
# 参数配置（需与训练时一致）
# ----------------------------------------
MODEL_PATH = "D:\\KGandLLM\\demo of querydecom\\results_albert\\best_model"  # 微调后的模型路径
NUM_LABELS = 2                            # 必须与训练时一致
MODEL_NAME = "albert/albert-base-v2"  # ALBERT基础模型
tokenizer = AlbertTokenizer.from_pretrained("D:\\KGandLLM\\original_model\\albert")

# ----------------------------------------
# 加载微调后的模型和tokenizer
# ----------------------------------------

# 方案1：直接使用pipeline（推荐简单场景）
def validate_json_dataset(input_path, output_path="predictions_withsimple97_albert.json"):
    """验证JSON格式的测试数据集"""
    # 加载测试数据
    test_data = []
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                test_data.append(json.loads(line))
    except Exception as e:
        print(f"加载测试数据失败：{str(e)}")
        return

    # 确保数据格式正确
    if not isinstance(test_data, list):
        print("测试数据应为JSON数组格式")
        return
    
    # 提取问题和标签（如果存在）
    questions = []
    true_labels = []
    for item in test_data:

        questions.append(item["text"])
        if "label" in item:  # 如果有真实标签
            true_labels.append(item["label"])

    # 执行批量预测
    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        framework="pt"
    )
    predictions = classifier(questions)

    # 组装结果
    results = []
    for i, (item, pred) in enumerate(zip(test_data, predictions)):
        result = {
            "id": item.get("id", i),
            "question": item["text"],
            "predicted_label": int(pred["label"].split("_")[-1]),  # 转换LABEL_0 -> 0
            "confidence": round(pred["score"], 4)
        }
        if "label" in item:
            result["true_label"] = item["label"]
        results.append(result)

    # 保存预测结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"预测结果已保存至 {output_path}")

    # 如果有真实标签则计算指标
    if true_labels:
        y_true = true_labels
        y_pred = [p["predicted_label"] for p in results]
        
        print("\n评估指标：")
        print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1分数: {f1_score(y_true, y_pred, average='binary'):.4f}")
        print(f"召回率: {recall_score(y_true, y_pred):.4f}")

def validate_single_question(question):
    """验证单个问题"""
    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        framework="pt"
    )
    result = classifier(question)
    # results = []
    # results = {
    #     # "id": question.get("id"),
    #     "question": question,
    #     "predicted_label": int(result["label"].split("_")[-1]),  # 转换LABEL_0 -> 0
    #     "confidence": round(result["score"], 4)
    # }
    print(f"问题：{question}")
    print(result)
    # print(f"预测类别：{results['label']}，置信度：{results['score']:.2f}\n")
# 方案2：手动加载模型（推荐复杂场景）
# def validate_manually():
#     # 加载tokenizer
#     tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    
#     # 加载模型
#     model = RobertaForSequenceClassification.from_pretrained(
#         MODEL_PATH,
#         num_labels=NUM_LABELS,
#         ignore_mismatched_sizes=True  # 关键参数
#     ).to("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
    
#     # 测试样本
#     test_texts = [
#         "How many movies did Alfred Hitchcock direct?",
#         "What was the death cause of the writer of the Cranberries?"
#     ]
    
#     # 执行预测
#     for text in test_texts:
#         inputs = tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=128,
#             return_tensors="pt"
#         ).to(model.device)
        
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
#         predicted_class = torch.argmax(probabilities).item()
#         confidence = probabilities[0][predicted_class].item()
        
#         print(f"问题：{text}")
#         print(f"预测类别：{predicted_class}，置信度：{confidence:.2f}\n")

# ----------------------------------------
# 执行验证
# ----------------------------------------
# if __name__ == "__main__":
#     print("=== Pipeline验证 ===")
#     TEST_PATH = "D:\\KGandLLM\\demo of querydecom\\finetune_roberta\\data\\processed\\hotpot_test_fullwiki_v1.jsonl"  # 修改为实际路径
#     # validate_json_dataset(TEST_PATH)
#     validate_single_question("What ethical concerns about AI misuse are discussed in the dataset?")
#     # print("\n=== 手动验证 ===")
#     # validate_manually()
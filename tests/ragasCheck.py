import json
from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# 1. 导入你的 RAG 加载逻辑（引用你之前的 app.py 中的函数或组件）
from src.rag_chat import create_rag_chain


def run_evaluation(file_path: str = "./data/golden_dataset.json",output_path: str = "./data/ragas_final_report.csv"):
    print("🚀 正在加载 RAG 引擎与评估数据集...")

    # 加载 RAG 链与测试数据
    rerank_retriever, rag_chain = create_rag_chain()

    with open(file_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    print(f"📊 黄金测试集加载完毕，共包含 {len(golden_data)} 条测试用例。")
    print("⚡ 开始批量跑评测，生成 RAG 预测结果...\n")

    # 2. 遍历测试集中所有问题，进行批量预测与上下文收集
    for i, item in enumerate(golden_data):
        q = item["question"]
        gt = item["ground_truth"]

        print(f"[{i+1}/{len(golden_data)}] 测试问题: {q}")

        # A. 执行检索，提取 Parent Docs 作为 Context
        retrieved_docs = rerank_retriever.invoke(q)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        print(f"\n🔍 [Q{i+1} 脚本真实检索到的 Docs 数量]: {len(retrieved_docs)}")
        for doc in retrieved_docs[:2]:
            print(
                f"   - Parent ID: {doc.metadata.get('parent_id')} | Preview: {doc.page_content[:100]}..."
            )

        # B. 执行 LLM 生成
        response = rag_chain.invoke({"context": retrieved_docs, "query": q})

        questions.append(q)
        ground_truths.append([gt])  # Ragas 要求 ground_truths 为 list[str]
        answers.append(response)
        contexts.append(retrieved_texts)

    # 3. 构建 Ragas 评估所需的大模型（由于国内/本地环境配置差异，这里展示标准测评数据集打包）
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [gt[0] for gt in ground_truths],
    }

    eval_dataset = Dataset.from_dict(data_dict)

    # 保存预测过程数据，防止断网或计算失败
    df_results = pd.DataFrame(data_dict)
    df_results.to_csv(output_path, index=False)
    print("\n✅ RAG 运行预测结果已保存至 {output_path}，可用于后续 Ragas 评测。")

    # 4. 执行 Ragas 多维度定量评估打分
    print("\n🧮 启动 Ragas 多维度自动评分器 (Faithfulness, Recall, Precision, Relevance)...")
    try:
        # 注意：默认 Ragas 需要 OpenAI API 评估，或者通过 LangChain 传入本地 Ollama / BGE 模型
        results = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ],
        )

        print("\n==========================================")
        print("🎉 RAG 系统评估最终得分 (Overall RAGAS Score):")
        print("==========================================")
        print(results)

        # 导出评测报表
        results_df = results.to_pandas()
        results_df.to_csv(output_path, index=False)
        print(f"📄 详细评测报告已成功导出至 {output_path}")

    except Exception as e:
        print(f"\n⚠️ Ragas 评分打分器执行异常 (可检查 API / 本地 Judge 模型配置): {e}")
        print("💡 原始生成数据已完好保存在 csv 文件中，可以直接手动查看召回的 Context 与 Answer 对比。")

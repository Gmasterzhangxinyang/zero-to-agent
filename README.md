# Zero to Agent

> Learn AI Agent development from scratch: LLM fundamentals, fine-tuning, RAG, LangChain, and MCP. 7-day hands-on curriculum with runnable code.

---

## 课程结构

| 天 | 主题 | 内容 |
|---|---|---|
| Day 1 | **LLM 推理基础** | Transformer 原理、Token、推理参数、Ollama 本地部署 |
| Day 2 | **微调：LoRA/QLoRA** | 高效微调、远程服务器训练 7B 模型 |
| Day 3 | **RAG 基础** | 向量数据库、Embedding、本地知识库问答 |
| Day 4 | **RAG 进阶** | 混合检索、Reranker、RAGAS 评估 |
| Day 5 | **AI Agent** | ReAct、工具调用、多步规划 |
| Day 6 | **LangChain / LangGraph** | 有状态工作流、LangSmith 追踪 |
| Day 7 | **MCP** | Model Context Protocol、自定义 MCP Server |

---

## 特点

- 每天一个主题，快速上手
- 每个概念配可运行代码
- 底层原理 + 工程实践并重
- 涉及训练的内容提供远程服务器脚本

---

## 环境要求

- Python 3.10+
- [Ollama](https://ollama.com)（本地推理）
- NVIDIA GPU 服务器（Day 2 微调用）

---

## 开始学习

```bash
git clone https://github.com/Gmasterzhangxinyang/zero-to-agent.git
cd zero-to-agent
conda create -n agent python=3.10
conda activate agent
```

从 [Day 1](./day1_llm_basics/README.md) 开始。

---

## License

MIT

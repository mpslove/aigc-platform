# AIGC Platform —— 多模态AI视频生成平台

多模态AI视频/图像推理生成平台，集成 Agent 编排、多 Provider 网关、RAG 检索增强、Gradio WebUI 和 REST API，支持从文本描述到完整视频的端到端生成。

## 架构

```
src/
├── agent/          # Agent 编排引擎
│   ├── orchestrator.py   # 主调度器（ReAct 风格）
│   ├── tools.py          # 工具定义框架（含多模态工具）
│   └── prompts/          # Agent prompt 模板
├── understanding/  # 多模态理解引擎
│   └── qwen_engine.py    # Qwen2-VL 推理（图片描述/视觉问答/场景分析）
├── pipeline/       # 视频/图像生产管线
│   ├── composer.py       # FFmpeg 视频合成（转场/字幕/背景音乐）
│   ├── generator.py      # AI 内容生成编排
│   ├── script_writer.py  # 剧本/Template 管理
│   └── schema.py         # 数据模型
├── gateway/        # AI Provider 网关
│   ├── base.py           # 统一接口定义
│   ├── agnes.py          # Agnes AI（免费图像/视频 API）
│   ├── comfyui.py        # ComfyUI（本地 GPU）
│   └── factory.py        # Provider 工厂
├── rag/            # 多模态检索增强生成
│   ├── embedder.py       # CLIP 特征提取（图文统一向量空间）
│   ├── vector_store.py   # FAISS 向量存储
│   ├── visual_rag.py     # 端到端 Visual RAG 管线
│   └── reranker.py       # Cross-encoder 重排序
├── api/            # REST API
│   └── app.py            # FastAPI 应用（理解/检索/生成 全端点）
├── webui/          # Web 界面
│   └── app.py            # Gradio WebUI（端口 7860）
├── eval/           # 质量评估
│   └── metrics.py        # 视频/图像质量指标
└── utils/          # 通用工具
    └── config.py         # 配置加载
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 WebUI
python src/webui/app.py
# → 浏览器打开 http://localhost:7860

# 启动 API
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
# → API 文档 http://localhost:8000/docs
```

## 配置

编辑 `config/providers.yaml` 配置 AI Provider：

| Provider | 类型 | 说明 |
|----------|------|------|
| Agnes AI | 免费 API | 图像/视频生成，需设置 `AIGC_AGNES_API_KEY` 环境变量 |
| ComfyUI | 本地 GPU | 需安装 ComfyUI 并启动服务 |

也可以在 `.env` 文件中设置环境变量覆盖配置。

## 功能

- **多模态理解**：Qwen2-VL 图片描述、视觉问答、场景分析（CPU/GPU 自动降级）
- **Visual RAG**：CLIP 特征提取 + FAISS 索引 + Cross-encoder 重排序 + 跨模态检索
- **多 Provider 网关**：统一接口切换不同 AI 生成后端（Agnes / ComfyUI）
- **Agent 编排**：ReAct 风格调度，支持多模态工具（理解/检索/生成）
- **AIGC 生成**：视频/图像 AI 生成 + FFmpeg 合成（转场/字幕/背景音乐）
- **REST API**：完整的 FastAPI 接口，覆盖理解、检索、生成全链路
- **WebUI**：Gradio 交互界面
- **质量评估**：自动计算视频/图像质量评分

## 测试

```bash
pytest tests/ -v
# 126 tests passed
# ├── agent/core     — 6 tests
# ├── api           — 8 tests
# ├── eval          — 20 tests
# ├── gateway       — 10 tests
# ├── pipeline      — 30 tests
# ├── rag           — 21 tests
# ├── understanding — 9 tests
# └── api_multimodal — 8 tests
```

## 环境要求

- Python 3.10+
- FFmpeg（用于视频合成）
- 可选：GPU + CUDA（用于 ComfyUI 本地生成）

---

## Demo 样片

通过 Agnes AI Provider 生成的样片（猫片海滩），验证端到端管线可用：

[→ `examples/agnes_cat_beach.mp4`](examples/agnes_cat_beach.mp4)

生成方式：
```bash
export AIGC_AGNES_API_KEY=your_key
python src/webui/app.py
# WebUI 输入 topic 或选择 template → 生成 → 合成 → 输出
```

> 真实生成产物。~/1.9MB, 1248×832, 3.0s

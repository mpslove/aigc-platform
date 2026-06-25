# AIGC Platform —— 多模态AI视频生成平台

多模态AI视频/图像推理生成平台，集成 Agent 编排、多 Provider 网关、RAG 检索增强、Gradio WebUI 和 REST API，支持从文本描述到完整视频的端到端生成。

## 架构

```
src/
├── agent/          # Agent 编排引擎
│   ├── orchestrator.py   # 主调度器（ReAct 风格）
│   ├── tools.py          # 工具定义框架
│   └── prompts/          # Agent prompt 模板
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
├── api/            # REST API
│   └── app.py            # FastAPI 应用
├── webui/          # Web 界面
│   └── app.py            # Gradio WebUI（端口 7860）
├── rag/            # 检索增强生成
│   └── vector_store.py   # FAISS 向量存储
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

- **多 Provider 网关**：统一接口切换不同 AI 生成后端
- **Agent 编排**：ReAct 风格调度，支持工具调用
- **视频合成**：FFmpeg 支持转场效果、字幕叠加、背景音乐
- **RAG 检索**：FAISS 向量库 + Sentence Transformer 语义检索
- **WebUI**：Gradio 交互界面，模板/主题输入即刻生成
- **REST API**：完整的 FastAPI 接口，适合 CI/CD 集成
- **质量评估**：自动计算视频/图像质量评分

## 测试

```bash
pytest tests/ -v
# 109 tests passed
```

## 环境要求

- Python 3.10+
- FFmpeg（用于视频合成）
- 可选：GPU + CUDA（用于 ComfyUI 本地生成）

## 验证数据

| 指标 | 值 |
|------|-----|
| 测试用例 | **109 tests, 全部通过** |
| 模块覆盖 | **6个模块** — agent / api / eval / gateway / pipeline / rag |
| 质量评估系统 | **6/6 打分规则验证通过** — 分辨率/音频/异常扣分准确率 100% |
| 内置模板 | **3套** — nursing-ad（6场景~22s）、product-ad、travel-ad |
| 样片质量 | seg_title.mp4 — 1248×832, 45/100 (无音频扣分) |
| API 启动 | FastAPI + Uvicorn，REST 端点可用 |
| WebUI 启动 | Gradio 交互界面，端口 7860 |

运行验证：
```bash
# 测试
pytest tests/ -v

# Benchmark
python scripts/benchmark.py

# 启动 WebUI
python src/webui/app.py
```

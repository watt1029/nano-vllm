# Nano-vLLM 源码导读

Nano-vLLM 是一个从零实现、用于学习 **大模型推理框架原理** 的轻量级 vLLM。整个 `nanovllm` 包约 **1400 行** Python 代码，却完整覆盖了一个现代推理引擎的核心机制：

- **Continuous Batching**（连续批处理）——调度器每步动态决定要执行哪些请求
- **Paged KV Cache**（分页 KV 缓存）——以固定大小 block 管理显存，几乎不浪费
- **Prefix Caching**（前缀缓存）——基于内容哈希的公共前缀复用
- **Chunked Prefill**（分块预填充）——避免长 prompt 阻塞 decode 的小批次
- **Tensor Parallelism**（张量并行）——多卡分片模型与 KV Cache
- **CUDA Graph** 与 **torch.compile**——降低 Kernel 启动开销

## 文档索引

| 文档 | 内容 |
|------|------|
| [01-architecture.md](01-architecture.md) | 总体架构：目录结构、模块职责、进程模型、组件协作图 |
| [02-request-lifecycle.md](02-request-lifecycle.md) | 请求生命周期：`generate()` 主循环、状态机、时序图 |
| [03-scheduler.md](03-scheduler.md) | 调度器：prefill/decode 两阶段调度、chunked prefill、抢占 |
| [04-paged-kvcache.md](04-paged-kvcache.md) | Paged KV Cache 与 BlockManager：分页、分配、前缀缓存 |
| [05-model-execution.md](05-model-execution.md) | 模型执行：ModelRunner、数据组织、Qwen3 前向、采样 |
| [06-optimizations.md](06-optimizations.md) | 优化技术：张量并行、CUDA Graph、torch.compile、分块预填充 |

## 建议阅读顺序

对于刚接触推理框架的读者，建议按下面顺序阅读：

1. 先读 [01-architecture.md](01-architecture.md)，建立全局认知：一个请求从进来到出去的完整链路。
2. 再读 [02-request-lifecycle.md](02-request-lifecycle.md)，理解引擎主循环 `while not finished: step()` 的心跳节奏。
3. 深入 [03-scheduler.md](03-scheduler.md) 与 [04-paged-kvcache.md](04-paged-kvcache.md)，这两个模块是推理框架与训练代码最核心的区别。
4. 读 [05-model-execution.md](05-model-execution.md)，看 GPU 侧如何组织变长 batch 喂给 flash-attention。
5. 最后读 [06-optimizations.md](06-optimizations.md)，了解吞吐量优化手段。

## 代码速查

```text
nanovllm/
├── llm.py                 # LLM 类（即 LLMEngine 的别名）
├── config.py              # Config：全局配置（KV 块数、batch 上限等）
├── sampling_params.py     # SamplingParams：每请求采样参数
├── engine/                # ★ 推理引擎核心
│   ├── llm_engine.py      #   LLMEngine：用户 API + 主循环 step()
│   ├── scheduler.py       #   Scheduler：决定每步执行哪些请求
│   ├── block_manager.py   #   BlockManager：KV 块分配 / 释放 / 前缀缓存
│   ├── sequence.py        #   Sequence：请求状态与 token 容器
│   └── model_runner.py    #   ModelRunner：GPU 执行、KV Cache、CUDA Graph、TP 进程
├── layers/                # 网络层（注意力 / 线性 / 归一化 / RoPE / 采样）
├── models/qwen3.py        # Qwen3 模型本体
└── utils/
    ├── context.py         # Context：每步注意力的元数据（global 传递）
    └── loader.py          # safetensors 权重加载（含打包 + TP 分片）
```

> 图中使用的所有流程图均为 Mermaid 语法，在 GitHub / VSCode 中可直接渲染。

# Cloud Transcribe — 云端 GPU 字幕提取

## 工作流

1. **本地**: 运行 `extract_audio.py` 提取音频
2. **上传**: 把整个 `cloud_transcribe/` 目录传到云服务器
3. **云端**: 安装环境 + 运行 `batch_transcribe.py`
4. **下载**: 把 `subtitle/` 目录下载回本地

---

## 云端环境搭建

### 1. 安装 Python 依赖

```bash
pip install faster-whisper
```

faster-whisper 依赖 CTranslate2，会自动安装。需要 CUDA 环境（CUDA 11.x 或 12.x）。

### 2. 下载模型

脚本默认使用 **large-v3** 模型。首次运行会自动从 HuggingFace 下载。

如果网络不通，可手动下载：

**HuggingFace 模型地址:**
- large-v3: https://huggingface.co/Systran/faster-whisper-large-v3

手动下载方式：

```bash
# 方式一: git clone（需安装 git-lfs）
git lfs install
git clone https://huggingface.co/Systran/faster-whisper-large-v3

# 方式二: huggingface-cli
pip install huggingface_hub
huggingface-cli download Systran/faster-whisper-large-v3 --local-dir faster-whisper-large-v3
```

下载后将模型目录放到任意位置，然后修改 `batch_transcribe.py` 中的 `MODEL_SIZE` 为模型目录的绝对路径：

```python
MODEL_SIZE = "/path/to/faster-whisper-large-v3"
```

### 3. 验证 GPU 可用

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 模型对比

| 模型 | 参数量 | VRAM 占用 | 速度 | 准确度 |
|------|--------|-----------|------|--------|
| medium | 769M | ~2 GB | 快 | 良好 |
| large-v2 | 1550M | ~4 GB | 中等 | 优秀 |
| **large-v3** | 1550M | ~4 GB | 中等 | **最佳** |

---

## 配置说明

编辑 `batch_transcribe.py` 顶部：

```python
MODEL_SIZE = "large-v3"      # 模型名称或本地路径
DEVICE = "cuda"              # cuda 或 cpu
COMPUTE_TYPE = "float16"     # GPU: float16, CPU: int8
LANGUAGE = "ja"              # ja/zh/en/留空自动检测
```

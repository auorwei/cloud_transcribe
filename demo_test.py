"""
云端测试脚本 — 验证 GPU + faster-whisper 是否正常工作
用法: python demo_test.py
"""

import sys
import time

# 1. 检查 CUDA
print("=" * 40)
print("1. 检查 CUDA 环境")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        print("   警告: CUDA 不可用，将使用 CPU")
except ImportError:
    print("   跳过 (torch 未安装，不影响 faster-whisper)")

# 2. 检查 faster-whisper
print("\n2. 检查 faster-whisper")
try:
    from faster_whisper import WhisperModel
    print("   faster-whisper 已安装")
except ImportError:
    print("   未安装! 请运行: pip install faster-whisper")
    sys.exit(1)

# 3. 下载并加载模型
print("\n3. 加载 large-v3 模型 (首次会自动下载 ~3GB)")
t0 = time.time()
try:
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print(f"   模型加载成功，耗时 {time.time() - t0:.1f}s")
except Exception as e:
    print(f"   GPU 加载失败: {e}")
    print("   尝试 CPU 模式...")
    try:
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        print(f"   CPU 模式加载成功，耗时 {time.time() - t0:.1f}s")
    except Exception as e2:
        print(f"   失败: {e2}")
        sys.exit(1)

# 4. 用一小段音频测试转写
print("\n4. 测试转写")
from pathlib import Path

audio_dir = Path(__file__).parent / "audio"
wav_files = sorted(audio_dir.glob("*.mp3"))

if wav_files:
    test_file = wav_files[0]
    print(f"   测试文件: {test_file.name}")
    t0 = time.time()
    segments, info = model.transcribe(str(test_file), language="ja", beam_size=5, vad_filter=True)
    # 只取前5段
    for i, seg in enumerate(segments):
        print(f"   [{seg.start:.1f}s -> {seg.end:.1f}s] {seg.text.strip()}")
        if i >= 4:
            print("   ... (仅显示前5段)")
            break
    print(f"   转写耗时 {time.time() - t0:.1f}s")
else:
    print("   audio/ 目录为空，跳过转写测试")
    print("   (请先在本地运行 extract_audio.py 提取音频)")

print("\n" + "=" * 40)
print("测试完成!")

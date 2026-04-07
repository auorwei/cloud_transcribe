"""
云端运行 — 使用 faster-whisper + GPU 批量转写音频为字幕
用法: python batch_transcribe.py
"""

import sys
import time
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("请先安装: pip install faster-whisper")
    sys.exit(1)

# ============ 配置 ============
AUDIO_DIR = Path(__file__).parent / "audio"
OUTPUT_DIR = Path(__file__).parent / "subtitle"
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # GPU 用 float16，CPU 用 int8
LANGUAGE = "ja"
# ==============================


def format_ts(seconds: float) -> str:
    """秒数转 SRT 时间戳 HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcribe_file(model: WhisperModel, audio_path: Path, srt_path: Path) -> bool:
    try:
        segments, info = model.transcribe(
            str(audio_path),
            language=LANGUAGE if LANGUAGE else None,
            beam_size=5,
            vad_filter=True,
        )

        lines = []
        for idx, seg in enumerate(segments, 1):
            lines.append(f"{idx}")
            lines.append(f"{format_ts(seg.start)} --> {format_ts(seg.end)}")
            lines.append(seg.text.strip())
            lines.append("")

        if not lines:
            print("  警告: 未识别到任何语音")
            return False

        srt_path.write_text("\n".join(lines), encoding="utf-8")
        return True

    except Exception as e:
        print(f"  异常: {e}")
        if srt_path.exists():
            srt_path.unlink()
        return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_audio = sorted(AUDIO_DIR.glob("*.mp3"))
    done_stems = {f.stem for f in OUTPUT_DIR.glob("*.srt")}
    todo = [f for f in all_audio if f.stem not in done_stems]

    print(f"音频总数: {len(all_audio)}  已完成: {len(done_stems)}  待处理: {len(todo)}")

    if not todo:
        print("全部完成!")
        return

    print(f"模型: {MODEL_SIZE}  设备: {DEVICE}  计算精度: {COMPUTE_TYPE}")
    print("正在加载模型...")
    t0 = time.time()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"模型加载完成，耗时 {format_time(time.time() - t0)}")
    print("=" * 50)

    success_count = 0
    fail_list = []

    try:
        for i, audio in enumerate(todo, 1):
            srt_path = OUTPUT_DIR / f"{audio.stem}.srt"
            print(f"\n[{i}/{len(todo)}] {audio.name}")

            start = time.time()
            ok = transcribe_file(model, audio, srt_path)
            elapsed = time.time() - start

            if ok:
                success_count += 1
                print(f"  完成，耗时 {format_time(elapsed)}")
            else:
                fail_list.append(audio.name)

    except KeyboardInterrupt:
        print("\n\n用户中断，已安全停止。")

    print("\n" + "=" * 50)
    print(f"成功: {success_count}  失败: {len(fail_list)}  "
          f"未处理: {len(todo) - success_count - len(fail_list)}")
    if fail_list:
        print(f"失败文件: {', '.join(fail_list)}")
    print("重新运行脚本可继续处理剩余文件。")


if __name__ == "__main__":
    main()

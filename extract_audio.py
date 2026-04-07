"""
本地运行 — 从待处理视频中提取音频（WAV 16kHz 单声道）
用法: python extract_audio.py
"""

import subprocess
import sys
from pathlib import Path

ORIGIN_DIR = Path(r"D:\projects\whisper\origin")
SUBTITLE_DIR = Path(r"D:\projects\whisper\subtitle")
AUDIO_DIR = Path(__file__).parent / "audio"
FFMPEG_EXE = r"D:\projects\whisper\Faster-Whisper-XXL\ffmpeg.exe"

VIDEO_EXTS = {".wmv", ".avi", ".mp4", ".mkv", ".mov", ".flv", ".webm"}


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    all_videos = sorted(
        f for f in ORIGIN_DIR.iterdir()
        if f.suffix.lower() in VIDEO_EXTS and not f.stem.endswith("_sub")
    )
    done_stems = {f.stem for f in SUBTITLE_DIR.glob("*.srt")}
    audio_stems = {f.stem for f in AUDIO_DIR.glob("*.wav")}
    todo = [
        f for f in all_videos
        if f.stem not in done_stems and f.stem not in audio_stems
    ]

    print(f"视频总数: {len(all_videos)}  已有字幕: {len(done_stems)}  "
          f"已提取音频: {len(audio_stems)}  待提取: {len(todo)}")

    if not todo:
        print("全部完成!")
        return

    for i, video in enumerate(todo, 1):
        out_wav = AUDIO_DIR / f"{video.stem}.wav"
        print(f"[{i}/{len(todo)}] {video.name} -> {out_wav.name}")

        cmd = [
            FFMPEG_EXE,
            "-i", str(video),
            "-vn",                  # 不要视频
            "-acodec", "pcm_s16le", # 16-bit PCM
            "-ar", "16000",         # 16kHz
            "-ac", "1",             # 单声道
            "-y",                   # 覆盖
            str(out_wav),
        ]

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )

        if result.returncode != 0 or not out_wav.exists():
            print(f"  失败: {result.stderr[-200:]}")
        else:
            size_mb = out_wav.stat().st_size / 1024 / 1024
            print(f"  完成 ({size_mb:.1f} MB)")

    print(f"\n音频文件已保存到: {AUDIO_DIR}")
    print("请将整个 cloud_transcribe 目录上传到云服务器。")


if __name__ == "__main__":
    main()

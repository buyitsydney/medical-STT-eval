#!/usr/bin/env python3
"""
并发执行 Speechmatics Medical Domain 评测
"""
import os
import sys
import time
import concurrent.futures
from pathlib import Path

# 添加 transcribe 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'transcribe'))

from speechmatics_transcribe import SpeechmaticsTranscriber

def process_file(audio_file, transcriber):
    """处理单个文件"""
    try:
        result = transcriber.transcribe_file(audio_file)
        return (audio_file, True, result.duration)
    except Exception as e:
        return (audio_file, False, str(e))

def main():
    # 配置
    audio_dir = Path("data/raw_audio")
    transcript_dir = Path("results/transcripts/speechmatics-enhanced-medical")
    excluded = ["day1_consultation07", "day3_consultation03"]
    max_workers = 10  # 并发数
    
    # 获取已完成的文件
    done_files = set()
    if transcript_dir.exists():
        for f in transcript_dir.glob("*_transcript.txt"):
            done_files.add(f.stem.replace("_transcript", ""))
    
    # 获取待处理的文件
    todo_files = []
    for wav in sorted(audio_dir.glob("*.wav")):
        base = wav.stem
        if any(ex in base for ex in excluded):
            continue
        if base not in done_files:
            todo_files.append(str(wav))
    
    print(f"已完成: {len(done_files)} 个文件")
    print(f"待处理: {len(todo_files)} 个文件")
    print(f"并发数: {max_workers}")
    print()
    
    if not todo_files:
        print("✅ 所有文件已完成！")
        return
    
    # 创建转写器
    transcriber = SpeechmaticsTranscriber(
        operating_point="enhanced",
        domain="medical",
        results_dir="results"
    )
    
    print(f"🚀 开始并发处理 {len(todo_files)} 个文件...")
    print("=" * 60)
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    # 并发执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, f, transcriber): f 
            for f in todo_files
        }
        
        for future in concurrent.futures.as_completed(futures):
            audio_file, success, info = future.result()
            filename = Path(audio_file).stem
            if success:
                success_count += 1
                print(f"✅ [{success_count + fail_count}/{len(todo_files)}] {filename} ({info:.1f}s)")
            else:
                fail_count += 1
                print(f"❌ [{success_count + fail_count}/{len(todo_files)}] {filename}: {info}")
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 60)
    print(f"✅ 完成！成功: {success_count}, 失败: {fail_count}")
    print(f"⏱️  总耗时: {total_time:.1f}s (平均 {total_time/len(todo_files):.1f}s/文件)")
    print(f"📁 结果保存到: {transcript_dir}")

if __name__ == "__main__":
    main()

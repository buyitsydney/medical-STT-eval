#!/bin/bash
# 渐进式 Speechmatics 评测脚本
# 每次运行增加 5 个新文件，不会重复！

set -e
cd "$(dirname "$0")"

# 加载环境变量
source ../MedPen/backend/local_env.sh 2>/dev/null || true

# 配置
BATCH_SIZE=${1:-5}  # 默认每批 5 个文件
TRANSCRIPT_DIR="results/transcripts/speechmatics-enhanced"
AUDIO_DIR="data/raw_audio"

# 需排除的问题文件
EXCLUDED_FILES="day1_consultation07 day3_consultation03"

echo "============================================================"
echo "🎯 Speechmatics 渐进式评测"
echo "============================================================"
echo ""

# 1. 获取已完成的文件
DONE_FILES=$(ls "$TRANSCRIPT_DIR" 2>/dev/null | sed 's/_transcript.txt//' | sort)
DONE_COUNT=$(echo "$DONE_FILES" | grep -c . || echo 0)

echo "📊 当前进度:"
echo "   已完成: $DONE_COUNT 个文件"

# 2. 获取所有音频文件（排除问题文件）
ALL_AUDIO=$(ls "$AUDIO_DIR"/*.wav | xargs -n1 basename | sed 's/.wav//' | sort)
TOTAL_COUNT=$(echo "$ALL_AUDIO" | wc -l | tr -d ' ')

# 过滤掉问题文件
for excl in $EXCLUDED_FILES; do
    ALL_AUDIO=$(echo "$ALL_AUDIO" | grep -v "$excl" || true)
done
VALID_COUNT=$(echo "$ALL_AUDIO" | grep -c . || echo 0)

echo "   总文件: $TOTAL_COUNT 个 (有效: $VALID_COUNT 个，排除 2 个问题文件)"

# 3. 找出待处理的文件
TODO_FILES=""
for audio in $ALL_AUDIO; do
    # 检查是否已完成
    if ! echo "$DONE_FILES" | grep -q "^${audio}$"; then
        TODO_FILES="$TODO_FILES $audio"
    fi
done
TODO_COUNT=$(echo $TODO_FILES | wc -w | tr -d ' ')

echo "   待处理: $TODO_COUNT 个文件"
echo ""

if [ "$TODO_COUNT" -eq 0 ]; then
    echo "✅ 所有文件已完成评测！"
    echo ""
    echo "运行以下命令更新指标:"
    echo "   python3 evaluate/metrics_generator.py --model_name speechmatics-enhanced"
    echo "   python3 evaluate/comparison_generator.py"
    exit 0
fi

# 4. 选取本批次要处理的文件
BATCH_FILES=$(echo $TODO_FILES | tr ' ' '\n' | head -n $BATCH_SIZE)
BATCH_COUNT=$(echo "$BATCH_FILES" | grep -c . || echo 0)

echo "📋 本批次将处理 $BATCH_COUNT 个文件:"
echo "$BATCH_FILES" | while read f; do echo "   - $f.wav"; done
echo ""

# 5. 确认执行
read -p "🚀 开始转写？(y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消"
    exit 1
fi

# 6. 逐个转写
echo ""
echo "============================================================"
echo "🎤 开始转写..."
echo "============================================================"

COUNT=0
for audio_base in $BATCH_FILES; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$BATCH_COUNT] 处理: $audio_base.wav"
    
    python3 transcribe/speechmatics_transcribe.py \
        --audio_dir "$AUDIO_DIR" \
        --pattern "${audio_base}.wav"
done

# 7. 更新指标
echo ""
echo "============================================================"
echo "📊 更新指标..."
echo "============================================================"
python3 evaluate/metrics_generator.py --model_name speechmatics-enhanced

echo ""
echo "============================================================"
echo "🏆 更新排行榜..."
echo "============================================================"
python3 evaluate/comparison_generator.py

# 8. 显示当前状态
NEW_DONE_COUNT=$(ls "$TRANSCRIPT_DIR" 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "============================================================"
echo "✅ 本批次完成！"
echo "============================================================"
echo "   已完成: $NEW_DONE_COUNT / $VALID_COUNT 个文件"
echo "   剩余: $((VALID_COUNT - NEW_DONE_COUNT)) 个文件"
echo ""
echo "💡 继续评测请再次运行: bash run_batch_eval.sh"

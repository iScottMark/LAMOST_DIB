#!/bin/bash
# This shell script is used to run the `main.py` script in parallel to measure the DIB.
# example: 
# ./multiple_tasks.sh 0 1000 42
# This means a total of 42 tasks are allocated for the spectra of 0-1000 index.

# 创建tmux会话
tmux new-session -d -s dib


# 获取当前终端的大小
terminal_width=$(tput cols)
terminal_height=$(tput lines)

# 计算每个窗格的宽度和高度
pane_width=$((terminal_width / 3))
pane_height=$((terminal_height / 4))

# 分割窗格
for i in {0..6}; do
    for j in {0..5}; do  # 修正此处的范围
        tmux split-window -h -l "$pane_width"
        tmux select-layout tiled
    done
done

# 获取当前窗口的窗格数量
# pane_count=$(tmux list-panes | wc -l)

# last_row_panes=$(tmux list-panes -F "#{pane_id}:#{pane_start_y}" | awk -F ':' '$2 == 2 { print $1 }')

# tmux kill-pane -t "$last_row_panes"


# 获取参数
start=$1
end=$2
segments=$3

# 计算每个段的大小
segment_size=$(( ($end - $start) / $segments ))

# 生成Python执行脚本
for ((i=0; i<$segments; i++))
do
    segment_start=$(( $start + ($i * $segment_size) ))
    segment_end=$(( $segment_start + $segment_size ))
    if [ $i -eq $(($segments-1)) ]; then
        segment_end=$end
    fi

   command="python ~/Project3/lamost_dib/main.py $segment_start $segment_end"
   # 构建 tmux 窗格名称
   tmux send-keys -t dib:0.$i "$command" Enter

    if [ $i -eq 20 ]; then
        # 进入tmux会话
        tmux attach-session -t dib
        for ((j=80; j>=0; j--))
        do
            echo "倒计时: $j 秒"
            sleep 1  # 等待 1 秒
        done
    fi
done

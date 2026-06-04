#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

save_root=${1:-'./results'}

# 指定可用的 GPU ID 列表（参数 2）
available_gpus=${2:-"0,1,2,3"}  # 默认 0,1,2,3，可以通过参数指定如 "0,2,4,6"

# General parameters
policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=${3:-0}
test_num=${4:-100}
start_port=29556

# 解析 GPU ID 列表
IFS=',' read -ra gpu_ids <<< "$available_gpus"
num_gpus=${#gpu_ids[@]}

echo "=========================================="
echo "Available GPUs: ${gpu_ids[*]} (Total: $num_gpus)"
echo "Save root: $save_root"
echo "=========================================="

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

# 展开所有任务到一个数组
all_tasks=()
for group in "${task_groups[@]}"; do
  read -r -a tasks <<< "$group"
  all_tasks+=("${tasks[@]}")
done

total_tasks=${#all_tasks[@]}
echo "Total tasks to execute: $total_tasks"
echo "Estimated parallel tasks: $num_gpus"
echo "=========================================="

log_dir="./logs"
mkdir -p "$log_dir"

batch_time=$(date +%Y%m%d_%H%M%S)

# 任务队列索引
current_task_idx=0
completed_tasks=0
failed_tasks=0

# GPU 状态数组：记录每个 GPU 是否空闲
declare -A gpu_status
for gpu_id in "${gpu_ids[@]}"; do
  gpu_status[$gpu_id]="idle"
done

# GPU 到端口的映射（基于 GPU 在列表中的索引）
declare -A gpu_to_port
for i in "${!gpu_ids[@]}"; do
  gpu_to_port[${gpu_ids[$i]}]=$((start_port + i))
done

# 进程到 GPU 的映射
declare -A pid_to_gpu
declare -A pid_to_task
declare -A pid_to_port
declare -A pid_to_logfile

# 函数：启动一个任务
launch_task() {
  local task_name=$1
  local gpu_id=$2
  local port=$3

  export CUDA_VISIBLE_DEVICES=${gpu_id}

  local log_file="${log_dir}/${task_name}_gpu${gpu_id}_${batch_time}.log"

  # 检查 save_root 下是否存在与 task_name 同名的文件夹
  local task_dir="${save_root}/stseed-10000/visualization/${task_name}"
  local actual_test_num=${test_num}

  if [ -d "$task_dir" ]; then
    # 统计文件夹中的文件数量
    local existing_files=$(find "$task_dir" -type f | wc -l)
    actual_test_num=$((50 - existing_files))

    if [ "$task_name" = "pick_dual_bottles" ] || [ "$task_name" = "handover_block" ]; then
      echo -e "MATCH"
      actual_test_num=$((50 - existing_files))
    fi

    # 确保 actual_test_num 不为负数
    if [ $actual_test_num -lt 0 ]; then
      actual_test_num=0
    fi

    echo -e "\033[34m[Check] Found ${existing_files} existing files in ${task_dir}, actual test_num: ${actual_test_num}\033[0m"
  else
    echo -e "\033[34m[Check] No existing directory at ${task_dir}, using default test_num: ${actual_test_num}\033[0m"
  fi

  # 如果 actual_test_num 为 0，跳过该任务
  if [ $actual_test_num -eq 0 ]; then
    echo -e "\033[32m[Skip] Task: ${task_name} - already completed (actual_test_num=0)\033[0m"
    return 1
  fi

  echo -e "\033[33m[Launch] Task: ${task_name}, GPU: ${gpu_id}, PORT: ${port}\033[0m"
  PYTHONUNBUFFERED=1 \
  PYTHONWARNINGS=ignore::UserWarning \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi \
    --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --save_root ${save_root} \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --test_num ${actual_test_num} \
    --port ${port} > "$log_file" 2>&1 &

  local pid=$!
  pid_to_gpu[$pid]=$gpu_id
  pid_to_task[$pid]=$task_name
  pid_to_port[$pid]=$port
  pid_to_logfile[$pid]=$log_file
  gpu_status[$gpu_id]="busy"

  echo "  PID: $pid, Log: $log_file"
}

# 函数：检查并获取空闲 GPU
get_idle_gpu() {
  for gpu_id in "${gpu_ids[@]}"; do
    if [[ "${gpu_status[$gpu_id]}" == "idle" ]]; then
      echo $gpu_id
      return
    fi
  done
  echo -1
}

# 函数：等待任意一个进程完成
wait_for_any_process() {
  local pids_to_check="$1"

  # 使用 wait -n 等待任意进程（bash 4.3+）
  if [[ "${BASH_VERSION%%.*}" -ge 5 ]] || [[ "${BASH_VERSION%%.*}" -eq 4 && "${BASH_VERSION#*.}" -ge 3 ]]; then
    wait -n $pids_to_check 2>/dev/null
    return $?
  else
    # 兼容旧版本 bash：轮询检查
    while true; do
      for pid in $pids_to_check; do
        if ! kill -0 $pid 2>/dev/null; then
          return 0
        fi
      done
      sleep 1
    done
  fi
}

# 函数：显示进度
show_progress() {
  local running=$(jobs -rp | wc -l)
  echo -e "\033[36m[Progress] Tasks: ${current_task_idx}/${total_tasks} | Running: ${running} | Completed: ${completed_tasks} | Failed: ${failed_tasks}\033[0m"
}

echo -e "\n\033[32m=== Starting Task Queue Execution ===\033[0m\n"

# 主循环
while [[ $current_task_idx -lt $total_tasks ]] || [[ $(jobs -rp | wc -l) -gt 0 ]]; do

  # 尝试为空闲 GPU 分配任务
  while [[ $current_task_idx -lt $total_tasks ]]; do
    idle_gpu=$(get_idle_gpu)

    if [[ $idle_gpu -eq -1 ]]; then
      break  # 没有空闲 GPU
    fi

    # 获取该 GPU 对应的端口
    port=${gpu_to_port[$idle_gpu]}

    # 获取下一个任务
    task_name="${all_tasks[$current_task_idx]}"

    # 启动任务
    launch_task "$task_name" "$idle_gpu" "$port"
    launch_result=$?

    ((current_task_idx++))

    # 如果任务被跳过 (launch_task 返回 1)，增加 completed_tasks
    if [[ $launch_result -eq 1 ]]; then
      ((completed_tasks++))
    fi
  done

  # 显示进度
  show_progress

  # 如果还有运行中的进程，等待完成
  if [[ $(jobs -rp | wc -l) -gt 0 ]]; then
    # 获取所有运行中的 PID
    running_pids=$(jobs -rp)
    # 等待任意一个进程完成
    wait_for_any_process "$running_pids"

    # 检查哪些进程已经完成
    for pid in "${!pid_to_gpu[@]}"; do
      if ! kill -0 $pid 2>/dev/null; then
        # 进程已结束
        gpu_id=${pid_to_gpu[$pid]}
        task_name=${pid_to_task[$pid]}
        port=${pid_to_port[$pid]}
        log_file=${pid_to_logfile[$pid]}

        # 检查退出状态（通过日志文件判断）
        if tail -20 "$log_file" | grep -q "Success rate:"; then
          echo -e "\033[32m[Done] Task: ${task_name}, GPU: ${gpu_id}, PID: ${pid}\033[0m"
          ((completed_tasks++))
        else
          echo -e "\033[31m[Fail] Task: ${task_name}, GPU: ${gpu_id}, PID: ${pid}\033[0m"
          ((failed_tasks++))
        fi

        # 标记 GPU 为空闲
        gpu_status[$gpu_id]="idle"

        # 清理映射
        unset pid_to_gpu[$pid]
        unset pid_to_task[$pid]
        unset pid_to_port[$pid]
        unset pid_to_logfile[$pid]
      fi
    done
  fi
done

echo -e "\n=========================================="
echo -e "\033[32m=== All Tasks Completed ===\033[0m"
echo "Total tasks: $total_tasks"
echo "Completed: $completed_tasks"
echo "Failed: $failed_tasks"
echo "GPU IDs used: ${gpu_ids[*]}"
echo "Save root: $save_root"
echo "=========================================="

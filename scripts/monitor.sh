#!/usr/bin/env bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/dma/clevrer-gen

start_time=$(date +%s)
baseline_rate=67  # vids/hr from 800 in 12 hrs at 32 samples

scenarios=("one_ball_nocollide" "two_balls_nocollide" "three_balls_nocollide" "four_balls_nocollide" "five_balls_nocollide")

# Snapshot baseline counts (videos that already existed before this run started)
declare -A baseline_counts
for scenario in "${scenarios[@]}"; do
  baseline_counts["$scenario"]=$(find "output/$scenario/videos" -name "*.mp4" 2>/dev/null | wc -l)
done
baseline_total=0
for scenario in "${scenarios[@]}"; do
  baseline_total=$((baseline_total + ${baseline_counts[$scenario]}))
done

echo "=== Monitoring: 5000-video run (16 samples, analytical flow) ==="
echo "Baseline: 67 vids/hr (800 vids in 12 hrs at 32 samples)"
echo "Videos already on disk at start: $baseline_total"
echo ""

iteration=0
last_total=0
last_time=$start_time

while true; do
  sleep 30
  iteration=$((iteration + 1))
  current_time=$(date +%s)
  elapsed_sec=$((current_time - start_time))
  elapsed_min=$((elapsed_sec / 60))

  # Count NEW .mp4 files as (current count - baseline count) per scenario
  total=0
  declare -A scenario_counts
  declare -A scenario_new
  for scenario in "${scenarios[@]}"; do
    count=$(find "output/$scenario/videos" -name "*.mp4" 2>/dev/null | wc -l)
    new=$((count - ${baseline_counts[$scenario]}))
    scenario_counts["$scenario"]=$count
    scenario_new["$scenario"]=$new
    total=$((total + new))
  done

  # Throughput
  time_delta=$((current_time - last_time))
  [ "$time_delta" -lt 60 ] && time_delta=60
  vids_delta=$((total - last_total))
  rate=$((vids_delta * 3600 / time_delta))  # vids/hr

  last_total=$total
  last_time=$current_time

  # ETA
  remaining=$((5000 - total))
  if [ "$rate" -gt 0 ]; then
    eta_sec=$((remaining * 3600 / rate))
    eta_hr=$((eta_sec / 3600))
    eta_min=$(((eta_sec % 3600) / 60))
  else
    eta_hr=0
    eta_min=0
  fi

  # Speedup factor
  if [ "$rate" -gt 0 ]; then
    speedup=$(awk "BEGIN {printf \"%.2f\", $rate / $baseline_rate}")
  else
    speedup="--"
  fi

  # Display
  clear
  echo "=== Progress Report (Iteration $iteration, ${elapsed_min}m elapsed) ==="
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
  echo "NEW Videos created this run: $total / 5000"
  echo "Throughput: $rate vids/hr (baseline: $baseline_rate vids/hr)"
  echo "Speedup: ${speedup}x"
  echo "ETA: ${eta_hr}h ${eta_min}m remaining"
  echo ""
  echo "Per-scenario (new / total on disk):"
  for scenario in "${scenarios[@]}"; do
    printf "  %-25s %4d new  (%4d total)\n" "$scenario:" "${scenario_new[$scenario]}" "${scenario_counts[$scenario]}"
  done

  if [ "$total" -ge 5000 ]; then
    echo ""
    echo "✓ DONE! Total time: ${elapsed_min}m"
    break
  fi
done

#!/bin/bash

# NCU Profiling 脚本 - 深度性能分析

echo "==================================="
echo "  NCU 性能分析"
echo "==================================="
echo ""

# 检查NCU是否安装
if ! command -v ncu &> /dev/null; then
    echo "错误: 找不到 ncu (Nsight Compute)"
    echo "请安装 NVIDIA Nsight Compute"
    exit 1
fi

NCU_VERSION=$(ncu --version | head -1)
echo "NCU 版本: $NCU_VERSION"
echo ""

# 确保测试程序已编译
if [ ! -f "./matmul_test" ]; then
    echo "错误: 找不到 matmul_test，请先运行 build_and_run.sh"
    exit 1
fi

OUTPUT_DIR="ncu_reports"
mkdir -p $OUTPUT_DIR

echo "性能分析模式:"
echo "1) 快速分析 (基础指标)"
echo "2) 完整分析 (所有指标)"
echo "3) 内存分析 (详细内存指标)"
echo "4) 计算分析 (Tensor Core & SM)"
echo "5) 瓶颈分析 (Roofline)"
echo ""
read -p "请选择 (1-5, 默认1): " MODE
MODE=${MODE:-1}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case $MODE in
    1)
        echo ""
        echo "运行快速分析..."
        ncu --set basic \
            --target-processes all \
            --kernel-name matmul_optimized_kernel \
            --launch-skip 10 \
            --launch-count 10 \
            -o $OUTPUT_DIR/matmul_basic_$TIMESTAMP \
            ./matmul_test
        ;;

    2)
        echo ""
        echo "运行完整分析 (这可能需要几分钟)..."
        ncu --set full \
            --target-processes all \
            --kernel-name matmul_optimized_kernel \
            --launch-skip 10 \
            --launch-count 1 \
            -o $OUTPUT_DIR/matmul_full_$TIMESTAMP \
            ./matmul_test
        ;;

    3)
        echo ""
        echo "运行内存分析..."
        ncu --set full \
            --target-processes all \
            --kernel-name matmul_optimized_kernel \
            --launch-skip 10 \
            --launch-count 5 \
            --metrics "dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum" \
            -o $OUTPUT_DIR/matmul_memory_$TIMESTAMP \
            ./matmul_test
        ;;

    4)
        echo ""
        echo "运行计算分析..."
        ncu --set full \
            --target-processes all \
            --kernel-name matmul_optimized_kernel \
            --launch-skip 10 \
            --launch-count 5 \
            --metrics "sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed_pipe_tensor.sum,\
smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_no_instruction.pct,\
smsp__average_warps_issue_stalled_wait.pct,\
smsp__average_warps_issue_stalled_barrier.pct,\
smsp__average_warps_issue_stalled_long_scoreboard.pct" \
            -o $OUTPUT_DIR/matmul_compute_$TIMESTAMP \
            ./matmul_test
        ;;

    5)
        echo ""
        echo "运行瓶颈分析..."
        ncu --set full \
            --target-processes all \
            --kernel-name matmul_optimized_kernel \
            --launch-skip 10 \
            --launch-count 5 \
            --metrics "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum" \
            -o $OUTPUT_DIR/matmul_roofline_$TIMESTAMP \
            ./matmul_test
        ;;

    *)
        echo "无效选择"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "  分析完成!"
    echo "==================================="
    echo ""
    echo "报告已保存到: $OUTPUT_DIR/"
    echo ""
    echo "查看报告:"
    echo "  ncu-ui $OUTPUT_DIR/matmul_*_$TIMESTAMP.ncu-rep"
    echo ""
    echo "或导出为文本:"
    echo "  ncu --import $OUTPUT_DIR/matmul_*_$TIMESTAMP.ncu-rep --page details > report.txt"
else
    echo ""
    echo "分析失败!"
    exit 1
fi

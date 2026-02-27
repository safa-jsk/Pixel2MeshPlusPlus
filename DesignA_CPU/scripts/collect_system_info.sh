#!/bin/bash
# Collect system information for Design A baseline documentation

OUTPUT_FILE="../outputs/designA/benchmark/system_info.txt"
mkdir -p ../outputs/designA/benchmark

echo "Design A Baseline - System Information" > $OUTPUT_FILE
echo "=======================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Collection Date: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "Environment:" >> $OUTPUT_FILE
echo "-------------" >> $OUTPUT_FILE
echo "Container: p2mpp:cpu (Docker)" >> $OUTPUT_FILE
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '\"')" >> $OUTPUT_FILE
echo "Kernel: $(uname -r)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "Python & TensorFlow:" >> $OUTPUT_FILE
echo "--------------------" >> $OUTPUT_FILE
python -c "import sys; print('Python: {}'.format(sys.version.split()[0]))" >> $OUTPUT_FILE
python -c "import tensorflow as tf; print('TensorFlow: {}'.format(tf.__version__))" >> $OUTPUT_FILE
python -c "import numpy as np; print('NumPy: {}'.format(np.__version__))" >> $OUTPUT_FILE
python -c "import cv2; print('OpenCV: {}'.format(cv2.__version__))" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "CPU Information:" >> $OUTPUT_FILE
echo "----------------" >> $OUTPUT_FILE
lscpu | grep "Model name" >> $OUTPUT_FILE
lscpu | grep "CPU(s):" | head -1 >> $OUTPUT_FILE
lscpu | grep "Thread(s) per core" >> $OUTPUT_FILE
lscpu | grep "Core(s) per socket" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "Memory:" >> $OUTPUT_FILE
echo "--------" >> $OUTPUT_FILE
free -h | grep "Mem:" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "GPU (if available):" >> $OUTPUT_FILE
echo "-------------------" >> $OUTPUT_FILE
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >> $OUTPUT_FILE
else
    echo "No NVIDIA GPU detected (CPU-only evaluation)" >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

echo "Evaluation Configuration:" >> $OUTPUT_FILE
echo "-------------------------" >> $OUTPUT_FILE
echo "Model: Pixel2Mesh++ (2-stage)" >> $OUTPUT_FILE
echo "Stage 1: Coarse MVP2M (checkpoint: 50)" >> $OUTPUT_FILE
echo "Stage 2: Refined P2MPP (checkpoint: 10)" >> $OUTPUT_FILE
echo "Eval samples: $(wc -l < designA_eval_list.txt)" >> $OUTPUT_FILE
echo "Categories: 6 (Airplane, Car, Chair, Table, Loudspeaker, Lamp)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

cat $OUTPUT_FILE
echo ""
echo "System info saved to: $OUTPUT_FILE"

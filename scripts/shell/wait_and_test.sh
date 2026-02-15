#!/bin/bash
echo "Waiting for Docker build to complete..."
while ps aux | grep -q "[d]ocker build"; do
    sleep 10
done

echo "✓ Build complete! Verifying image..."
docker images | grep p2mpp-gpu

echo ""
echo "Testing single sample inference with TF 2.6.0 + CUDA 11.2..."
docker run --rm --gpus all -v "$(pwd)":/app -w /app p2mpp-gpu:latest bash -c "
head -1 data/designB_eval_full.txt > /tmp/test_single.txt
timeout 90 python3 test_p2mpp.py \
  -f cfgs/p2mpp.yaml \
  --restore 1 \
  --test_file_path /tmp/test_single.txt \
  --test_data_path data/designA_subset/p2mppdata/test \
  --test_image_path data/designA_subset/ShapeNetRendering/rendering_only \
  --test_mesh_root outputs/designA/eval_meshes \
  2>&1 | tail -40
"

echo ""
echo "If successful, run full benchmark with: bash env/gpu/benchmark.sh"

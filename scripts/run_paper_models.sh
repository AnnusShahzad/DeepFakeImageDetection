#!/usr/bin/env bash
# MS-COCOAI — exact commands for the paper's models (nothing magic).
# Run from the repo root. Dataset downloads from Hugging Face on first use
# (needs internet enabled on Kaggle / lab PC).
set -euo pipefail

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/data/hf_cache}"
mkdir -p "$HF_DATASETS_CACHE" outputs reports

echo "== Task A: binary detection (AdamW lr=1e-4, 30 epochs, patience 5, eff. batch 64)"
python -m mscocoai.train --task binary --backbone resnet50       --out-dir outputs/resnet50_binary --resume auto --balance-loss
python -m mscocoai.train --task binary --backbone efficientnet_b0 --out-dir outputs/effb0_binary   --resume auto --balance-loss

echo "== Task B: 6-class attribution (class-weighted CE; paper protocol)"
python -m mscocoai.train --task attribution --backbone resnet50        --out-dir outputs/resnet50_attr --resume auto --attr-class-weights
python -m mscocoai.train --task attribution --backbone efficientnet_b0 --out-dir outputs/effb0_attr   --resume auto --attr-class-weights

echo "== Task A/B: frozen CLIP ViT-B/16 + linear probe (UniFD-style)"
python -m mscocoai.clip_probe --mode extract --out-dir outputs/clip_vitb16
python -m mscocoai.clip_probe --mode train   --task binary      --out-dir outputs/clip_vitb16 --balance-loss
python -m mscocoai.clip_probe --mode train   --task attribution --out-dir outputs/clip_vitb16 --attr-class-weights
python -m mscocoai.clip_probe --mode eval    --task binary      --out-dir outputs/clip_vitb16 --report-dir reports/clip_binary_test
python -m mscocoai.clip_probe --mode eval    --task attribution --out-dir outputs/clip_vitb16 --report-dir reports/clip_attr_test

echo "== Task A+B: multi-task EfficientNet-B0 (lambda_d = lambda_a = 1)"
python -m mscocoai.train --task multitask --backbone efficientnet_b0 --out-dir outputs/mt_effb0 --resume auto --balance-loss --attr-class-weights

echo "== Final test-split reports (test is ONLY read here)"
python -m mscocoai.evaluate --checkpoint outputs/resnet50_binary/best.pt      --split test --report-dir reports/resnet50_binary_test
python -m mscocoai.evaluate --checkpoint outputs/effb0_binary/best.pt         --split test --report-dir reports/effb0_binary_test
python -m mscocoai.evaluate --checkpoint outputs/resnet50_attr/best.pt        --split test --report-dir reports/resnet50_attr_test
python -m mscocoai.evaluate --checkpoint outputs/effb0_attr/best.pt           --split test --report-dir reports/effb0_attr_test
python -m mscocoai.evaluate --checkpoint outputs/mt_effb0/best.pt             --split test --report-dir reports/mt_effb0_test

echo "== Robustness probes (JPEG q75 / Gaussian blur)"
python -m mscocoai.evaluate --checkpoint outputs/resnet50_binary/best.pt --split test --degradation jpeg75 --report-dir reports/resnet50_binary_jpeg75 --no-samples
python -m mscocoai.evaluate --checkpoint outputs/resnet50_binary/best.pt --split test --degradation blur   --report-dir reports/resnet50_binary_blur   --no-samples

echo "== SOTA levers (optional; disclose in your writeup)"
echo "   EMA:          add --ema-decay 0.999"
echo "   EffNet-B4:    --backbone efficientnet_b4"
echo "   ConvNeXtV2:   --backbone convnextv2_base"
echo "   CLIP unfreeze --train-mode last2 (or full) with --unfreeze-lr 1e-5"
echo "   Ensemble:     python -m mscocoai.ensemble --task binary --checkpoints outputs/effb0_binary/best.pt outputs/resnet50_binary/best.pt --split test --report-dir reports/ensemble_binary"

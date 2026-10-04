# E7：逐图配对统计验证

E7 不训练新模型。使用 E6 的三个**独立训练** seed (`10 11 12`) 的 RGB baseline 与 UDR-v2 final checkpoint，在原五数据集上复跑相同的分块推理、图像排序、hard-Gumbel seed、uint8 Y 通道 ×4/crop 4 指标，并保存每一张图的配对结果。每条 CSV 行严格包含附件要求的 `dataset,image_id,seed,baseline_psnr,udrv2_psnr,delta_psnr,baseline_ssim,udrv2_ssim,delta_ssim`。任何图像或 seed 无法一一配对，程序立即报错。

逐图结果会再次与 E6 的各数据集/seed 聚合 PSNR 和 SSIM 对账（绝对容差 `1e-6`），以防 E7 误用不同权重、数据顺序或随机路由。E6 checkpoint SHA256 和逐 seed E2/E3→E4 合并来源也重新验证。没有 E6 真正训练和测试的权重/报告时，E7 不会生成伪造的统计结果。

## 10,000 次配对 bootstrap

对每个数据集，以**图像 ID**为有放回抽样单位，重复 10,000 次。抽到一张图时保留该图三个 seed 的完整 RGB/UDR-v2 配对差，先求这张图跨 seed 的平均 `Δ_i`，再求本次样本的平均。这样一张图的三次观测不会被错误当成三张独立图。固定 bootstrap RNG seed=`2026`，取 bootstrap 平均分布的 2.5% 与 97.5% 分位数，得到逐数据集 95% 区间。该区间**条件于已观察的三个训练 seed**；初始化/训练波动另见 E6 的 seed 间标准差，不应只用 E7 区间替代它。

另输出逐数据集的正增益图像比例（按图像跨 seed 的平均 `Δ_i>0`）、中位数、P25/P75、最差/最佳最多十张图及每张图的三 seed 差值。区间下界 `>0` 标记较强正增益统计证据；若均值为正但区间跨零，只称平均正向趋势，不称稳定改善。

## 命令

如果 E6 已在单独仓库完成，直接在 E7 克隆目录读取其绝对路径：

```bash
git clone https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E7.git
cd v3.2-M3SR-MambaIRv2-E7
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
E6ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E6
test -f "$E6ROOT/experiments/E6_plan/E6_plan.json"
test -f "$E6ROOT/results/E6_multiseed/E6_multiseed.json"
python scripts/udr/e7_image_statistics.py --self-test
python scripts/udr/check_e7_image_statistics.py
CUDA_VISIBLE_DEVICES=0 python scripts/udr/e7_image_statistics.py \
  --plan "$E6ROOT/experiments/E6_plan/E6_plan.json" \
  --e6-report "$E6ROOT/results/E6_multiseed/E6_multiseed.json" \
  --bootstrap-seed 2026 \
  --output results/E7_image_statistics
```

若 E6 尚未运行，可在 E7 项目中按 [E6 指南](E6_GUIDE.md)先生成 E4 选择、三 seed 独立训练计划并完成训练和 E6 正式测试；随后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/e7_image_statistics.py \
  --plan experiments/E6_plan/E6_plan.json \
  --e6-report results/E6_multiseed/E6_multiseed.json \
  --bootstrap-seed 2026 \
  --output results/E7_image_statistics
```

输出 `E7_per_image.csv`、`E7_statistics.json`、`E7_summary.md`，以及供审计的逐模型/seed 测试 YAML 与原始逐图 JSON。报告只使用实测逐图值，不从 E6 数据集平均值反推图像结果。

# 实验运行说明

本仓库的正式实验入口是 `run_experiment.py`。它基于现有 `dataset.py`、`model.py`、`unlearn.py`、`metrics.py` 和 `utils.py`，完成以下流程：

1. 训练或加载原始分类模型；
2. 构造 forget / retain 数据划分；
3. 运行 Amnesiac Unlearning；
4. 运行 Bad Teacher / Blindspot Unlearning；
5. 运行 Retain-Aware Bad Teacher；
6. 输出 Forget Acc、Retain Acc、MIA、ZRF；
7. 保存 `unlearning_results.csv`、`fig_accuracy_comparison.png`、`fig_privacy_metrics.png`。

## 环境

当前本机已验证：

- Python: `E:\Program Files\python\python.exe`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- torch: `2.8.0+cu126`
- torchvision: `0.23.0+cu126`
- CUDA available: `True`

如果换环境，先安装：

```powershell
python -m pip install -r requirements.txt
```

有 NVIDIA GPU 时，建议安装 CUDA 版 torch/torchvision。当前本机使用过的安装方式是：

```powershell
python -m pip install --force-reinstall torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

如果官方 PyTorch 源下载过慢，可以先下载 wheel 到本地，再用：

```powershell
python -m pip install --force-reinstall --no-deps .\wheels\torch-2.8.0+cu126-cp313-cp313-win_amd64.whl .\wheels\torchvision-0.23.0+cu126-cp313-cp313-win_amd64.whl
```

## 本地流程验证

下面命令使用随机小数据和 TinyCNN，只验证代码链路，不作为报告正式结果：

```powershell
python run_experiment.py --synthetic --quick --skip-gold --out-dir outputs\smoke --force-retrain
```

我已在本机验证该命令可以跑通，并能生成结果 CSV 和两张图。

GPU 链路验证命令：

```powershell
python run_experiment.py --synthetic --quick --skip-gold --out-dir outputs\gpu_smoke --force-retrain
```

本机验证输出包含 `device=cuda`。

## 正式实验

有 GPU 时推荐运行：

```powershell
python run_experiment.py --pretrained --out-dir outputs\full --batch-size 256 --base-epochs 5 --gold-epochs 5 --amnesiac-epochs 3 --badteacher-epochs 1 --force-retrain
```

如果只想先跳过 Gold Model 节省时间：

```powershell
python run_experiment.py --pretrained --skip-gold --out-dir outputs\full_no_gold --batch-size 256 --base-epochs 5 --amnesiac-epochs 3 --badteacher-epochs 1 --force-retrain
```

当前机器已能识别 RTX 4060 并使用 CUDA。完整实验的主要外部依赖是 CIFAR-100 数据集下载；如果官方源下载不稳定，可以先手动准备 `data\cifar-100-python.tar.gz`，再运行上面的正式实验命令。

## 当前可用于汇报的中等规模结果

已保存到 `results/medium_no_gold/`。

运行命令：

```powershell
python run_experiment.py --skip-gold --out-dir outputs\medium_no_gold --batch-size 128 --base-epochs 3 --amnesiac-epochs 2 --badteacher-epochs 1 --max-forget-train 500 --max-retain-train 5000 --max-forget-valid 100 --max-retain-valid 2000 --force-retrain
```

核心结果：

| Method | Forget Acc ↓ | Retain Acc ↑ | MIA ↓ | ZRF ↑ |
| --- | ---: | ---: | ---: | ---: |
| Original | 78.00 | 15.07 | 0.000 | 0.494 |
| Amnesiac | 3.00 | 24.04 | 0.012 | 0.700 |
| Bad Teacher | 68.00 | 15.92 | 0.000 | 0.762 |
| Bad Teacher + Retain (Ours) | 56.00 | 20.31 | 0.000 | 0.734 |

当前汇报重点建议放在 `Bad Teacher` 与 `Bad Teacher + Retain (Ours)` 的对比：加入 retain-set 约束后，Forget Acc 从 `68.00%` 降到 `56.00%`，Retain Acc 从 `15.92%` 提升到 `20.31%`，说明改进方法在增强目标类遗忘的同时，也更好地保留了非遗忘知识。

## 当前主结果：完整 no-gold 实验

已保存到 `results/full_no_gold/`。

运行命令：

```powershell
python run_experiment.py --skip-gold --out-dir outputs\full_no_gold --batch-size 256 --base-epochs 5 --amnesiac-epochs 3 --badteacher-epochs 1 --force-retrain
```

核心结果：

| Method | Forget Acc ↓ | Retain Acc ↑ | MIA ↓ | ZRF ↑ |
| --- | ---: | ---: | ---: | ---: |
| Original | 55.00 | 47.14 | 1.000 | 0.492 |
| Amnesiac | 7.00 | 60.02 | 1.000 | 0.628 |
| Bad Teacher | 22.00 | 47.67 | 1.000 | 0.745 |
| Bad Teacher + Retain (Ours) | 23.00 | 54.25 | 1.000 | 0.702 |

该结果更适合作为汇报主结果：`Bad Teacher + Retain (Ours)` 与 `Bad Teacher` 保持接近的遗忘效果，Forget Acc 分别为 `23.00%` 和 `22.00%`；同时 Retain Acc 从 `47.67%` 提升到 `54.25%`。因此这版结果应表述为“在基本维持遗忘效果的同时提升非遗忘知识保留能力”，不要表述为“遗忘更强”。

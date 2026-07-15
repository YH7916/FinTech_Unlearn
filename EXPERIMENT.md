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
- torch: `2.8.0+cpu`
- torchvision: `0.23.0+cpu`

如果换环境，先安装：

```powershell
python -m pip install -r requirements.txt
```

有 NVIDIA GPU 时，建议按 PyTorch 官网安装 CUDA 版 torch/torchvision，再运行正式实验。

## 本地流程验证

下面命令使用随机小数据和 TinyCNN，只验证代码链路，不作为报告正式结果：

```powershell
python run_experiment.py --synthetic --quick --skip-gold --out-dir outputs\smoke --force-retrain
```

我已在本机验证该命令可以跑通，并能生成结果 CSV 和两张图。

## 正式实验

有 GPU 时推荐运行：

```powershell
python run_experiment.py --pretrained --out-dir outputs\full --batch-size 256 --base-epochs 5 --gold-epochs 5 --amnesiac-epochs 3 --badteacher-epochs 1 --force-retrain
```

如果只想先跳过 Gold Model 节省时间：

```powershell
python run_experiment.py --pretrained --skip-gold --out-dir outputs\full_no_gold --batch-size 256 --base-epochs 5 --amnesiac-epochs 3 --badteacher-epochs 1 --force-retrain
```

当前机器只有 CPU，完整 ResNet18 + CIFAR100 训练会非常慢；建议把正式实验放到 GPU 环境跑。CPU 环境可以用 `--quick` 或调小 `--max-retain-train` 等参数检查流程。


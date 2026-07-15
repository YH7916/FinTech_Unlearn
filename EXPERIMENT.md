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

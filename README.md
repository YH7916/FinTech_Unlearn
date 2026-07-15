# FinTech_Unlearn

《金融科技导论与实践》隐私保护方向大作业：面向图像分类模型的机器逆学习。

本仓库围绕 CIFAR100 Super20 图像分类任务，实现和整理机器逆学习实验：

- `unlearn.py`：Bad Teacher / Blindspot Unlearning 复现，以及 Retain-Aware Bad Teacher 改进方法。
- `metrics.py`：成员推断攻击 MIA、零重训遗忘分数 ZRF 等评估指标。
- `run_experiment.py`：训练、逆学习和统一评估流程，支持真实 CIFAR100 实验和 synthetic smoke test。
- `unlearn_and_eval.ipynb`：稳定 notebook 入口，调用 `run_experiment.py`。
- `实验报告.md`：最终报告源稿。
- `EXPERIMENT.md`：实验运行说明。
- `presentation/`：基于 `机器逆学习_pre.pdf` 的课堂汇报材料。

明天课堂汇报优先使用：

- `presentation/slides/机器逆学习_pre.pptx`
- `presentation/source/机器逆学习_pre.pdf`

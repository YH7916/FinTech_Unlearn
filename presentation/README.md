# 隐私保护机器逆学习课堂汇报

本目录用于明天课堂 6-7 分钟汇报，方向统一写作“隐私保护”，子方向选择“针对图像分类模型的机器逆学习”。

汇报题目：

面向图像分类模型的机器逆学习：方法复现、隐私评估与改进

## 明天直接使用

- `slides/隐私保护_机器逆学习_课堂汇报.pptx`：课堂汇报 PPT。
- `docs/02_speaker_script_6min.md`：逐页讲稿，按 6-7 分钟准备。
- `docs/01_research_proposal.md`：课堂上可讲的研究设计稿。
- `docs/04_report_outline_10pages.md`：最终 10 页以上报告骨架。

## 选题结论

选择隐私保护方向下的机器逆学习任务。理由是：仓库已经围绕 CIFAR100 Super20、ResNet18、Amnesiac、Bad Teacher、MIA/ZRF 指标和 Retain-Aware Bad Teacher 改进形成了完整主线。明天课堂可以讲清楚“为什么要遗忘、如何遗忘、如何证明真的遗忘、改进点在哪里”。

## 后续实验主线

1. 跑通 `unlearn_and_eval.ipynb` 中 Amnesiac Unlearning。
2. 使用 `unlearn.py` 中已补全的 `blindspot_unlearner` 复现 Bad Teacher。
3. 使用 `metrics.py` 中的 MIA 与 ZRF 评价遗忘是否充分。
4. 对比 Retain-Aware Bad Teacher 与普通 Bad Teacher 的遗忘/保留折中。
5. 将 notebook 输出的表格和图片填入 `实验报告.md`。

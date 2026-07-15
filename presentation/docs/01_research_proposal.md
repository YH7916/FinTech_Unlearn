# 研究设计稿

## 研究主题

面向图像分类模型的机器逆学习：方法复现、隐私评估与改进。

所属方向：隐私保护。

具体任务：针对图像分类模型的机器逆学习。以 CIFAR100 的 Super20 分类任务为基础，从已经训练好的 ResNet18 模型中删除指定类别数据的影响，并尽量保持其余类别的分类能力。仓库当前主线是复现 Amnesiac 与 Bad Teacher，引入 MIA/ZRF 指标，并提出 Retain-Aware Bad Teacher 改进。

## 背景与意义

金融科技系统会持续使用用户行为、交易、画像、风险偏好等敏感数据训练模型。现实中可能出现两类需求：用户撤回授权后要求删除数据影响，或者训练集中发现错误、过期、违规数据后需要修正模型。简单删除数据库记录不能保证模型已经“忘记”这些样本，因为模型参数中可能仍然保留了训练数据的统计痕迹。

机器逆学习要解决的问题是：在不完全重训模型的情况下，尽量消除指定数据对模型的影响，同时保留其他数据上的性能。对金融场景来说，这对应模型合规、隐私保护、数据治理和模型审计。

## 国内外研究与产业现状概况

现有机器逆学习方法大致有三类：

1. 完全重训：从训练集中删除目标数据后重新训练模型，效果最可靠，但成本最高，可作为 Gold model 对照。
2. 参数或梯度修正：记录训练过程中的更新信息，删除数据时反向抵消相关影响，Amnesiac Machine Learning 属于这一类思路。
3. 蒸馏式逆学习：使用教师模型引导学生模型。Bad Teacher 方法让学生模型在保留数据上模仿原模型，在遗忘数据上模仿“无效教师”，从而降低对遗忘数据的依赖。

课程资料中的 `unlearn` 代码已经覆盖了第三类方法的实现入口，并要求补全 Blindspot/Bad Teacher 的核心训练逻辑。

## 研究思路

我准备在 Bad Teacher 框架上做一个轻量改进：Retain-Aware Bad Teacher。

核心想法是：普通 Bad Teacher 在保留数据上主要模仿原教师的软标签，但当遗忘类别和保留类别特征纠缠时，纯蒸馏可能损伤保留集准确率。因此在原有双教师蒸馏损失之外，对 retain set 增加监督交叉熵锚定项，用真实标签稳定学生模型在保留数据上的决策边界。

目标函数由两部分组成：

- 遗忘损失：让学生模型在 forget set 上接近 incompetent teacher。
- 保留蒸馏损失：让学生模型在 retain set 上接近 full-trained teacher。
- 保留监督损失：让学生模型在 retain set 上继续拟合真实标签。

总损失为：`L_total = L_KD + λ * CE(student(retain), y_true)`。

## 具体步骤

1. 复现实验环境：使用资料中的 CIFAR100 Super20 数据、ResNet18 模型和 `unlearn_and_eval.ipynb`。
2. 跑通 baseline：记录原模型、Gold retrain、Amnesiac Unlearning 的 forget accuracy 和 retain accuracy。
3. 复现 Bad Teacher：使用 `unlearn.py` 的 `blindspot_unlearner` 运行双教师蒸馏。
4. 扩展指标：使用 `metrics.py` 中的成员推断攻击 MIA 与零重训遗忘分数 ZRF。
5. 实现改进方法：使用 `blindspot_unlearner_retain_aware` 加入 retain set 监督锚定项。
6. 对比分析：比较五个模型在 forget set、retain set、整体测试集、分布差异和运行时间上的表现。

## 实验与分析设计

实验对象：

- Original model：使用全部数据训练的模型。
- Gold model：删除 forget data 后重新训练的模型。
- Amnesiac：课程 notebook 中已有 baseline。
- Bad Teacher：补全 `blindspot_unlearner` 后得到。
- Ours：加入置信度自适应权重的 Bad Teacher。

评价指标：

- Forget accuracy：越接近 Gold model 越好，不追求简单降到 0。
- Retain accuracy：越高越好，反映模型是否保留非遗忘类别能力。
- JS divergence：比较输出分布与 Gold model 的接近程度。
- Entropy：观察遗忘集合上的输出是否不再过度自信。
- Time cost：比较逆学习和完全重训的成本差异。

预期结论：

改进方法应在保持 retain accuracy 的前提下，使 forget set 上的表现更接近 Gold model，并在 JS divergence 或 entropy 上体现出更稳定的遗忘效果。

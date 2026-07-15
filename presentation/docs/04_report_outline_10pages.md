# 最终报告 10 页以上骨架

## 1. 研究主题

题目：面向图像分类模型的机器逆学习：方法复现、隐私评估与改进。

方向：隐私保护。

研究对象：针对图像分类模型的机器逆学习。

## 2. 研究背景与意义

可写 1.5-2 页：

- 金融科技模型对敏感数据的依赖。
- 数据撤回、合规删除、错误数据修复的现实需求。
- 删除原始数据与删除模型影响的区别。
- 机器逆学习对隐私保护、模型审计、数据治理的意义。

## 3. 国内外研究现状与产业现状

可写 2 页：

- 完全重训方法。
- Amnesiac Machine Learning。
- Bad Teacher / Blindspot unlearning。
- 评价指标：准确率、分布距离、成员推断攻击、运行成本。
- 金融场景中的合规删除和可审计模型更新。

## 4. 研究思路与方法

可写 2 页：

- 任务定义：forget set、retain set、original model、gold model、student model。
- Bad Teacher 框架。
- Retain-Aware Bad Teacher：在保留数据上加入监督锚定项。
- 损失函数：遗忘损失 + 保留损失。
- 方法优势与可能局限。

## 5. 实验设计

可写 1.5-2 页：

- 数据集：CIFAR100 Super20。
- 模型：ResNet18。
- 对比方法：Original、Gold retrain、Amnesiac、Bad Teacher、Ours。
- 参数设置：epoch、batch size、learning rate、temperature、forget class。
- 指标：forget accuracy、retain accuracy、MIA、ZRF、time cost。

## 6. 实验结果与分析

可写 2-3 页：

- 总体结果表。
- forget set 与 retain set 的权衡分析。
- 与 Gold model 的距离分析。
- 参数敏感性分析。
- 运行成本分析。
- 失败案例或局限性讨论。

## 7. 结论

可写 0.5-1 页：

- 总结本文完成了 Amnesiac/Bad Teacher 复现、MIA/ZRF 指标扩展和 Retain-Aware Bad Teacher 改进。
- 说明改进方法在哪些指标上有效。
- 说明后续可扩展到金融风控、智能投顾、推荐系统等更贴近金融业务的数据。

## 8. 参考文献

参考 `docs/references.md`。

# 实验计划

## 数据与模型

- 数据集：CIFAR100，按资料代码映射到 Super20 分类。
- 模型：ResNet18，输出 20 个 super class。
- 遗忘目标：资料 notebook 中默认遗忘 CIFAR100 fine class 69，对应 rocket。
- 设备：优先使用 CUDA；如果本机没有 GPU，需要把 notebook 中硬编码的 `device = 'cuda'` 改为自动选择设备。

## 当前代码注意点

- `unlearn.py` 中 `blindspot_unlearner` 仍有 TODO，需要补训练循环。
- `metrics.py` 目前只有 `JSDiv` 和 `entropy`，还没有完整集成评估流程。
- notebook 中多处写死 `num_workers=32`，Windows 下可能过高，建议改成 0、2 或 4。
- notebook 中部分 cell 写死 `map_location='cuda'`，CPU 环境会报错。

## 模型对比表

| 模型 | 作用 | 是否必须 |
| --- | --- | --- |
| Original | 全量训练模型，作为逆学习起点 | 必须 |
| Gold retrain | 删除 forget set 后重训，作为理想参照 | 建议 |
| Amnesiac | 课程要求 baseline | 必须 |
| Bad Teacher | 课程要求复现方法 | 必须 |
| Retain-Aware Bad Teacher | 本文改进方法 | 必须 |

## 指标

| 指标 | 含义 | 预期方向 |
| --- | --- | --- |
| Forget accuracy | 遗忘集合上的准确率 | 接近 Gold model |
| Retain accuracy | 保留集合上的准确率 | 尽量接近 Original |
| MIA | 成员推断攻击将 forget 样本判为训练成员的比例 | 越低越好 |
| ZRF | 与随机初始化教师在 forget set 上的行为接近度 | 越高越好 |
| Time cost | 训练或逆学习耗时 | 低于完全重训 |

## 最小可交付实验

1. 跑出 Original、Amnesiac、Bad Teacher、Ours 四组结果。
2. 如果完全重训成本太高，可以用较少 epoch 的 Gold retrain 作为近似参照，并在报告中说明。
3. 表格至少包含 forget accuracy、retain accuracy、MIA、ZRF。
4. 图表至少包含准确率对比图和隐私/遗忘指标对比图。

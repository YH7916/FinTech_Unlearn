#import "lib.typ": *

// 仅保留字体（思源宋体）与垂直居中，其余一律用模板自带样式
#set text(font: ("Source Han Serif", "STSong", "SimSun"))

#show: slides.with(
  title: "图像分类模型的机器逆学习：复现、评估与保留感知改进",
  subtitle: "开题答辩",
  date: "2026年8月",
  authors: ("组员：___ · ___"),
  title-color: rgb("#20456e"),
  ratio: 16 / 9,
  layout: "medium",
  toc: true,
  count: "dot",
  footer: true,
  theme: "normal",
)

#set align(horizon)

// ================================================================
= 研究背景与目标

== 研究背景与动机

- 合规与安全要求：从已训练模型中*移除某些数据的影响*（"被遗忘权"、修复被污染/错标的数据）；
- 但*删数据 ≠ 删影响*，而从头重训代价太高 → 催生"近似逆学习"；
- 现有方法多数只用*遗忘集准确率*判断成败，未必真忘（Hayes 2024）。

/ 本文目标: 让模型可靠地忘掉一个指定类别、保住其余类别，并能*验证*它真的忘了。

== 研究内容：四项任务

围绕指南的四项任务展开：

#table(
  columns: (auto, 1fr),
  [*任务*], [*内容*],
  [一 · 理解], [运行 Amnesiac，调超参分析遗忘影响],
  [二 · 复现], [复现 Bad Teacher，补全其蒸馏训练核心],
  [三 · 评估], [在准确率外引入 MIA、ZRF 两个指标并分析],
  [四 · 改进], [提出"保留感知"改进，并对照验证],
)

== 相关工作与实验设定

/ 相关工作: 本文沿*蒸馏路线*（Bad Teacher、SCRUB），参照近年评估研究（Hayes 2024、Deep Unlearn 2024）补隐私指标；另有 SISA、SalUn 等路线可参考。

#table(
  columns: (auto, 1fr),
  [数据 / 模型], [CIFAR-100（20 超类）· ResNet-18（预训练）],
  [遗忘目标], [细类"火箭"（1 个类别）],
  [数据划分], [forget（要忘）/ retain（要留）],
  [金标准], [Gold Model：仅用保留数据重训的"理想遗忘"],
)

// ================================================================
= 任务一 · 理解 Amnesiac

== Amnesiac Unlearning（基线）

/ 原理: 遗忘样本重标为*随机错误标签*后微调，覆盖原有映射。

- 定位：最简基线；
- 调节*学习率 / 批大小*，分析对"遗忘—保留"折中的影响。

// ================================================================
= 任务二 · 复现 Bad Teacher

== 核心方法：双教师蒸馏

#table(
  columns: (auto, 1fr),
  [*有能教师*（原模型）], [保留样本 → 正确软标签 → 维持知识],
  [*无能教师*（随机网络）], [遗忘样本 → 随机软标签 → 定向遗忘],
)

学生按遗忘标记选择模仿对象，经带温度 KL 散度蒸馏（$T$ 温度，$ell in {0,1}$）：

$ p_"tgt" = ell dot "softmax"(z_"无能"/T) + (1-ell) dot "softmax"(z_"有能"/T) $

$ quad cal(L) = "KL"("logsoftmax"(z_"学生"/T) || p_"tgt") $

== 我们补全的核心：蒸馏训练循环

`blindspot_unlearner` 里"模型更新 + 损失计算"这段是空的，由我们补全：

```python
for x, y in loader:                  # y=1 遗忘样本, y=0 保留样本
    with torch.no_grad():            # ① 两位老师冻结，只给参考
        p_good = softmax(good_teacher(x) / T)   # 有能 = 原模型
        p_bad  = softmax(bad_teacher(x)  / T)   # 无能 = 随机网络
    logit_s  = student(x)            # ② 学生前向
    p_target = y * p_bad + (1 - y) * p_good     # ③ 逐样本挑老师
    loss = KL(log_softmax(logit_s / T), p_target)
    loss.backward(); opt.step(); opt.zero_grad()  # ④ 只更新学生
```

+ 两位老师用 `no_grad` 冻结：只提供目标分布，本身不训练；
+ *关键在第 ③ 步*：按标记 `y` 逐样本选目标 —— 遗忘样本学"无能"、保留样本学"有能"，温度 `T` 软化分布、传递更多信息；
+ 用 KL 散度把学生拉向目标分布，反传只更新学生。

// ================================================================
= 任务三 · 完善评估指标

== 为什么还要两个额外指标

准确率只说明模型"答错了"，不代表它"没记住"—— 它可能只是嘴上不认、心里还记得。于是从两个角度补两把"尺子"：

/ MIA（成员推断攻击）: 站在攻击者视角，判断某条数据*是否被训练过*。模型对训练过的样本往往更自信；若攻击者已分辨不出，说明遗忘到位。#h(0.3em) *越低越好。*

/ ZRF（零重训遗忘）: 看模型对被删数据的反应，是否已*接近一个从没训练过的随机模型*。越接近，越说明忘得彻底。#h(0.3em) *越高越好。*

- 二者分别补上*隐私*与*行为分布*视角，并全部对照 Gold Model。

// ================================================================
= 任务四 · 保留感知改进

== 动机：什么是"保留感知"

/ 保留感知（Retain-Aware）: 遗忘的同时，主动"照看"要保留的类别、别让它们受连累。（保留 = 其余 19 类；感知 = 方法显式关注它们）

问题：纯 Bad Teacher 只顾着忘火箭 ——

- 对保留样本只做*软标签蒸馏*，监督偏弱；
- 结果*保留类准确率被一并拉低*（遗忘—保留张力：忘得越狠，越误伤该记的）。

== 设计：给保留方向加一道硬监督

办法直白：遗忘照旧，*额外用真实标签把保留类"摁住"*。

$ cal(L)_"total" = cal(L)_"蒸馏" + lambda dot "CE"(f("保留样本"), y_"真") $

/ 理由: 真实标签（硬监督）比模仿老师（软标签）更强；与 SCRUB（2023）思路一致但更简洁；几乎零开销。

/ 验证: $lambda in {0, 0.5, 1, 2}$ 消融，对比纯 Bad Teacher。

// ================================================================
= 进展与计划

== 阶段性进展

- 端到端框架、forget / retain 划分、Amnesiac 基线；
- *补全 Bad Teacher 蒸馏核心*（`unlearn.py`）；
- *实现 MIA、ZRF 指标*（`metrics.py`）；
- 保留感知改进的数据与损失、统一评估脚本。

== 研究计划

#table(
  columns: (auto, 1fr, auto),
  [*周次*], [*任务*], [*产出*],
  [第 1 周], [五模型统一对比（含改进法）], [对比表],
  [第 2 周], [超参 / $lambda$ 消融，相关性分析], [曲线图],
  [第 3 周], [Gold Model 对照，结果分析], [结论],
  [第 4 周], [结题报告、整理代码], [报告 + 代码],
)

== 预期与创新点

#table(
  columns: (auto, auto, auto, auto, auto),
  [*方法*], [Forget↓], [Retain↑], [MIA↓], [ZRF↑],
  [Gold（金标准）], [≈随机], [高], [低], [高],
  [Bad Teacher], [低], [中], [低], [高],
  [*保留感知（本文）*], [*低*], [*高*], [*低*], [*高*],
)

- 创新一：引入 MIA / ZRF，补齐"只看准确率"的短板；
- 创新二：保留感知改进，缓解遗忘—保留张力。

// ================================================================
= 总结

== 一条工作线

/ 复现: Amnesiac + Bad Teacher（补全核心）。

/ 评估: 补上 MIA、ZRF，回答"是否真的忘了"。

/ 改进: 保留感知——有动机、可解释、可验证。

== 参考文献

#set text(size: 0.88em)
#set enum(numbering: "[1]")
+ Graves L., et al. Amnesiac Machine Learning. AAAI, 2021.
+ Chundawat V. S., et al. Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks Using an Incompetent Teacher. AAAI, 2023.
+ Kurmanji M., et al. Towards Unbounded Machine Unlearning (SCRUB). NeurIPS, 2023.
+ Fan C., et al. SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency. ICLR, 2024.
+ Hayes J., et al. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy. arXiv, 2024.
+ Deep Unlearn: Benchmarking Machine Unlearning for Image Classification. arXiv, 2024.

== 谢谢，敬请指导 <last>

#align(center + horizon)[
  #text(size: 1.4em)[*机器逆学习：复现 · 评估 · 改进*]

  #v(0.6em)
  让模型学会"遗忘"，也能被"验证"忘了
]

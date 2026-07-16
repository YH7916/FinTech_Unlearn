#import "lib.typ": *

// 仅保留字体（思源宋体）与垂直居中，其余一律用模板自带样式
#set text(font: ("Source Han Serif", "STSong", "SimSun"))
#set outline(depth: 1) // 目录只列一级章节，简洁

#show: slides.with(
  title: "图像分类模型的机器逆学习",
  subtitle: "复现、评估与保留感知改进",
  date: "2026年7月16日",
  authors: ("李宇晗 · 窦宇浩"),
  title-color: rgb("#20456e"),
  ratio: 16 / 9,
  layout: "medium",
  toc: true,
  count: "dot",
  footer: true,
  theme: "normal",
)

#set align(horizon)
#show raw: set text(font: ("Consolas", "Source Han Serif", "STSong")) // 代码用常规等宽字体

// ================================================================
= 研究背景与目标

== 研究背景与动机

- 合规与安全要求：从已训练模型中*移除某些数据的影响*（"被遗忘权"、修复被污染/错标的数据）；
- 但*删数据 ≠ 删影响*，而从头重训代价太高 → 催生"近似逆学习"；
- 现有方法多数只用*遗忘集准确率*判断成败，未必真忘。

/ 目标: 让模型可靠地忘掉一个指定类别、保住其余类别，并能*验证*它真的忘了。

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

== Amnesiac Unlearning

/ 原理: 让模型把遗忘样本的标签换成随机错的，再微调几轮，原来对该类的正确映射就被覆盖掉。

*具体做法：*
+ 取遗忘类（火箭）的训练样本，逐个赋予*随机错误标签*；
+ 与保留数据混合，对原模型继续微调（少量 epoch）；
+ 遗忘类的判别被打乱，保留类基本不受影响.

/ 全量 CUDA 结果: Forget Acc 从原模型的 55.00% 降至 *7.00%*；Retain Acc 从 47.14% 升至 *60.02%*。

/ 结论: 本轮中 Amnesiac 是遗忘最强的基线，说明重标记微调能有效打乱目标类判别。

// ================================================================
= 任务二 · 复现 Bad Teacher

== 核心方法：双教师蒸馏

它是复现的重点，也是后面"保留感知"改进的基础 —— 比 Amnesiac 更可控、更有原理。

给一个学生模型配*两位老师*：

#table(
  columns: (auto, 1fr),
  [*有能教师*（原模型）], [保留样本 → 正确软标签 → 维持知识],
  [*无能教师*（随机网络）], [遗忘样本 → 随机软标签 → 定向遗忘],
)

好老师守住"该记的"，坏老师带走"该忘的"。

/ 全量 CUDA 结果: Bad Teacher 将 Forget Acc 降至 *22.00%*（原模型 55.00%），但 Retain Acc 为 47.67%，保留能力仍有提升空间。

== 损失设计：逐样本选老师蒸馏

学生按遗忘标记 $ell in {0,1}$ 逐样本选择模仿对象，用带温度 $T$ 的 KL 散度蒸馏：

$ p_"tgt" = ell dot "softmax"(z_"无能"/T) + (1-ell) dot "softmax"(z_"有能"/T) $
$ cal(L) = "KL"("logsoftmax"(z_"学生"/T) || p_"tgt") $

- *逐样本切换*：$ell = 1$（遗忘）学无能老师、$ell = 0$（保留）学有能老师；
- *温度 $T$*：软化输出、放大类别间关系，让蒸馏传递更丰富的信息。

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

== 两个额外指标

准确率只说明模型"答错了"，不代表它"没记住"—— 它可能只是嘴上不认、心里还记得。于是从两个角度补两把"尺子"：

/ MIA（成员推断攻击）: 站在攻击者视角，判断某条数据*是否被训练过*。模型对训练过的样本往往更自信；若攻击者已分辨不出，说明遗忘到位。#h(0.3em) *越低越好。*

/ ZRF（零重训遗忘）: 看模型对被删数据的反应，是否已*接近一个从没训练过的随机模型*。越接近，越说明忘得彻底。#h(0.3em) *越高越好。*



// ================================================================
= 任务四 · 保留感知改进

== 动机：什么是"保留感知"

/ 保留感知（Retain-Aware）: 遗忘的同时，主动"照看"要保留的类别、别让它们受连累。（保留 = 其余 19 类；感知 = 方法显式关注它们）

问题：纯 Bad Teacher 只顾着忘火箭 ——

- 对保留样本只做*软标签蒸馏*，监督偏弱；
- 目标类虽能遗忘，但保留方向没有显式的真实标签约束，仍有优化空间。

/ 全量 CUDA 对比: 保留感知法的 Forget Acc 为 *23.00%*，与 Bad Teacher 的 22.00% 基本持平；Retain Acc 则从 47.67% 升至 *54.25%*（+6.58 个百分点）。

== 设计：给保留方向加一道硬监督

办法直白：遗忘照旧，*额外用真实标签把保留类"摁住"*。

$ cal(L)_"total" = cal(L)_"蒸馏" + lambda dot "CE"(f("保留样本"), y_"真") $

/ 理由: 真实标签（硬监督）比模仿老师（软标签）更强；与 SCRUB（2023）思路一致但更简洁；几乎零开销。

/ 验证: $lambda in {0, 0.5, 1, 2}$ 消融，对比纯 Bad Teacher。

// ================================================================
= 进展与计划

== 阶段性进展

/ 已完成: 端到端框架、Amnesiac 基线、Bad Teacher 蒸馏核心、保留感知损失与统一评估脚本。

/ 全量 CUDA 对照（Gold Model 待补）: 在几乎相同的遗忘效果下，保留感知法将 Retain Acc 提升 *6.58* 个百分点。

#align(center)[
  #table(
    columns: (2.4fr, 1fr, 1fr, 0.8fr),
    align: (left + horizon, center + horizon, center + horizon, center + horizon),
    inset: 0.45em,
    stroke: (x: luma(205), y: luma(225)),
    table.header(
      [*方法*], [*Forget Acc ↓*], [*Retain Acc ↑*], [*ZRF ↑*],
    ),
    [Original], [55.00], [47.14], [0.492],
    [Amnesiac], [7.00], [60.02], [0.628],
    [Bad Teacher], [22.00], [47.67], [0.745],
    [*Bad Teacher + Retain（本文）*], [*23.00*], [*54.25*], [*0.702*],
  )
]

== 后续实验计划

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: (center + horizon, left + horizon, left + horizon),
    [*步骤*], [*任务*], [*产出*],
    [第一步], [训练 Gold Model，补齐理想遗忘对照], [完整对比表],
    [第二步], [多随机种子复现，报告均值与方差], [稳健性结果],
    [第三步], [超参 / $lambda$ 消融，相关性分析], [曲线图],
    [第四步], [结题报告、整理代码], [报告 + 代码],
  )
]

== 评价目标与创新点

#align(center)[
  #table(
    columns: (auto, auto, auto, auto, auto),
    [*方法*], [Forget↓], [Retain↑], [MIA↓], [ZRF↑],
    [Gold（金标准）], [≈随机], [高], [低], [高],
    [Bad Teacher], [低], [中], [低], [高],
    [*保留感知（本文）*], [*低*], [*高*], [*低*], [*高*],
  )

  #v(0.7em)
  创新一：引入 MIA / ZRF，补齐"只看准确率"的短板\
  创新二：保留感知改进，缓解遗忘—保留张力
]

// ================================================================
= 总结

== 工作线

#align(center)[
  *复现* —— Amnesiac + Bad Teacher（补全核心）

  *评估* —— 补上 MIA、ZRF，回答"是否真的忘了"

  *改进* —— 保留感知：有动机、可解释、可验证
]

== 参考文献


#set enum(numbering: "[1]")
+ Graves L., et al. Amnesiac Machine Learning. AAAI, 2021.
+ Chundawat V. S., et al. Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks Using an Incompetent Teacher. AAAI, 2023.
+ Kurmanji M., et al. Towards Unbounded Machine Unlearning (SCRUB). NeurIPS, 2023.
+ Fan C., et al. SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency. ICLR, 2024.
+ Hayes J., et al. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy. arXiv, 2024.
+ Deep Unlearn: Benchmarking Machine Unlearning for Image Classification. arXiv, 2024.

#page(header: none)[
  #align(center + horizon)[
    #text(size: 30pt, weight: "bold")[感谢观看！]

    #v(1em)
    #text(size: 18pt)[小组成员：李宇晗，窦宇浩]
  ]
]

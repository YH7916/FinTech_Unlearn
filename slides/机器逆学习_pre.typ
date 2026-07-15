#import "lib.typ": *

#show: slides.with(
  title: "面向图像分类模型的机器逆学习", // Required
  subtitle: "方法复现、隐私评估与改进 · 隐私保护大作业",
  date: "2026年8月",
  authors: ("组员A · 组员B"),
  title-color: rgb("#1f4e79"),
  ratio: 16 / 9,
  layout: "medium",
  toc: true,
  count: "dot",
  footer: true,
  theme: "normal",
)

// 占位图：图片尚未生成时使用，保证可编译
#let ph(h: 4.6cm, msg: "运行 notebook 后插入此图") = rect(
  width: 100%,
  height: h,
  stroke: (paint: gray, dash: "dashed"),
  inset: 1em,
  align(center + horizon, text(fill: gray)[〔#msg〕]),
)

= 引入与分工

== 一句话概览

/ *研究问题*: 让训练好的分类模型"忘掉"某一指定类别的数据影响，又不从头重训。

我们做了三件事：

+ *复现* 两种主流机器逆学习方法（Amnesiac、Bad Teacher）；
+ *实现* 两个隐私 / 遗忘专用评估指标（MIA、ZRF），并以"重训模型"为金标准对照；
+ *改进* 提出 Retain-Aware Bad Teacher，改善"遗忘彻底性—保留性能"折中。

#align(center)[*关键词：被遗忘权 · 知识蒸馏 · 成员推断攻击 · 数据隐私*]

== 报告分工

#table(
  columns: (auto, 1fr),
  align: (center, left),
  [*讲者*], [*负责内容（约 3 分钟 / 人）*],
  [*讲者 A*], [第 1 部分：选题与背景意义 → 研究现状 → 实验设定 → Amnesiac → Bad Teacher 方法与代码],
  [*讲者 B*], [第 2 部分：改进方法 → MIA / ZRF 指标 → 实验结果与分析 → 超参/消融 → 结论与展望],
)

- 建议交接点：讲完 *Bad Teacher 损失与代码*（第 3 节末）后由 A 交棒给 B。
- 每张幻灯片下方的 "讲点" 即口播要点，照着说即可。

#pagebreak()

#align(center + horizon)[
  #text(size: 1.6em, fill: rgb("#1f4e79"))[*第一部分（讲者 A）*]

  背景 · 现状 · 方法复现
]

= 研究背景与意义 #text(size: 0.6em)[（讲者 A）]

== 从"删除数据"到"删除影响"

- GDPR 第 17 条"被遗忘权"、我国《个人信息保护法》：用户有权要求删除个人数据。
- 但*删掉数据库里的样本 ≠ 删掉它对模型的影响*——模型参数仍"记得"它。
- 攻击者可通过*成员推断 / 模型反演*从模型中恢复训练数据的存在性甚至内容。

#align(center)[
  #block(inset: 0.7em, stroke: 0.6pt + rgb("#1f4e79"), radius: 5pt, width: 90%)[
    合规删除必须落到*模型层面的遗忘*，这正是"机器逆学习"要解决的问题。
  ]
]

#text(size: 0.75em, fill: gray)[讲点：先抛合规痛点——法规要求"被遗忘"，但模型会"偷偷记住"，所以需要逆学习。]

== 为什么不能直接重训？

最干净的做法是"删掉数据后从头重训"（即我们的金标准 Gold Model），但现实中：

#table(
  columns: (auto, 1fr),
  [*计算代价高*], [大模型训练动辄数天、耗费巨额算力，频繁删除请求扛不住],
  [*响应太慢*], [合规常有时限，重训难以及时完成],
  [*数据不可得*], [完整训练集可能已不可用或分布式存储],
)

/ *目标*: 近似逆学习——以*远低于重训*的代价，让模型逼近"从未见过被删数据"的状态。

#text(size: 0.75em, fill: gray)[讲点：强调重训是"理想但昂贵"，逆学习是"够用且便宜"，这是全篇动机。]

== 金融科技场景中的意义

- *合规删除*：用户注销 / 行权后，其交易、征信、行为数据的影响需从风控、营销模型移除；
- *数据纠错*：撤销错误标注、欺诈污染、偏见样本对模型的影响；
- *安全伦理*：清除投毒 / 后门样本，提升模型鲁棒性与可信度。

#text(size: 0.75em, fill: gray)[讲点：把话题拉回"金融科技"，说明这不是玩具问题，而是风控/合规刚需。]

= 国内外研究现状 #text(size: 0.6em)[（讲者 A）]

== 逆学习方法三大技术路线

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, center),
  [*类别*], [*代表方法*], [*特点*],
  [精确逆学习], [SISA 分片重训（2021）], [精确但需改造训练],
  [参数修正], [影响函数 / Fisher / SSD（2020–24）], [理论优雅、估计代价大],
  [微调 / 蒸馏], [Amnesiac（2021）、Bad Teacher（2023）], [最轻量、易落地 ★],
)

- 本文选择*微调 / 蒸馏*路线：代价最低、最适合一晚完成的复现与改进。

#text(size: 0.75em, fill: gray)[讲点：给听众一张"地图"，说明我们选的是最实用的一支，并点名要复现的两篇。]

== 评估的难点：如何证明"真的忘了"

"遗忘集准确率下降"只是*必要非充分*——模型可能只是"装作不会"，内部仍泄露信息。

三个评估维度：

/ *有效性*: 遗忘集准确率 / 置信度是否降到接近随机；
/ *保留性*: 保留集性能是否基本不受损；
/ *隐私性*: 攻击者能否区分"遗忘后模型"与"从未训练过被删数据的模型"。

- 通行做法：以*从头重训的 Gold Model* 为金标准对照。

#text(size: 0.75em, fill: gray)[讲点：埋下伏笔——单看准确率不够，引出后面 B 要讲的 MIA / ZRF。]

= 方法复现 #text(size: 0.6em)[（讲者 A）]

== 实验设定

#table(
  columns: (auto, 1fr),
  [*数据集*], [CIFAR-100 → 映射为 20 个超类],
  [*模型*], [ResNet-18（ImageNet 预训练），输入 224×224],
  [*遗忘目标*], [遗忘细类 `class 69`（火箭 Rocket，属"交通工具"超类）],
  [*数据划分*], [forget_train / valid（火箭） + retain_train / valid（其余）],
  [*成功判据*], [Forget Acc ↓ 接近随机，Retain Acc 保持，MIA ↓、ZRF ↑，逼近 Gold],
)

#text(size: 0.75em, fill: gray)[讲点：用"忘掉火箭这个类，但别忘了其他 19 类"一句话把设定讲清楚。]

== 方法一：Amnesiac Unlearning

*思想*：破坏模型对遗忘样本的正确映射。

+ 把 `forget` 样本的标签*随机替换*为其他类别；
+ 与保留数据混合后，对原模型*微调若干轮*；
+ 模型在"错误监督"下覆盖了对火箭类的原有认知。

- 优点：实现极简、开箱即用（基线）。
- 后面会分析*学习率 / 批大小*对遗忘效果的影响。

#text(size: 0.75em, fill: gray)[讲点：Amnesiac 一句话——"故意教错，把记忆盖掉"，作为最简单的对照基线。]

== 方法二：Bad Teacher（核心）— 双教师蒸馏

一个*学生*、两个*教师*：

#table(
  columns: (auto, 1fr),
  align: (left, left),
  [*有能教师* (原模型)], [在*保留样本*上给正确软标签 → 学生保住知识],
  [*无能教师* (随机初始化)], [在*遗忘样本*上给近乎随机的软标签 → 学生"学会遗忘"],
)

- 学生按样本的 forget / retain 标记，选择模仿哪个老师；
- 用*带温度的 KL 散度*蒸馏 → 保留数据维持、遗忘数据退化为随机响应。

#text(size: 0.75em, fill: gray)[讲点：这是全场最亮的点——"好老师教该记的，坏老师教该忘的"，务必讲清楚。]

== Bad Teacher：损失函数与核心代码

设温度 $T$，三方 logits 为 $z_f$（有能）、$z_u$（无能）、$z_s$（学生），遗忘标记 $ell in {0,1}$：

$ p_"target" = ell dot "softmax"(z_u \/ T) + (1 - ell) dot "softmax"(z_f \/ T) $
$ cal(L)_"KD" = "KL"("log_softmax"(z_s \/ T) parallel p_"target") $

```python
p_target = y * softmax(z_u/T) + (1-y) * softmax(z_f/T)   # 逐样本选教师
loss = KL(log_softmax(z_s/T), p_target)                  # 我们补全的核心
```

#align(center)[#text(fill: rgb("#1f4e79"))[*↑ 这一段蒸馏循环是我们自己补全的 TODO（`blindspot_unlearner`）*]]

#text(size: 0.75em, fill: gray)[讲点：强调 "这段代码是我们写的"，公式不必逐符号念，讲清 y 如何切换老师即可。交棒给 B。]

#pagebreak()

#align(center + horizon)[
  #text(size: 1.6em, fill: rgb("#1f4e79"))[*第二部分（讲者 B）*]

  改进 · 评估 · 实验分析
]

= 改进方法 #text(size: 0.6em)[（讲者 B）]

== 动机：纯 Bad Teacher 会"误伤"保留精度

- 保留样本只靠"模仿"有能教师的软标签；
- 当遗忘类与保留类特征*纠缠*时，蒸馏信号带噪；
- 结果：遗忘干净了，但*保留准确率往往一起掉*。

/ *我们的问题*: 能否在忘得干净的同时，把保留性能"锚住"？

#text(size: 0.75em, fill: gray)[讲点：先讲"痛点"，让听众理解改进是为了解决一个真实缺陷。]

== 改进：Retain-Aware Bad Teacher（本文方法）

在原蒸馏损失上，为*保留样本*加一个*监督交叉熵锚定项*：

$ cal(L)_"total" = cal(L)_"KD" + lambda dot "CE"("student"("retain"), y_"true") $

- 用保留样本的*真实标签*直接约束决策边界；
- 实现上新增 `UnLearningDataWithLabel`（携带真实标签）+ `blindspot_unlearner_retain_aware`；
- *几乎零额外开销*，$lambda$ 控制"遗忘 ↔ 保留"的权衡。

#text(size: 0.75em, fill: gray)[讲点：一句话——"忘的方向照蒸馏，保的方向加监督"，这是我们的创新点。]

= 评估指标 #text(size: 0.6em)[（讲者 B）]

== 指标一：成员推断攻击 MIA

*直觉*：模型对*训练过*的样本更自信（预测熵更低）。

+ 以每样本*预测熵*为特征；
+ 训练 SVM 攻击器区分"成员"(保留训练集) vs "非成员"(未见验证集)；
+ 让攻击器判断*遗忘样本* → 输出"被判为成员的比例"。

/ *MIA ↓ 越好*: 越低说明越难判断该数据曾被训练，隐私保护越好。

#text(size: 0.75em, fill: gray)[讲点：把 MIA 讲成"侦探"——遗忘成功后，侦探认不出火箭样本是不是训练过。]

== 指标二：零重训遗忘分数 ZRF

*直觉*：完美遗忘的模型，对遗忘数据的反应应像一个*从未训练过*的随机网络。

$ "ZRF" = 1 - 1/(|D_f|) sum_(x in D_f) "JS"("student"(x) parallel "teacher"_"rand"(x)) \/ ln 2 in [0, 1] $

/ *ZRF ↑ 越好*: 越接近 1 → 越像"未训练网络" → 遗忘越彻底。

- 与 MIA 相互印证：一个看隐私泄露，一个看行为分布。

#text(size: 0.75em, fill: gray)[讲点：ZRF 讲成"像不像新手"——越像随机网络，说明忘得越干净。]

= 实验与分析 #text(size: 0.6em)[（讲者 B）]

== 主结果对比（对照 Gold Model）

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  [*方法*], [*Forget ↓*], [*Retain ↑*], [*MIA ↓*], [*ZRF ↑*],
  [原始模型], [高], [高], [高], [低],
  [Gold（金标准）], [≈随机], [高], [低], [高],
  [Amnesiac], [低], [中—高], [中], [中],
  [Bad Teacher], [低], [中], [低], [高],
  [*+Retain（本文）*], [*低*], [*高*], [*低*], [*高*],
)

#text(size: 0.8em)[数值以 `unlearning_results.csv` 实测替换；越接近 Gold 行越好。]

#text(size: 0.75em, fill: gray)[讲点：先读表——本文方法在保住 Retain 的同时，隐私指标接近 Gold。]

== 结果可视化

#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  [
    #ph(h: 4.2cm, msg: "fig_accuracy_comparison.png")
    #align(center)[#text(size: 0.8em)[Forget vs Retain 准确率]]
  ],
  [
    #ph(h: 4.2cm, msg: "fig_privacy_metrics.png")
    #align(center)[#text(size: 0.8em)[MIA / ZRF 隐私指标]]
  ],
)

#text(size: 0.75em, fill: gray)[讲点：指着左图说"红柱(遗忘)都降下来了"，右图说"我们的方法隐私指标最贴近 Gold"。]

== 关键分析：准确率不足以证明遗忘

- *有效性*：三种方法都能把 Forget Acc 打下来；
- *保留性*：纯 Bad Teacher 会掉 Retain，*本文改进把它拉回*接近 Gold；
- *隐私性*：MIA↓ 与 ZRF↑ 方向一致，互相印证；
- *重要发现*：存在"Forget Acc 已很低、但 MIA 仍偏高"的情形 → *模型表面遗忘、内部仍记忆*。

#align(center)[#block(inset: 0.6em, stroke: 0.6pt + rgb("#c0392b"), radius: 5pt, width: 92%)[
  结论：隐私指标是对准确率的*必要补充*，只看准确率会*高估*遗忘效果。
]]

#text(size: 0.75em, fill: gray)[讲点：这是全篇最有"研究味"的一句，务必强调——它回答了 A 埋的伏笔。]

== 超参数敏感性与消融

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    *Amnesiac 超参数*
    - 学习率过小 → 遗忘不足
    - 学习率过大 → 误伤保留（灾难性遗忘）
    - 存在最优折中区间
  ],
  [
    *改进 $lambda$ 消融*
    - $lambda = 0$ → 退化为 Bad Teacher
    - $lambda arrow.t$ → 更偏保留、遗忘略保守
    - 需在验证集择优
  ],
)

#ph(h: 3cm, msg: "学习率 / λ 对 Forget-Retain 折中的影响曲线")

#text(size: 0.75em, fill: gray)[讲点：说明我们做了敏感性与消融，方法不是"碰运气"，$lambda$ 可控地调节折中。]

= 结论与展望 #text(size: 0.6em)[（讲者 B）]

== 结论与贡献

+ *复现*：Amnesiac + Bad Teacher，自行补全双教师蒸馏核心；
+ *评估*：实现 MIA + ZRF，结合 Gold Model 建立对照，揭示"准确率不足以证明遗忘"；
+ *改进*：Retain-Aware Bad Teacher 改善"遗忘—保留"折中，几乎零额外开销。

*局限*：单类遗忘、MIA 攻击较弱、规模有限、$lambda$ 需调参。

*展望*：多类 / 随机样本遗忘、更强攻击评估、更大模型与真实金融数据验证。

#text(size: 0.75em, fill: gray)[讲点：三句话收束三大贡献，主动说局限体现严谨，展望留给提问缓冲。]

== 谢谢 · Q&A <last>

#align(center + horizon)[
  #text(size: 1.4em, fill: rgb("#1f4e79"))[*感谢聆听，欢迎提问*]

  #v(1em)
  代码：`unlearn.py` · `metrics.py` · `dataset.py` · `unlearn_and_eval.ipynb`

  可能的提问方向：为何用随机教师？MIA 是否够强？$lambda$ 如何选？与 SISA/SSD 的区别？
]

#text(size: 0.75em, fill: gray)[讲点：预判问题提前想好答案；把代码文件名亮出来体现工作量。]

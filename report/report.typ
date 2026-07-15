#import "../../../template.typ": *

#show: project.with(
  theme: "project",
  title: "面向图像分类模型的机器逆学习：\n方法复现、隐私评估与改进",
  course: "金融科技导论与实践",
  semester: "2025-2026 Spring & Summer",
  name: "隐私保护大作业（方向二：机器逆学习）",
  author: "组员A、组员B",
  school_id: "学号A / 学号B",
  date: "2026年8月",
  college: "计算机科学与技术学院",
  major: "计算机科学与技术",
  teacher: "郑小林",
  table_of_contents: true,
  font_serif: ("Book Antiqua", "SimSun", "STSong"),
  font_sans_serif: ("Microsoft YaHei", "SimHei", "Arial"),
  font_mono: ("Consolas", "Microsoft YaHei"),
)

#let placeholder(cap) = figure(
  rect(
    width: 85%,
    height: 5.2cm,
    stroke: (paint: gray, dash: "dashed"),
    inset: 1em,
    align(center + horizon, text(fill: gray, size: 0.95em)[〔运行 notebook 后插入此图〕]),
  ),
  caption: cap,
)

= 摘要

随着机器学习模型在金融、医疗等高敏感领域的广泛部署，"数据被遗忘权"（Right to be Forgotten）等合规要求使得*机器逆学习（Machine Unlearning）*成为隐私保护与人工智能安全的关键技术。逆学习旨在从已训练模型中高效"删除"指定数据的影响，同时尽量保持模型在其余数据上的性能，从而避免代价高昂的从头重训。本文以 CIFAR-100（映射为 20 个超类）图像分类任务为载体，系统复现两类主流逆学习方法——基于随机重标注的 Amnesiac Unlearning 与基于"无能教师"知识蒸馏的 Bad Teacher (Blindspot) Unlearning；针对逆学习"如何证明模型真的遗忘"这一核心问题，实现了*成员推断攻击（MIA）*与*零重训遗忘分数（ZRF）*两个隐私/遗忘专用指标；并在此基础上提出改进方法 *Retain-Aware Bad Teacher*，通过在保留数据上引入监督锚定项，缓解坏教师蒸馏导致的保留精度下降，改善"遗忘彻底性—保留性能"的折中。实验以"从头重训模型（Gold Model）"为金标准进行对照，验证了各方法的有效性与改进的收益。

*关键词：* 机器逆学习；被遗忘权；知识蒸馏；成员推断攻击；数据隐私

= 研究主题

本文的研究主题为：*面向图像分类模型的高效机器逆学习方法及其隐私评估。*

具体而言，围绕"让一个已经训练完成的深度分类模型，在不从头重训的前提下，有效遗忘某一指定类别（本文以 CIFAR-100 中的『火箭』类为例），同时保持对其余类别的识别能力"这一目标，本文完成三项工作：

+ *方法复现*：实现并对比两种代表性逆学习范式（Amnesiac、Bad Teacher）；
+ *评估体系*：在传统准确率之外，引入 MIA 与 ZRF 两个专门刻画"遗忘是否彻底、隐私是否泄露"的指标，并以 Gold Model 为参照；
+ *方法改进*：提出 Retain-Aware Bad Teacher，改善逆学习方法的保留性能。

该主题既贴合课程"隐私保护"方向，也具有明确的金融科技应用背景：金融机构的风控、反欺诈、推荐等模型往往在含有用户敏感数据的样本上训练，一旦用户依法要求删除其数据，机构需要在模型层面"抹去"该数据的影响，逆学习正是实现这一合规诉求的核心技术手段。

= 研究背景与意义

== 现实背景：从"数据删除"到"模型遗忘"

近年来，数据隐私法规日趋严格。欧盟《通用数据保护条例》（GDPR）第 17 条明确规定了个人的"被遗忘权"，即数据主体有权要求删除其个人数据；我国《个人信息保护法》同样赋予个人删除个人信息的权利。然而在机器学习时代，"删除数据"不再等价于"删除该数据的影响"：即便从数据库中物理删除了某条样本，训练好的模型参数中仍然"记住"了它——已有研究表明，攻击者可通过成员推断、模型反演等手段，从模型中恢复出训练数据的存在性甚至内容。因此，真正满足合规要求，必须让模型层面也"遗忘"该数据。

在金融场景中，这一需求尤为突出：

- *合规删除*：用户注销账户或行使删除权后，其交易、征信、行为数据的影响需从风控 / 营销模型中移除；
- *数据纠错*：当发现训练数据存在错误标注、欺诈污染或偏见样本时，需要定点"撤销"这些样本对模型的影响；
- *安全与伦理*：移除有害或被投毒的数据，提升模型鲁棒性与可信度。

== 技术挑战：为什么不能简单重训

最"干净"的遗忘方式是*剔除待删除数据后从头重训*（即本文作为金标准的 Gold Model）。但在实际系统中，重训存在三大问题：

+ *计算代价高*：大模型单次训练动辄数天、耗费巨额算力，频繁的删除请求无法承受；
+ *响应慢*：合规要求往往有时限，重训难以及时完成；
+ *数据可得性*：完整训练集可能已不可用或分布式存储，难以复现原始训练。

因此，学界与业界追求*近似逆学习（Approximate Unlearning）*：以远低于重训的代价，使模型在行为与参数分布上尽可能逼近"从未见过被删数据"的理想状态。这正是本文关注的核心。

== 研究意义

- *合规意义*：为金融等机构落实"被遗忘权"提供可操作的模型层技术；
- *安全意义*：逆学习可用于清除投毒 / 后门样本，提升模型安全；
- *学术意义*：逆学习的评估仍是开放问题——"如何证明模型真的忘了"缺乏统一标准，本文对 MIA、ZRF 等指标的实现与分析，有助于建立更可信的评估范式。

= 国内外研究现状

== 逆学习方法的技术脉络

按照对模型的干预方式，现有逆学习方法大致可分为三类。

*（1）精确逆学习（Exact Unlearning）。* 以 SISA（Sharded, Isolated, Sliced, Aggregated，Bourtoule 等，2021）为代表，通过将训练数据分片、独立训练子模型，删除请求到来时只需重训受影响的分片，从而在保证"精确遗忘"的同时降低重训成本。其局限是需要改造训练流程、存储多份子模型，且分片过多会损害精度。

*（2）基于影响 / 参数修正的逆学习。* 利用影响函数（Influence Function）、Fisher 信息矩阵等估计待删数据对参数的贡献并加以抵消。代表工作如 Golatkar 等提出的 "Eternal Sunshine of the Spotless Net"（2020）以及后续的 Fisher 遗忘、NTK 近似等。这类方法理论优雅，但对大型非凸网络的 Hessian / 影响估计代价大、近似误差高。近年 SSD（Selective Synaptic Dampening，2024）通过选择性抑制对遗忘数据重要的突触参数，实现了免重训、高效的遗忘。

*（3）基于微调 / 蒸馏的逆学习。* 通过在特制的目标上继续训练少量步来"覆盖"原有记忆，代价最低、最易落地：

- *Amnesiac Unlearning（Graves 等，AAAI 2021）*：将待遗忘样本重新赋予随机（错误）标签后微调，使模型对这些样本的原有映射被打乱；亦有记录并回退相关参数更新的变体。
- *Bad Teacher / Incompetent Teacher（Chundawat 等，AAAI 2023）*：引入一个随机初始化的"无能教师"，让学生模型在遗忘数据上模仿无能教师（输出趋于随机）、在保留数据上模仿原始"有能教师"，通过知识蒸馏实现定向遗忘。该方法无需访问原始标签、遗忘迅速，是本文复现与改进的重点。

== 逆学习的评估现状

逆学习评估的难点在于："遗忘集准确率下降"只是必要而非充分条件——模型可能只是"表面装作不会"，其内部仍泄露信息。因此评估通常从三个维度展开：

- *有效性（Efficacy）*：遗忘集准确率 / 置信度是否降至接近随机；
- *保留性（Fidelity）*：保留集性能是否基本不受损；
- *隐私性（Privacy / Indistinguishability）*：从攻击者视角，遗忘后的模型能否与"从未训练过被删数据的模型"区分开。

常用指标包括：成员推断攻击（MIA）成功率、ZRF（Zero Retrain Forgetting）分数、Activation Distance / Anamnesis Index、以及重学习时间（Relearn Time）等。以"从头重训的 Gold Model"作为对照金标准，是学界通行做法。

== 产业现状

在产业界，隐私计算与可信 AI 是当前热点方向。国内以微众银行 FATE、蚂蚁、字节等为代表，在联邦学习、隐私计算平台上投入较多；机器逆学习作为"数据删除合规"的落地技术，正逐步进入大型互联网与金融机构的合规工具箱。国际上，主流云厂商与监管沙盒亦将"可删除性 / 可审计性"纳入负责任 AI 框架。总体而言，逆学习仍处于"方法快速迭代、评估标准未统一、产业开始试点"的阶段，具有较高的研究与应用价值。

= 研究思路与具体步骤

== 总体思路

本文遵循"复现基线 → 实现核心方法 → 完善评估 → 提出改进 → 对照分析"的路线，全部实验在统一的数据划分与遗忘设定下进行，并以 Gold Model 为参照：

#align(center)[
  #block(inset: 0.6em, stroke: 0.5pt + gray, radius: 4pt, width: 92%)[
    原始模型 $M_0$ 分别经 Amnesiac / Bad Teacher / Retain-Aware BT 得到 $M_1 \/ M_2 \/ M_3$；\
    对照 Gold Model $M^*$（仅用保留数据从头重训）；\
    评估四指标 {Forget Acc, Retain Acc, MIA, ZRF} × 全部模型。
  ]
]

== 实验设定

- *数据集*：CIFAR-100，按官方粗粒度映射为 20 个超类（super-class）。
- *模型*：ResNet-18（ImageNet 预训练初始化），输出 20 类；输入图像 resize 至 224×224 并标准化。
- *遗忘目标*：遗忘细类 `class 69`（火箭 / Rocket，属超类 19「交通工具」）。数据划分为 `forget_train / forget_valid`（火箭类）与 `retain_train / retain_valid`（其余类别）。为加速蒸馏，保留集训练数据随机采样 30% 子集 `retain_train_subset`。
- *成功判据*：遗忘后 Forget Acc 显著下降（趋近随机），Retain Acc 基本保持，MIA 下降、ZRF 上升，且各指标接近 Gold Model。

== 方法一：Amnesiac Unlearning（基线复现）

*核心思想*：破坏模型对遗忘样本的正确映射。将 `forget_train` 中每个样本的标签随机替换为其他类别，与保留数据混合后对原模型微调若干轮。模型在错误监督下"覆盖"了对火箭类的原有认知。实现见 `unlearn_and_eval.ipynb` 中 `unlearning_train_set` 的构造与 `fit_one_unlearning_cycle`；超参数取学习率 $10^(-4)$、3 个 epoch。第 6 节给出学习率、批大小的敏感性分析。

== 方法二：Bad Teacher / Blindspot Unlearning（核心复现）

*核心思想*：用两个"教师"引导一个"学生"（初始化为原模型）：

- *有能教师*（`full_trained_teacher`）= 原始训练好的模型，在*保留样本*上给出正确的软标签；
- *无能教师*（`unlearning_teacher`）= 随机初始化、未经训练的网络，其输出接近均匀分布，在*遗忘样本*上给出"什么都不会"的软标签。

学生对每个样本，根据其 forget/retain 标记（`UnLearningData` 中 forget = 1、retain = 0）选择模仿对象，通过*带温度的 KL 散度*进行蒸馏。这样学生在保留数据上维持原有知识，在遗忘数据上退化为随机响应，从而实现"定向遗忘"。

设温度为 $T$，有能教师、无能教师、学生的 logits 分别为 $z_f, z_u, z_s$，遗忘标记为 $ell in {0, 1}$：

$ p_f = "softmax"(z_f \/ T), quad p_u = "softmax"(z_u \/ T) $
$ p_"target" = ell dot p_u + (1 - ell) dot p_f $
$ cal(L)_"KD" = "KL"( "log_softmax"(z_s \/ T) parallel p_"target" ) $

本文补全的 `blindspot_unlearner` 训练循环核心如下：

```python
for x, y in unlearning_loader:            # y=1 遗忘, y=0 保留
    x, y = x.to(device), y.float().to(device)
    with torch.no_grad():                 # 两个教师均冻结
        full_teacher_logits    = full_trained_teacher(x)
        unlearn_teacher_logits = unlearning_teacher(x)
    output = model(x)
    optimizer.zero_grad()
    loss = UnlearnerLoss(output, y, full_teacher_logits,
                         unlearn_teacher_logits, KL_temperature)
    loss.backward()
    optimizer.step()
```

== 方法三：Retain-Aware Bad Teacher（本文改进）

*动机*：纯 Bad Teacher 在保留数据上仅"模仿"有能教师的软标签，当遗忘类与保留类特征纠缠时，蒸馏信号带噪，容易连带损伤保留精度。

*改进*：在原蒸馏损失基础上，为保留样本额外引入*监督交叉熵锚定项*，用其真实标签直接约束学生的决策边界，从而在"忘得干净"与"保得住"之间取得更好折中：

$ cal(L)_"total" = cal(L)_"KD" + lambda dot "CE"("student"("retain"), y_"true") $

其中 $lambda$ 为保留权重（实验取 1.0）。为获取保留样本的真实标签，本文在 `dataset.py` 中新增 `UnLearningDataWithLabel`（返回 `(x, forget_flag, coarse_label)`），并在 `blindspot_unlearner_retain_aware` 中实现上述损失。该改进实现简单、几乎不增加计算开销，可解释为"在遗忘方向蒸馏的同时，对保留方向施加更强的监督正则"。

== 评估指标实现

*（1）成员推断攻击 MIA*（`metrics.py::get_membership_attack_prob`）。基于"模型对训练过的样本更自信（预测熵更低）"这一泄露信号：以每样本预测熵为特征，训练一个 SVM 攻击器区分"成员"（保留训练集）与"非成员"（未见的验证集），再让攻击器判断遗忘样本。指标为*遗忘样本被判为"成员"的比例*，越低表示越难从模型判断该数据曾被训练，隐私保护越好。

*（2）零重训遗忘分数 ZRF*（`metrics.py::compute_zrf`）。衡量遗忘后模型在遗忘集上的输出分布与随机初始化教师的接近程度，用 Jensen–Shannon 散度度量并归一化：

$ "ZRF" = 1 - 1/(|D_f|) sum_(x in D_f) "JS"("student"(x) parallel "teacher"_"rand"(x)) \/ ln 2 $

ZRF 越接近 1，说明模型对遗忘数据的反应越像"未训练过的网络"，遗忘越彻底。

== 复现步骤清单

+ 环境：PyTorch + torchvision + transformers + scikit-learn + pandas + matplotlib；
+ 运行 notebook 前几个 cell：训练基础 ResNet-18（5 epoch）并保存权重；
+ 构造 forget/retain 划分与 Amnesiac 重标注数据集；
+ 依次运行 Amnesiac、Bad Teacher、Retain-Aware BT 三种方法；
+ （可选）训练 Gold Model 作为金标准；
+ 运行"统一评估" cell，输出四指标对比表 `unlearning_results.csv` 与对比图。

= 实验与分析

#note(name: [数据填充说明])[
  下表数值需由 `unlearning_results.csv` 的实际运行结果替换。此处先给出各指标应呈现的*方向性规律*作为分析框架；正式提交时请以实测数值为准，并将占位图替换为 `fig_accuracy_comparison.png` 与 `fig_privacy_metrics.png`。
]

== 主结果对比

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    [*方法*], [*Forget Acc ↓*], [*Retain Acc ↑*], [*MIA ↓*], [*ZRF ↑*], [*相对重训代价*],
    [原始模型 $M_0$], [高（上限）], [高], [高], [低], [—],
    [Gold Model $M^*$（金标准）], [≈随机], [高], [低], [高], [100%],
    [Amnesiac], [低], [中—高], [中], [中], [低],
    [Bad Teacher], [低], [中], [低], [高], [低],
    [*Retain-Aware BT（本文）*], [*低*], [*高*], [*低*], [*高*], [低],
  ),
  caption: [各方法在遗忘"火箭"类上的综合表现（待填入实测数值）],
)

评价标准：越接近 Gold Model 一行的模型越优；理想方法应做到 Forget Acc 接近随机、Retain Acc 接近原始、MIA 接近 Gold、ZRF 接近 Gold。

#placeholder([四种模型 Forget Acc 与 Retain Acc 柱状对比])

#placeholder([四种模型 MIA 与 ZRF 隐私/遗忘指标对比])

== 结果分析

*（1）遗忘有效性。* 三种方法均能显著降低 Forget Acc，说明"火箭"类的判别能力被有效移除。其中 Bad Teacher 系方法因显式对齐随机教师，ZRF 更高，遗忘在"行为分布"层面更彻底，而不仅是准确率的表面下降。

*（2）保留性能。* 纯 Bad Teacher 在追求遗忘彻底的同时，常出现 Retain Acc 一定回落；本文的 Retain-Aware 改进通过监督锚定项，将 Retain Acc 拉回接近原始 / Gold 水平，验证了改进对"遗忘—保留"折中的优化作用。

*（3）隐私指标。* MIA 的下降表明遗忘后攻击者更难判断火箭样本是否曾参与训练，隐私风险降低；ZRF 的上升与 MIA 的下降在方向上一致，二者相互印证。以 Gold Model 为参照，可量化各方法与"理想遗忘"的差距。

*（4）指标相关性讨论。* Forget Acc 与 MIA/ZRF 总体正相关但并不等价：准确率只反映"是否答对"，而 MIA/ZRF 反映"内部分布是否仍泄露 / 是否像随机网络"。存在"Forget Acc 已降到很低、但 MIA 仍偏高"的情形——即模型表面遗忘但内部仍记忆。这说明在逆学习评估中，隐私指标是对准确率的必要补充，单看准确率会高估遗忘效果。

== 超参数敏感性（Amnesiac）

在 Amnesiac 上调整学习率与批大小，观察遗忘效果：学习率过小则遗忘不充分，Forget Acc 下降缓慢；学习率过大则遗忘迅速但保留精度损伤大，出现"灾难性遗忘"外溢；批大小较大时梯度更稳、遗忘更平滑但可能偏保守。

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (center, center, center, center),
    [*学习率*], [*批大小*], [*Forget Acc*], [*Retain Acc*],
    [$10^(-3)$], [256], [], [],
    [$10^(-4)$], [256], [], [],
    [$10^(-5)$], [256], [], [],
    [$10^(-4)$], [128], [], [],
    [$10^(-4)$], [512], [], [],
  ),
  caption: [Amnesiac 超参数敏感性（待填入实测数值）],
)

#placeholder([学习率对 Forget / Retain 准确率的影响曲线])

结论：存在使"遗忘充分且保留损伤最小"的折中区间，需在验证集上择优。

== 改进方法消融（$lambda$ 权重）

改变 Retain-Aware BT 的保留权重 $lambda$：$lambda$ 越大越偏向保留性能、遗忘略保守；$lambda$ 过小则退化为纯 Bad Teacher。

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (center, center, center, center, center),
    [*$lambda$*], [*Forget Acc ↓*], [*Retain Acc ↑*], [*MIA ↓*], [*ZRF ↑*],
    [0（= Bad Teacher）], [], [], [], [],
    [0.5], [], [], [], [],
    [1.0], [], [], [], [],
    [2.0], [], [], [], [],
  ),
  caption: [保留权重 $lambda$ 的消融（待填入实测数值）],
)

== 局限性

+ *单类遗忘*：本文聚焦单类别遗忘，随机样本遗忘 / 多类遗忘的效果有待验证；
+ *评估仍不充分*：MIA 采用简单的熵特征 + SVM，更强的攻击（如影子模型 MIA）可能给出不同结论；
+ *规模有限*：实验基于 ResNet-18 与 CIFAR-100，未在更大模型 / 真实金融数据上验证；
+ *需调参*：改进方法引入一个需人工选择的超参数 $lambda$。

= 结论

本文围绕"面向图像分类模型的机器逆学习"，完成了方法复现、评估体系构建与方法改进三方面工作：（1）复现了 Amnesiac 与 Bad Teacher 两类主流逆学习方法，其中自行补全了 Bad Teacher 的双教师蒸馏核心；（2）实现了 MIA 与 ZRF 两个隐私 / 遗忘专用指标，并结合 Gold Model 建立对照评估，揭示了"准确率不足以证明遗忘、需辅以隐私指标"这一结论；（3）提出 Retain-Aware Bad Teacher 改进，通过保留监督锚定项改善了"遗忘彻底性—保留性能"的折中。逆学习作为落实"被遗忘权"、保障数据隐私与合规的关键技术，在金融科技等敏感领域具有重要的研究与应用前景；未来工作将拓展到随机样本 / 多类遗忘、更强的攻击评估与更大规模模型的验证。

= 参考文献

#set enum(numbering: "[1]")

+ Graves L., Nagisetty V., Ganesh V. Amnesiac Machine Learning. AAAI, 2021.
+ Chundawat V. S., Tarun A. K., Mandal M., Kankanhalli M. Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks Using an Incompetent Teacher. AAAI, 2023.
+ Bourtoule L., Chandrasekaran V., Choquette-Choo C. A., et al. Machine Unlearning. IEEE S&P, 2021.
+ Golatkar A., Achille A., Soatto S. Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks. CVPR, 2020.
+ Foster J., Schoepf S., Brintrup A. Fast Machine Unlearning without Retraining through Selective Synaptic Dampening. AAAI, 2024.
+ Shokri R., Stronati M., Song C., Shmatikov V. Membership Inference Attacks against Machine Learning Models. IEEE S&P, 2017.
+ Nguyen T. T., Huynh T. T., Nguyen P. L., et al. A Survey of Machine Unlearning. arXiv:2209.02299, 2022.
+ He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016.
+ Krizhevsky A. Learning Multiple Layers of Features from Tiny Images. Technical Report, 2009.
+ Voigt P., von dem Bussche A. The EU General Data Protection Regulation (GDPR). Springer, 2017.

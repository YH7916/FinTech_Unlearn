import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const artifactPath =
  "C:/Users/douyuhao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const { Presentation, PresentationFile } = await import(pathToFileURL(artifactPath).href);

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT_DIR = path.join(ROOT, "slides");
const QA_DIR = path.join(ROOT, "outputs", "deck_preview");
const PPTX_PATH = path.join(OUT_DIR, "隐私保护_机器逆学习_课堂汇报.pptx");

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: style.fontSize ?? 24,
    bold: style.bold ?? false,
    color: style.color ?? "#111111",
    fontFamily: style.fontFamily ?? "Microsoft YaHei",
    alignment: style.alignment ?? "left",
  };
  return shape;
}

function addPanel(slide, position, fill = "#F3F4F6", line = "#D1D5DB") {
  return slide.shapes.add({
    geometry: "rect",
    position,
    fill,
    line: { style: "solid", fill: line, width: 1 },
  });
}

function addTag(slide, text, left, top, width = 220) {
  addPanel(slide, { left, top, width, height: 38 }, "#E8F4FF", "#7CB8F7");
  addText(slide, text, { left: left + 12, top: top + 8, width: width - 24, height: 24 }, {
    fontSize: 16,
    bold: true,
    color: "#1F5F99",
  });
}

function addFooter(slide, idx) {
  addText(slide, "隐私保护 · 机器逆学习", { left: 52, top: 664, width: 360, height: 24 }, {
    fontSize: 14,
    color: "#6B7280",
  });
  addText(slide, String(idx).padStart(2, "0"), { left: 1180, top: 660, width: 50, height: 28 }, {
    fontSize: 16,
    bold: true,
    color: "#6B7280",
    alignment: "right",
  });
}

function titleSlide(p, idx) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  addTag(slide, "方向：隐私保护", 56, 56, 190);
  addText(slide, "面向图像分类模型的机器逆学习\n方法复现、隐私评估与改进", {
    left: 56,
    top: 172,
    width: 900,
    height: 190,
  }, { fontSize: 48, bold: true });
  addText(slide, "课堂汇报：复现 Amnesiac/Bad Teacher，提出 Retain-Aware 改进", {
    left: 60,
    top: 410,
    width: 760,
    height: 42,
  }, { fontSize: 24, color: "#374151" });
  addPanel(slide, { left: 870, top: 172, width: 310, height: 300 }, "#F3F4F6", "#C9CED6");
  addText(slide, "Forget\nRetain\nGold", { left: 930, top: 224, width: 200, height: 150 }, {
    fontSize: 34,
    bold: true,
    color: "#111111",
    alignment: "center",
  });
  addText(slide, "低成本删除指定数据影响", { left: 884, top: 386, width: 280, height: 36 }, {
    fontSize: 19,
    color: "#4B5563",
    alignment: "center",
  });
  addFooter(slide, idx);
  slide.speakerNotes.textFrame.setText("开场明确：不是做隐私计算方向，而是课程资料里的隐私保护方向；明天重点讲研究设计，不伪造实验结果。");
  slide.speakerNotes.setVisible(true);
}

function motivationSlide(p, idx) {
  const slide = p.slides.add();
  addText(slide, "为什么金融模型需要“忘记”数据", { left: 56, top: 52, width: 900, height: 58 }, {
    fontSize: 40,
    bold: true,
  });
  const items = [
    ["用户撤回授权", "用户要求删除数据，模型也应降低对这些数据的依赖"],
    ["错误或违规数据", "训练集中发现问题样本后，需要修正模型行为"],
    ["合规与审计", "模型更新应能解释：删了什么、影响是否消除"],
  ];
  items.forEach((item, i) => {
    const top = 164 + i * 132;
    addPanel(slide, { left: 82, top, width: 300, height: 86 }, "#F7F7F7", "#D8DDE5");
    addText(slide, item[0], { left: 108, top: top + 18, width: 250, height: 36 }, {
      fontSize: 28,
      bold: true,
    });
    addText(slide, item[1], { left: 430, top: top + 18, width: 690, height: 48 }, {
      fontSize: 24,
      color: "#374151",
    });
  });
  addText(slide, "关键问题：删除原始记录，不等于删除模型参数中的影响。", {
    left: 108,
    top: 585,
    width: 940,
    height: 42,
  }, { fontSize: 26, bold: true, color: "#1F5F99" });
  addFooter(slide, idx);
  slide.speakerNotes.textFrame.setText("用金融场景解释意义：交易、画像、风险偏好都可能进入模型。逆学习的价值是让模型层面可删除、可审计。");
  slide.speakerNotes.setVisible(true);
}

function sourceSlide(p, idx) {
  const slide = p.slides.add();
  addText(slide, "资料支持这个选题能快速落地", { left: 56, top: 52, width: 900, height: 58 }, {
    fontSize: 40,
    bold: true,
  });
  const rows = [
    ["作业方向", "方向二：针对图像分类模型的机器逆学习"],
    ["已有代码", "CIFAR100 Super20、ResNet18、Amnesiac baseline"],
    ["已推进点", "Bad Teacher、Retain-Aware BT、MIA、ZRF"],
    ["明天汇报", "讲问题、方法、指标、改进和实验计划"],
  ];
  rows.forEach((row, i) => {
    const top = 152 + i * 92;
    addText(slide, row[0], { left: 86, top, width: 210, height: 34 }, {
      fontSize: 26,
      bold: true,
      color: "#111111",
    });
    addPanel(slide, { left: 306, top: top - 8, width: 820, height: 56 }, i === 2 ? "#E8F4FF" : "#F5F5F5", i === 2 ? "#7CB8F7" : "#D8DDE5");
    addText(slide, row[1], { left: 332, top: top + 6, width: 760, height: 28 }, {
      fontSize: 24,
      color: "#1F2937",
    });
  });
  addFooter(slide, idx);
  slide.speakerNotes.textFrame.setText("强调选择依据来自 final/2026隐私保护大作业，不再说隐私计算。说明这个方向已有代码和明确 TODO，适合做成研究报告。");
  slide.speakerNotes.setVisible(true);
}

function problemSlide(p, idx) {
  const slide = p.slides.add();
  addText(slide, "研究问题：接近 Gold model，而不是简单让准确率归零", {
    left: 56,
    top: 52,
    width: 1080,
    height: 58,
  }, { fontSize: 38, bold: true });
  const boxes = [
    ["Original model", "全量数据训练\n含 forget data 影响", 80, 190],
    ["Unlearning", "低成本更新\n删除指定影响", 458, 190],
    ["Gold model", "删除后重训\n理想参照", 836, 190],
  ];
  boxes.forEach(([h, b, l, t]) => {
    addPanel(slide, { left: l, top: t, width: 300, height: 180 }, "#F7F7F7", "#D8DDE5");
    addText(slide, h, { left: l + 22, top: t + 26, width: 256, height: 34 }, {
      fontSize: 26,
      bold: true,
    });
    addText(slide, b, { left: l + 22, top: t + 78, width: 250, height: 72 }, {
      fontSize: 22,
      color: "#4B5563",
    });
  });
  addText(slide, "→", { left: 390, top: 245, width: 52, height: 54 }, { fontSize: 44, bold: true, color: "#3D8DFF", alignment: "center" });
  addText(slide, "→", { left: 768, top: 245, width: 52, height: 54 }, { fontSize: 44, bold: true, color: "#3D8DFF", alignment: "center" });
  addText(slide, "评价标准：forget set 上接近 Gold，retain set 上接近 Original。", {
    left: 128,
    top: 470,
    width: 980,
    height: 42,
  }, { fontSize: 28, bold: true });
  addText(slide, "这能避免把“遗忘”误解成盲目降低所有相关预测能力。", {
    left: 128,
    top: 526,
    width: 840,
    height: 34,
  }, { fontSize: 24, color: "#374151" });
  addFooter(slide, idx);
  slide.speakerNotes.textFrame.setText("这里把问题讲清楚：Gold model 是删除后重训的参照。逆学习不是乱改模型，而是用更低成本逼近这个参照。");
  slide.speakerNotes.setVisible(true);
}

function baselineSlide(p, idx) {
  const slide = p.slides.add();
  addText(slide, "先复现，再改进：两条 baseline 给出参照", { left: 56, top: 52, width: 980, height: 58 }, {
    fontSize: 40,
    bold: true,
  });
  addPanel(slide, { left: 84, top: 156, width: 500, height: 360 }, "#F7F7F7", "#D8DDE5");
  addText(slide, "Amnesiac", { left: 116, top: 190, width: 420, height: 42 }, {
    fontSize: 34,
    bold: true,
  });
  addText(slide, "思路：抵消目标数据带来的训练影响\n作用：课程 notebook 中已有测试入口\n输出：作为第一组可复现实验结果", {
    left: 116,
    top: 260,
    width: 410,
    height: 160,
  }, { fontSize: 24, color: "#374151" });
  addPanel(slide, { left: 696, top: 156, width: 500, height: 360 }, "#E8F4FF", "#7CB8F7");
  addText(slide, "Bad Teacher / Blindspot", { left: 728, top: 190, width: 430, height: 42 }, {
    fontSize: 34,
    bold: true,
  });
  addText(slide, "思路：保留数据模仿原教师\n遗忘数据模仿无效教师\n输出：双教师 KL 蒸馏训练逻辑", {
    left: 728,
    top: 260,
    width: 410,
    height: 160,
  }, { fontSize: 24, color: "#1F2937" });
  addText(slide, "我的方法在 Bad Teacher 上做轻量可解释改进。", {
    left: 170,
    top: 574,
    width: 820,
    height: 40,
  }, { fontSize: 28, bold: true, color: "#1F5F99", alignment: "center" });
  addFooter(slide, idx);
  slide.speakerNotes.textFrame.setText("这页说明工作路径：先复现 Amnesiac 和 Bad Teacher，再补 MIA/ZRF 指标，最后做 Retain-Aware Bad Teacher 改进。");
  slide.speakerNotes.setVisible(true);
}

function methodSlide(p, idx) {
  const slide = p.slides.add();
  addText(slide, "改进方法：Retain-Aware Bad Teacher", { left: 56, top: 52, width: 900, height: 58 }, {
    fontSize: 40,
    bold: true,
  });
  addText(slide, "普通 Bad Teacher 的不足：保留数据只靠软标签蒸馏，可能损伤保留类别边界。", {
    left: 80,
    top: 130,
    width: 1000,
    height: 36,
  }, { fontSize: 25, color: "#374151" });
  const rows = [
    ["Forget set", "模仿随机初始化的无能教师", "加强遗忘"],
    ["Retain set", "模仿原教师，同时加入真实标签 CE", "保留能力"],
  ];
  rows.forEach((row, i) => {
    const top = 224 + i * 138;
    addPanel(slide, { left: 92, top, width: 240, height: 82 }, "#111111", "#111111");
    addText(slide, row[0], { left: 112, top: top + 22, width: 200, height: 32 }, {
      fontSize: 27,
      bold: true,
      color: "#FFFFFF",
      alignment: "center",
    });
    addPanel(slide, { left: 374, top, width: 520, height: 82 }, "#F7F7F7", "#D8DDE5");
    addText(slide, row[1], { left: 400, top: top + 22, width: 470, height: 32 }, {
      fontSize: 24,
      color: "#1F2937",
    });
    addPanel(slide, { left: 934, top, width: 190, height: 82 }, "#E8F4FF", "#7CB8F7");
    addText(slide, row[2], { left: 954, top: top + 22, width: 150, height: 32 }, {
      fontSize: 25,
      bold: true,
      color: "#1F5F99",
      alignment: "center",
    });
  });
  addText(slide, "总损失：L_total = L_KD + λ · CE(student(retain), y_true)。", {
    left: 112,
    top: 544,
    width: 760,
    height: 42,
  }, { fontSize: 28, bold: true });
  addFooter(slide, idx);
  slide.speakerNotes.textFrame.setText("把方法讲成直观逻辑：forget set 仍按 Bad Teacher 去随机化，retain set 额外用真实标签锚定，减少遗忘外溢。");
  slide.speakerNotes.setVisible(true);
}

function experimentSlide(p, idx) {
  const slide = p.slides.add();
  addText(slide, "实验看两件事：是否忘得掉，是否保得住", {
    left: 56,
    top: 52,
    width: 980,
    height: 58,
  }, { fontSize: 40, bold: true });
  const headers = ["对象", "作用", "核心指标"];
  const widths = [260, 520, 320];
  let left = 78;
  headers.forEach((h, i) => {
    addPanel(slide, { left, top: 140, width: widths[i], height: 54 }, "#111111", "#111111");
    addText(slide, h, { left: left + 16, top: 154, width: widths[i] - 32, height: 24 }, {
      fontSize: 20,
      bold: true,
      color: "#FFFFFF",
    });
    left += widths[i];
  });
  const rows = [
    ["Original", "全量训练模型，作为逆学习起点", "retain acc"],
    ["Gold retrain", "删除 forget set 后重训，作为理想参照", "JS distance"],
    ["Amnesiac", "课程 baseline", "forget acc"],
    ["Bad Teacher", "复现方法", "forget + retain"],
    ["Ours", "Retain-Aware Bad Teacher", "综合表现"],
  ];
  rows.forEach((row, r) => {
    let l = 78;
    const top = 194 + r * 66;
    row.forEach((cell, c) => {
      addPanel(slide, { left: l, top, width: widths[c], height: 66 }, r % 2 === 0 ? "#F7F7F7" : "#FFFFFF", "#D8DDE5");
      addText(slide, cell, { left: l + 16, top: top + 18, width: widths[c] - 32, height: 30 }, {
        fontSize: c === 0 ? 20 : 18,
        bold: c === 0,
        color: "#1F2937",
      });
      l += widths[c];
    });
  });
  addText(slide, "新增指标：MIA 衡量隐私泄露，ZRF 衡量与随机教师的接近度。", {
    left: 104,
    top: 555,
    width: 980,
    height: 36,
  }, { fontSize: 24, bold: true, color: "#1F5F99" });
  addFooter(slide, idx);
  slide.speakerNotes.textFrame.setText("这页讲清楚最终报告的实验表怎么来。不要说已有结果，只说将比较哪些对象、用哪些指标评价。");
  slide.speakerNotes.setVisible(true);
}

function closingSlide(p, idx) {
  const slide = p.slides.add();
  addText(slide, "最终交付：方法复现、隐私指标、保留锚定改进", {
    left: 56,
    top: 52,
    width: 1000,
    height: 58,
  }, { fontSize: 40, bold: true });
  const items = [
    ["1", "复现 Amnesiac 与 Bad Teacher", "形成可对比的逆学习 baseline"],
    ["2", "加入 MIA 与 ZRF 指标", "准确率之外评估隐私泄露和遗忘程度"],
    ["3", "验证 Retain-Aware BT", "比较遗忘效果、保留能力和运行成本"],
  ];
  items.forEach((item, i) => {
    const top = 168 + i * 126;
    addPanel(slide, { left: 96, top, width: 86, height: 86 }, "#111111", "#111111");
    addText(slide, item[0], { left: 112, top: top + 18, width: 54, height: 42 }, {
      fontSize: 36,
      bold: true,
      color: "#FFFFFF",
      alignment: "center",
    });
    addText(slide, item[1], { left: 220, top: top + 2, width: 520, height: 36 }, {
      fontSize: 28,
      bold: true,
    });
    addText(slide, item[2], { left: 220, top: top + 46, width: 780, height: 32 }, {
      fontSize: 23,
      color: "#4B5563",
    });
  });
  addText(slide, "明天汇报到研究设计即可；最终大报告补实验表、图和局限性讨论。", {
    left: 100,
    top: 585,
    width: 1010,
    height: 38,
  }, { fontSize: 25, bold: true, color: "#1F5F99" });
  addFooter(slide, idx);
  slide.speakerNotes.textFrame.setText("收束到三个最终交付，让老师知道这个题不是泛泛而谈，而是有代码入口、指标入口和可验证改进点。");
  slide.speakerNotes.setVisible(true);
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  [
    titleSlide,
    motivationSlide,
    sourceSlide,
    problemSlide,
    baselineSlide,
    methodSlide,
    experimentSlide,
    closingSlide,
  ].forEach((fn, i) => fn(p, i + 1));

  for (const [index, slide] of p.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(QA_DIR, `${stem}.png`), await p.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  await writeBlob(path.join(QA_DIR, "deck-montage.webp"), await p.export({ format: "webp", montage: true, scale: 1 }));
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(PPTX_PATH);
  console.log(PPTX_PATH);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

# 课堂汇报材料

本目录基于 `source/机器逆学习_pre.pdf` 生成课堂汇报 PPT。

- `source/机器逆学习_pre.pdf`：用户提供的原始汇报 PDF。
- `slides/机器逆学习_pre.pptx`：由 PDF 每页渲染后生成的 PowerPoint 版本，内容和版式保持 PDF 原稿。
- `scripts/build_pptx_from_pdf_pages.mjs`：从 `rendered_pages/page-*.png` 重新生成 PPTX 的脚本。

如果需要重新生成 PPT：

```powershell
pdftoppm -png -r 144 .\presentation\source\机器逆学习_pre.pdf .\presentation\rendered_pages\page
node .\presentation\scripts\build_pptx_from_pdf_pages.mjs
```


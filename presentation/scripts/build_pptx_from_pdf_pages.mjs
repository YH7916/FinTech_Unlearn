import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const artifactPath =
  "C:/Users/douyuhao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const { Presentation, PresentationFile } = await import(pathToFileURL(artifactPath).href);

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const renderedDir = path.join(root, "rendered_pages");
const outDir = path.join(root, "slides");
const outPath = path.join(outDir, "机器逆学习_pre.pptx");

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function main() {
  const files = (await fs.readdir(renderedDir))
    .filter((name) => /^page-\d+\.png$/i.test(name))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

  if (files.length === 0) {
    throw new Error(`No rendered pages found in ${renderedDir}`);
  }

  await fs.mkdir(outDir, { recursive: true });
  const presentation = Presentation.create({
    slideSize: { width: 1280, height: 720 },
  });

  for (const file of files) {
    const slide = presentation.slides.add();
    slide.background.fill = "#FFFFFF";
    const imagePath = path.join(renderedDir, file);
    slide.images.add({
      blob: await readImageBlob(imagePath),
      contentType: "image/png",
      alt: file,
      fit: "contain",
      position: { left: 0, top: 0, width: 1280, height: 720 },
    });
  }

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(outPath);
  console.log(outPath);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});


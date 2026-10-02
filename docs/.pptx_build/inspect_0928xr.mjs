import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePath = "C:/Users/DELL/Desktop/0928xr - 副本.pptx";
const outputDir = "E:/时刻表/SAC/docs/.pptx_build/inspect_0928xr";
await fs.mkdir(outputDir, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const snapshot = await presentation.inspect({
  kind: "deck,slide,textbox,shape,image,table,chart,notes,layout",
  maxChars: 50000,
});
await fs.writeFile(`${outputDir}/inspect.ndjson`, snapshot.ndjson, "utf8");
const montage = await presentation.export({ format: "png", montage: true, scale: 0.6 });
await fs.writeFile(`${outputDir}/montage.png`, new Uint8Array(await montage.arrayBuffer()));
for (let index = 0; index < presentation.slides.items.length; index += 1) {
  const slide = presentation.slides.getItem(index);
  const preview = await slide.export({ format: "png", scale: 1 });
  await fs.writeFile(`${outputDir}/slide-${index + 1}.png`, new Uint8Array(await preview.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${outputDir}/slide-${index + 1}.layout.json`, await layout.text(), "utf8");
}
console.log(JSON.stringify({
  slideCount: presentation.slides.items.length,
  slideSize: presentation.slideSize,
  masters: presentation.masters.items.length,
  layouts: presentation.layouts.items.length,
  outputDir,
}, null, 2));

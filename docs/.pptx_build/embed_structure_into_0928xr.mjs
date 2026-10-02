import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "C:/Users/DELL/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const sourcePath = "C:/Users/DELL/Desktop/0928xr - 副本.pptx";
const workspaceDir = "E:/时刻表/SAC/docs";
const TMP_DIR = path.join(workspaceDir, ".pptx_build", "embed-0928xr");
const FINAL_PPTX = path.join(workspaceDir, "pptx", "0928xr_已加入课题结构图_无箭头.pptx");
const RUNTIME_PYTHON = "C:/Users/DELL/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const fontFamily = "Microsoft YaHei";

const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")
).href);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const slide = presentation.slides.getItem(1);

const C = {
  navy: "#0B2A53", blue: "#1368AA", paleBlue: "#E8F4FF", paleBlue2: "#D5ECFF",
  teal: "#07867F", paleTeal: "#E8FAF7", tealHeader: "#CFF3EA",
  orange: "#F47B20", paleOrange: "#FFF2E7", brown: "#9A3412",
};

function box(x, y, w, h, fill, stroke, radius=12, width=1.5, name="") {
  return slide.shapes.add({
    geometry: "roundRect", name,
    position: { left:x, top:y, width:w, height:h },
    fill, line: { style:"solid", fill:stroke, width },
    borderRadius: radius,
  });
}

function rect(x, y, w, h, fill, stroke="none", width=0, name="") {
  return slide.shapes.add({
    geometry: "rect", name,
    position: { left:x, top:y, width:w, height:h },
    fill, line: { style:"solid", fill:stroke, width },
  });
}

function text(x, y, w, h, value, size=16, color=C.navy, bold=false,
              align="center", valign="middle", name="") {
  const shape = slide.shapes.add({
    geometry: "textbox", name,
    position: { left:x, top:y, width:w, height:h },
    fill:"none", line:{ style:"solid", fill:"none", width:0 },
  });
  shape.text = value;
  shape.text.style = {
    typeface: fontFamily, fontSize:size, color, bold,
    alignment:align, verticalAlignment:valign,
    autoFit:"shrinkText", wrap:"square",
    insets:{ top:3, right:5, bottom:3, left:5 },
  };
  return shape;
}

function card(x, y, w, h, title, body, theme="blue", titleH=34, titleSize=16, bodySize=13) {
  const stroke = theme === "teal" ? C.teal : C.blue;
  const fill = theme === "teal" ? "#F5FFFC" : "#F7FBFF";
  const titleFill = theme === "teal" ? C.paleTeal : C.paleBlue2;
  const outer = box(x,y,w,h,fill,stroke,10,1.5,title);
  rect(x+1.5,y+1.5,w-3,titleH,titleFill,"none",0,`${title}-header`);
  text(x+5,y+2,w-10,titleH,title,titleSize,C.navy,true,"center","middle",`${title}-title`);
  text(x+7,y+titleH+3,w-14,h-titleH-7,body,bodySize,C.navy,false,"center","middle",`${title}-body`);
  return outer;
}

// Left input block.
const data = box(18,132,145,520,"#F7FBFF",C.blue,12,1.5,"foundation-data");
rect(20,134,141,40,C.paleBlue2,"none",0,"foundation-data-header");
text(25,136,131,36,"基础数据",18,C.navy,true);
text(29,190,123,250,
  "既有时刻表与运行数据\n\n线路坡度、曲线、限速\n\n列车质量与牵引/制动特性",
  13,C.navy,false,"center","middle","foundation-data-body");
const identified = box(29,530,123,94,C.paleOrange,C.orange,9,1.5,"identified-resistance");
text(35,536,111,38,"阻力参数识别结果",13,C.brown,true);
text(35,575,111,34,"Davis A、B、C",14,C.brown,true);

// Upper MAPPO block.
const upper = box(178,98,925,170,"#F7FBFF",C.blue,15,2.5,"upper-MAPPO");
rect(181,101,919,44,C.paleBlue2,"none",0,"upper-MAPPO-header");
text(205,103,870,40,"上层：多列车时刻表规划（MAPPO）",25,C.navy,true);

card(198,158,292,84,"上层观测状态",
  "当前到发时刻、前后车间隔\n供电分区与能量状态、上一站间运行反馈",
  "blue",31,16,12);
card(515,158,245,84,"MAPPO决策模型",
  "分散 Actor    集中 Critic",
  "blue",31,16,13);
card(785,158,292,84,"时刻表调整动作",
  "停站时间调整\n形成新的发车时刻",
  "blue",31,16,13);

// Interaction labels only. The user will add arrows later.
const dispatch = box(315,286,300,58,"#F4FAFF",C.blue,9,1.5,"upper-to-lower-message");
text(325,291,280,48,"下发：目标站间运行时间、停站时间与发车约束",13,C.navy,true,"left");
const feedback = box(650,286,320,58,C.paleOrange,C.orange,9,1.5,"lower-to-upper-message");
text(660,291,300,48,"反馈：实际运行时间、牵引/再生能量、可行性与安全指标",13,C.brown,true,"left");

// Lower SAC block.
const lower = box(178,360,925,310,"#F5FFFC",C.teal,15,2.5,"lower-SAC");
rect(181,363,919,45,C.tealHeader,"none",0,"lower-SAC-header");
text(205,365,870,40,"下层：共享 SAC 站间驾驶控制",25,C.navy,true);

card(195,424,162,122,"线路与列车环境",
  "坡度、曲线、限速\n列车质量\n牵引/制动特性",
  "teal",32,15,12);
card(372,424,155,122,"状态空间",
  "速度、运行时间、位置\n上一时刻执行控制力\n质量、所处站间",
  "teal",32,16,11.5);
card(542,424,165,122,"SAC驾驶策略",
  "Actor\n双 Critic\n经验回放池",
  "blue",32,16,12);
card(722,424,145,122,"连续控制动作",
  "牵引 / 惰行 /\n制动力比例",
  "teal",32,15,12);
card(882,424,202,122,"安全执行层",
  "牵引/制动力、功率\n加速度与限速约束\n终点停车保护",
  "teal",32,16,11.5);

const dynamics = box(370,575,525,70,"#F4FFFC",C.teal,10,1.5,"train-dynamics");
text(382,579,501,28,"列车纵向动力学模型",16,C.navy,true);
text(382,606,501,31,"牵引/制动力、Davis阻力、坡度阻力、曲线阻力",12.5,C.navy,false);

// Right output block.
card(1118,160,145,260,"优化结果",
  "优化时刻表\n（发车时间）\n\n牵引能耗\n\n再生制动利用率",
  "blue",42,18,14);

slide.speakerNotes.textFrame.setText(
  "本页加入可编辑的SAC上下层课题实现结构图。按照用户要求，模块之间未添加箭头或连接线。"
);

const previewBeforeFinal = await slide.export({ format:"png", scale:1 });
await fs.writeFile(path.join(TMP_DIR,"slide-2-preview.png"),new Uint8Array(await previewBeforeFinal.arrayBuffer()));
const layout = await slide.export({format:"layout"});
await fs.writeFile(path.join(TMP_DIR,"slide-2.layout.json"),await layout.text());
const montage = await presentation.export({ format:"png", montage:true, scale:0.6 });
await fs.writeFile(path.join(TMP_DIR,"montage.png"),new Uint8Array(await montage.arrayBuffer()));

const candidatePath = path.join(TMP_DIR,"candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const stagingDir = path.join(TMP_DIR,"finalizer");
await fs.mkdir(stagingDir,{recursive:true});
const result = await finalizePresentation({
  explicitTotalSlideCount:3,
  workspaceDir,
  candidatePath,
  finalPath:FINAL_PPTX,
  pythonExecutable:RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-bullet-geometry","--validate-heading-fit"],
  requiredNativeTableOwnerSlides:[],
  requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:"design",families:["Arial","Calibri","楷体",fontFamily]},
  verifyArtifactToolImport:true,
  receiptPath:path.join(stagingDir,"0928xr_embed.validation.json"),
});
console.log(JSON.stringify({FINAL_PPTX,result},null,2));

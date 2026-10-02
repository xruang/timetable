import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "C:/Users/DELL/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const workspaceDir = "E:/时刻表/SAC/docs";
const TMP_DIR = path.join(workspaceDir, ".pptx_build");
const FINAL_PPTX = path.join(workspaceDir, "pptx", "SAC_上下层课题实现结构图_新版_可编辑_v2.pptx");
const RUNTIME_PYTHON = "C:/Users/DELL/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const fontFamily = "Microsoft YaHei";

const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")
).href);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const p = Presentation.create({ slideSize: { width: 1600, height: 900 } });
const slide = p.slides.add();
slide.background.fill = "#F8FAFC";

const C = {
  navy: "#0B2A53", blue: "#1368AA", blue2: "#2196E6",
  paleBlue: "#E8F4FF", paleBlue2: "#D5ECFF",
  teal: "#07867F", paleTeal: "#E8FAF7", tealHeader: "#CFF3EA",
  orange: "#F47B20", paleOrange: "#FFF2E7", brown: "#9A3412",
  white: "#FFFFFF", gray: "#4B6075", bg: "#F8FAFC",
};

function box(x, y, w, h, fill, stroke, radius=16, width=2, name="") {
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

function text(x, y, w, h, value, size=22, color=C.navy, bold=false,
              align="center", valign="middle", name="") {
  const s = slide.shapes.add({
    geometry: "textbox", name,
    position: { left:x, top:y, width:w, height:h },
    fill:"none", line:{ style:"solid", fill:"none", width:0 },
  });
  s.text = value;
  s.text.style = {
    typeface: fontFamily, fontSize:size, color, bold,
    alignment:align, verticalAlignment:valign,
    autoFit:"shrinkText", wrap:"square",
    insets:{ top:5, right:8, bottom:5, left:8 },
  };
  return s;
}

function card(x, y, w, h, title, body, theme="blue", titleH=44, titleSize=21, bodySize=17) {
  const stroke = theme === "teal" ? C.teal : C.blue;
  const fill = theme === "teal"
    ? "linear(0deg, #FFFFFF 0%, #E8FAF7 100%)"
    : "linear(0deg, #FFFFFF 0%, #E8F4FF 100%)";
  const titleFill = theme === "teal" ? C.paleTeal : C.paleBlue2;
  const outer = box(x,y,w,h,fill,stroke,14,2,title);
  rect(x+2,y+2,w-4,titleH,titleFill,"none",0,`${title}-header`);
  text(x+7,y+2,w-14,titleH,title,titleSize,C.navy,true,"center","middle",`${title}-title`);
  text(x+10,y+titleH+5,w-20,h-titleH-10,body,bodySize,C.navy,false,"center","middle",`${title}-body`);
  return outer;
}

function arrow(x,y,w,h,dir,fill,name) {
  return slide.shapes.add({
    geometry: dir === "down" ? "downArrow" : "upArrow", name,
    position:{left:x,top:y,width:w,height:h}, fill,
    line:{style:"solid",fill,width:1},
  });
}

function rightArrow(x,y,w,h,fill,name) {
  return slide.shapes.add({
    geometry:"rightArrow", name,
    position:{left:x,top:y,width:w,height:h}, fill,
    line:{style:"solid",fill,width:1},
  });
}

function connect(a,b,fromSide,toSide,color=C.blue,kind="straight",width=3) {
  return slide.shapes.connect(a,b,{
    kind, fromSide, toSide,
    line:{style:"solid",fill:color,width},
    tail:{type:"triangle",width:"med",length:"med"},
  });
}

// Upper level: observation -> model -> action.
const upper = box(245,40,1110,226,"linear(0deg, #FFFFFF 0%, #DDEFFF 100%)",C.blue,22,4,"upper-MAPPO");
rect(249,44,1102,66,C.paleBlue2,"none",0,"upper-header");
text(285,48,1030,56,"上层：多列车时刻表规划（MAPPO）",34,C.navy,true);

const upperState = card(
  278,126,354,112,"上层观测状态",
  "当前到发时刻、前后车间隔\n供电分区与能量状态\n上一站间运行反馈",
  "blue",42,21,16
);
const mappoModel = card(
  675,126,250,112,"MAPPO决策模型",
  "分散 Actor\n集中 Critic",
  "blue",42,20,18
);
const upperAction = card(
  968,126,350,112,"时刻表调整动作",
  "停站时间调整\n形成新的发车时刻",
  "blue",42,21,18
);
connect(upperState,mappoModel,"right","left",C.blue,"straight",3);
connect(mappoModel,upperAction,"right","left",C.blue,"straight",3);
rightArrow(637,168,32,24,C.blue,"upper-state-to-model");
rightArrow(930,168,32,24,C.blue,"upper-model-to-action");

// Upper/lower information exchange.
arrow(430,282,58,156,"down","linear(90deg, #53B8F5 0%, #1B88D8 100%)","dispatch-arrow");
const dispatch = box(498,306,326,92,"#F4FAFF",C.blue2,14,2,"dispatch-label");
text(510,313,302,77,"下发\n目标站间运行时间、停站时间与发车约束",18,C.navy,true,"left");

arrow(882,286,58,152,"up","linear(90deg, #FFB15B 0%, #F47B20 100%)","feedback-arrow");
const feedback = box(950,306,334,92,C.paleOrange,C.orange,14,2,"feedback-label");
text(962,313,310,77,"反馈\n实际运行时间、牵引/再生能量、可行性与安全指标",18,C.brown,true,"left");

// Lower level: environment and state -> SAC -> action -> safety -> dynamics -> next state.
const lower = box(220,450,1160,400,"linear(0deg, #FFFFFF 0%, #DDF8F2 100%)",C.teal,22,4,"lower-SAC");
rect(224,454,1152,68,C.tealHeader,"none",0,"lower-header");
text(260,458,1080,58,"下层：共享 SAC 站间驾驶控制",34,C.navy,true);

const lineEnv = card(
  246,550,204,166,"线路与列车环境",
  "坡度、曲线、限速\n列车质量\n牵引/制动特性",
  "teal",44,19,16
);
const lowerState = card(
  474,550,205,166,"状态空间",
  "速度、运行时间、位置\n上一时刻执行控制力\n质量、所处站间",
  "teal",44,21,16
);
const sacModel = card(
  703,550,222,166,"SAC驾驶策略",
  "Actor\n双 Critic\n经验回放池",
  "blue",44,21,17
);
const lowerAction = card(
  949,550,172,166,"连续控制动作",
  "牵引 / 惰行 /\n制动力比例",
  "teal",44,19,17
);
const safety = card(
  1145,550,208,166,"安全执行层",
  "牵引/制动力、功率\n加速度与限速约束\n终点停车保护",
  "teal",44,21,16
);

connect(lineEnv,lowerState,"right","left",C.teal,"straight",3);
connect(lowerState,sacModel,"right","left",C.blue,"straight",3);
connect(sacModel,lowerAction,"right","left",C.blue,"straight",3);
connect(lowerAction,safety,"right","left",C.teal,"straight",3);
rightArrow(454,618,16,26,C.teal,"lower-env-to-state");
rightArrow(683,618,16,26,C.blue,"lower-state-to-policy");
rightArrow(929,618,16,26,C.blue,"lower-policy-to-action");
rightArrow(1125,618,16,26,C.teal,"lower-action-to-safety");

const dynamics = box(485,752,590,72,"#F4FFFC",C.teal,14,2,"train-dynamics");
text(500,756,560,30,"列车纵向动力学模型",20,C.navy,true);
text(500,784,560,32,"牵引/制动力 · Davis阻力 · 坡度阻力 · 曲线阻力",16,C.navy,false);
connect(safety,dynamics,"bottom","right",C.teal,"elbow",3);
connect(dynamics,lowerState,"top","bottom",C.teal,"elbow",3);
connect(lineEnv,dynamics,"bottom","left",C.teal,"elbow",2);

// Foundation data with an explicit identified-resistance input to dynamics.
const data = box(10,315,196,510,"linear(0deg, #FFFFFF 0%, #E8F4FF 100%)",C.blue,16,2,"foundation-data");
rect(12,317,192,54,C.paleBlue2,"none",0,"foundation-header");
text(20,320,176,48,"基础数据",23,C.navy,true);
text(24,392,168,205,
  "既有时刻表与运行数据\n\n线路坡度、曲线、限速\n\n列车质量与牵引/制动特性",
  17,C.navy,false,"center","middle","foundation-body");
const identified = box(23,708,170,92,C.paleOrange,C.orange,12,2,"identified-resistance");
text(30,714,156,35,"阻力参数识别结果",17,C.brown,true);
text(30,748,156,42,"Davis A、B、C",18,C.brown,true);

// Result block.
const result = card(
  1394,330,194,260,"优化结果",
  "优化时刻表\n（发车时间）\n\n牵引能耗\n\n再生制动利用率",
  "blue",52,23,18
);

connect(data,upper,"top","left",C.blue,"elbow",4);
connect(data,lineEnv,"right","left",C.blue,"straight",3);
connect(identified,dynamics,"right","left",C.orange,"straight",4);
text(214,728,252,34,"识别参数输入动力学模型",15,C.brown,true,"center");
rightArrow(194,771,287,20,C.orange,"identified-parameters-to-dynamics");
connect(upper,result,"right","top",C.blue,"elbow",4);
connect(dynamics,result,"right","bottom",C.blue,"elbow",4);

slide.speakerNotes.textFrame.setText(
  "新版结构图依据当前课题实现与用户确认内容重绘。上层采用状态、MAPPO模型、动作的统一层级；下层删除高斯采样等训练细节；阻力参数识别结果作为列车动力学模型输入。所有模块、文字和箭头均为可编辑原生PPT对象。"
);

const preview = await p.export({ slide, format:"png", scale:1 });
await fs.writeFile(path.join(TMP_DIR,"structure-v3-preview.png"),new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({format:"layout"});
await fs.writeFile(path.join(TMP_DIR,"structure-v3.layout.json"),await layout.text());

const candidatePath = path.join(TMP_DIR,"candidate_structure_v3.pptx");
await (await PresentationFile.exportPptx(p)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = { basis:"design", families:[fontFamily] };
const stagingDir = path.join(TMP_DIR,"finalizer-v3b");
await fs.mkdir(stagingDir,{recursive:true});
const resultFinal = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath:FINAL_PPTX,
  pythonExecutable:RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","15240000,8572500","--validate-bullet-geometry","--validate-heading-fit"],
  requiredNativeTableOwnerSlides:[],
  fontPolicy,
  verifyArtifactToolImport:true,
  receiptPath:path.join(stagingDir,"structure_v3b.validation.json"),
});
console.log(JSON.stringify({FINAL_PPTX, resultFinal},null,2));

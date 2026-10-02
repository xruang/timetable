import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "C:/Users/DELL/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const workspaceDir = "E:/时刻表/SAC/docs";
const TMP_DIR = path.join(workspaceDir, ".pptx_build");
const FINAL_PPTX = path.join(workspaceDir, "pptx", "SAC_上下层课题实现结构图_可编辑_v2.pptx");
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
  teal: "#07867F", teal2: "#12A89E", paleTeal: "#E8FAF7",
  orange: "#F47B20", paleOrange: "#FFF2E7",
  white: "#FFFFFF", gray: "#4B6075", light: "#EDF3F8",
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

function card(x, y, w, h, title, body, theme="blue", titleH=48, titleSize=22) {
  const stroke = theme === "teal" ? C.teal : C.blue;
  const fill = theme === "teal" ? "linear(0deg, #FFFFFF 0%, #E8FAF7 100%)" : "linear(0deg, #FFFFFF 0%, #E8F4FF 100%)";
  const titleFill = theme === "teal" ? C.paleTeal : C.paleBlue2;
  const outer = box(x,y,w,h,fill,stroke,14,2,title);
  rect(x+2,y+2,w-4,titleH,titleFill,"none",0,`${title}-header`);
  text(x+7,y+2,w-14,titleH,title,titleSize,C.navy,true,"center","middle",`${title}-title`);
  text(x+12,y+titleH+5,w-24,h-titleH-10,body,18,C.navy,false,"center","middle",`${title}-body`);
  return outer;
}

function arrow(x,y,w,h,dir,fill,name) {
  return slide.shapes.add({
    geometry: dir === "down" ? "downArrow" : "upArrow", name,
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

// Upper MAPPO region
const upper = box(260,52,1080,220,"linear(0deg, #FFFFFF 0%, #DDEFFF 100%)",C.blue,22,4,"upper-MAPPO");
rect(264,56,1072,66,C.paleBlue2,"none",0,"upper-header");
text(300,60,1000,56,"上层：多列车时刻表规划（MAPPO）",34,C.navy,true);

const agents = card(292,137,286,104,"多列车智能体","独立 Actor\n集中 Critic","blue",44);
const upperState = card(605,137,360,104,"时刻表状态","发车 / 到达 / 追踪间隔 / 供电分区","blue",44);
const upperAction = card(990,137,320,104,"上层动作","站间运行时间调整\n停站时间调整","blue",44);

// Central dispatch and feedback
arrow(430,286,60,164,"down","linear(90deg, #53B8F5 0%, #1B88D8 100%)","dispatch-arrow");
const dispatch = box(498,316,310,92,"#F4FAFF",C.blue2,14,2,"dispatch-label");
text(508,324,290,75,"下发\n目标站间运行时间、停站时间、发车约束",19,C.navy,true,"left");

arrow(882,292,60,158,"up","linear(90deg, #FFB15B 0%, #F47B20 100%)","feedback-arrow");
const feedback = box(952,316,310,92,C.paleOrange,C.orange,14,2,"feedback-label");
text(962,324,290,75,"反馈\n实际运行时间、能耗、可行性、安全指标",19,"#9A3412",true,"left");

// Lower SAC region
const lower = box(240,464,1120,374,"linear(0deg, #FFFFFF 0%, #DDF8F2 100%)",C.teal,22,4,"lower-SAC");
rect(244,468,1112,68,"#CFF3EA","none",0,"lower-header");
text(280,472,1040,58,"下层：共享 SAC 站间驾驶控制",34,C.navy,true);

const state16 = card(262,558,195,166,"16维状态","速度、位置、时刻偏差\n载重、坡度、限速\n方向、站间编号","teal",46);
const actor = card(482,558,224,166,"SAC Actor","训练：高斯分布采样 μ, σ\n\n评估：策略均值 tanh(μ)","blue",46);
const action = card(732,558,170,166,"连续动作","牵引 / 制动比例","teal",46);
const safety = card(928,558,210,210,"安全执行层","牵引、功率、制动力\n及加速度硬约束\n\n预测超速时最小安全投影\n终点停车保护","teal",46);
const environment = card(1164,558,176,166,"列车与线路环境","Davis阻力、坡度\n曲线、限速\n纵向动力学","teal",46,19);

const critic = box(480,754,230,60,"#E6F2FF",C.blue,12,2,"critic-replay");
text(490,761,210,46,"双 Critic + 经验回放池",20,C.navy,true);

connect(state16,actor,"right","left",C.blue,"straight",3);
connect(actor,action,"right","left",C.blue,"straight",3);
connect(action,safety,"right","left",C.blue,"straight",3);
connect(safety,environment,"right","left",C.blue,"straight",3);
connect(actor,critic,"bottom","top",C.blue,"elbow",3);
connect(critic,actor,"top","bottom",C.blue,"elbow",3);
text(702,733,205,34,"Q值估计与策略更新",17,C.navy,true,"center");

// Input and result blocks
const data = card(12,342,210,204,"基础数据","真实时刻表\n线路坡度 / 曲线 / 限速\n列车质量与牵引特性","blue",52);
const result = card(1380,346,208,238,"结果","优化时刻表\n速度轨迹\n牵引 / 制动力曲线\n能耗与可行率","blue",52);

connect(data,upper,"top","left",C.blue,"elbow",4);
connect(data,state16,"bottom","left",C.blue,"elbow",4);
connect(upper,result,"right","top",C.blue,"elbow",4);
connect(environment,result,"right","bottom",C.blue,"elbow",4);

slide.speakerNotes.textFrame.setText(
  "根据用户提供的 SAC_上下层课题实现结构图.png 重建。所有模块、文字和箭头均为可编辑原生PPT对象。"
);

const preview = await p.export({ slide, format:"png", scale:1 });
await fs.writeFile(path.join(TMP_DIR,"structure-preview.png"),new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({format:"layout"});
await fs.writeFile(path.join(TMP_DIR,"structure.layout.json"),await layout.text());

const candidatePath = path.join(TMP_DIR,"candidate_structure.pptx");
await (await PresentationFile.exportPptx(p)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = {
  basis:"design",
  families:[fontFamily],
};
const stagingDir = path.join(TMP_DIR,"finalizer");
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
  receiptPath:path.join(stagingDir,"structure_v2.validation.json"),
});
console.log(JSON.stringify({FINAL_PPTX, resultFinal},null,2));

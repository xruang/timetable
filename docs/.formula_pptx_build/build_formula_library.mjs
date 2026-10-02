import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "C:/Users/DELL/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const workspaceDir = "E:/时刻表/SAC/docs";
const TMP_DIR = path.join(workspaceDir, ".formula_pptx_build");
const FINAL_PPTX = path.join(workspaceDir, "pptx", "SAC_MAPPO_建模公式_可复制.pptx");
const RUNTIME_PYTHON = "C:/Users/DELL/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const CN = "Microsoft YaHei";
const MATH = "Cambria Math";

const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")
).href);
await fs.mkdir(TMP_DIR, {recursive:true});
await fs.mkdir(path.dirname(FINAL_PPTX), {recursive:true});

const deck = Presentation.create({slideSize:{width:1600,height:900}});
const C = {
  navy:"#0B2A53", blue:"#1668A9", teal:"#07867F", orange:"#E56B1F",
  gray:"#526579", light:"#EEF4F8", line:"#C8D7E5", white:"#FFFFFF",
  blueFill:"#EAF5FF", tealFill:"#E9F9F6", orangeFill:"#FFF2E8", code:"#F4F7FA",
};

function textbox(slide,x,y,w,h,value,size=22,color=C.navy,bold=false,font=CN,align="left",fill="none",line="none"){
  const s=slide.shapes.add({geometry:"textbox",position:{left:x,top:y,width:w,height:h},fill,
    line:{style:"solid",fill:line,width:line==="none"?0:1.2},borderRadius:fill!=="none"?10:0});
  s.text=value;
  s.text.style={typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:"middle",
    autoFit:"shrinkText",wrap:"square",insets:{top:6,right:12,bottom:6,left:12}};
  return s;
}

function slideBase(title,section,color){
  const s=deck.slides.add();
  s.background.fill="#FBFCFE";
  textbox(s,72,40,1456,62,title,38,C.navy,true,CN,"left");
  textbox(s,72,105,220,34,section,17,color,true,CN,"left");
  const rule=s.shapes.add({geometry:"line",position:{left:72,top:145,width:1456,height:0},fill:"none",
    line:{style:"solid",fill:color,width:3}});
  textbox(s,1450,842,78,28,String(deck.slides.items.length).padStart(2,"0"),14,C.gray,false,CN,"right");
  return s;
}

function formula(slide,y,label,expr,latex,note,color=C.blue,h=150){
  textbox(slide,86,y,250,34,label,20,color,true,CN,"left");
  textbox(slide,86,y+38,1060,h-38,expr,30,C.navy,false,MATH,"center",C.white,C.line);
  textbox(slide,1172,y+38,342,h-38,note,17,C.gray,false,CN,"left",color===C.orange?C.orangeFill:color===C.teal?C.tealFill:C.blueFill,"none");
  textbox(slide,86,y+h+8,1428,43,`LaTeX：${latex}`,15,C.gray,false,MATH,"left",C.code,"none");
}

function notes(slide,paths){
  slide.speakerNotes.textFrame.setText(`公式依据当前项目代码：${paths.join("；")}。公式与LaTeX源码均为可编辑文本。`);
}

// 1 动力学
{
  const s=slideBase("下层列车纵向动力学","SAC下层控制",C.teal);
  formula(s,170,"运动方程",
    "dx/dt = v，      m·dv/dt = F − R(v,x)",
    "\\dot{x}=v,\\quad m\\dot{v}=F-R(v,x)",
    "x为位置，v为速度，F为执行力，m为列车质量。",C.teal,128);
  formula(s,355,"离散状态更新",
    "vₜ₊₁ = max[0, vₜ + (Fₜ − Rₜ)Δt/m]\nxₜ₊₁ = xₜ + (vₜ + vₜ₊₁)Δt/2",
    "v_{t+1}=\\max\\{0,v_t+\\frac{F_t-R_t}{m}\\Delta t\\},\\quad x_{t+1}=x_t+\\frac{v_t+v_{t+1}}{2}\\Delta t",
    "仿真采用匀加速度离散积分。",C.teal,150);
  formula(s,565,"线路总阻力",
    "R(v,x) = A + B(3.6v) + C(3.6v)² + mg·i(x)/1000 + mg·600/[1000R꜀(x)]",
    "R(v,x)=A+B(3.6v)+C(3.6v)^2+mg\\frac{i(x)}{1000}+mg\\frac{600}{1000R_c(x)}",
    "依次包含Davis基本阻力、坡度阻力和曲线阻力。",C.teal,150);
  notes(s,["timetable/train_model.py","timetable/lower_control.py"]);
}

// 2 state/action
{
  const s=slideBase("SAC状态与连续动作","SAC下层控制",C.teal);
  formula(s,170,"16维共享状态",
    "sₜᴸ = [sₜᵐᵒᵗⁱᵒⁿ, sₜˢᶜʰᵉᵈᵘˡᵉ, sₜᵗʳᵃⁱⁿ, sₜˡⁱⁿᵉ, sₜᶜᵒⁿᵗᵉˣᵗ]",
    "s_t^L=[s_t^{motion},s_t^{schedule},s_t^{train},s_t^{line},s_t^{context}]",
    "状态包含运动、时刻、载重、线路条件以及上下行和站间编号。",C.teal,130);
  formula(s,360,"时刻进度偏差",
    "eₜˢᶜʰ = Tᵗᵃʳ · |xₜ/L − t/Tᵗᵃʳ|",
    "e_t^{sch}=T^{tar}\\left|\\frac{x_t}{L}-\\frac{t}{T^{tar}}\\right|",
    "将空间进度和时间进度之差转换为秒级偏差。",C.teal,130);
  formula(s,550,"动作与牵引特性",
    "Fᵗʳₘₐₓ(v) = min(Fₘₐₓ, Pₘₐₓ/max(v,ε))\nFᵖʳᵒᵖ = a·Fᵗʳₘₐₓ(v), a≥0；   Fᵖʳᵒᵖ = a·Fᵇʳₘₐₓ, a<0",
    "F_{tr,max}(v)=\\min(F_{max},\\frac{P_{max}}{\\max(v,\\varepsilon)}),\\quad F^{prop}=\\begin{cases}aF_{tr,max}(v),&a\\ge0\\\\aF_{br,max},&a<0\\end{cases}",
    "actor输出a∈[−1,1]。正值牵引，负值制动。",C.teal,170);
  notes(s,["timetable/lower_control.py","timetable/lower_continuous.py","timetable/train_model.py"]);
}

// 3 safety + reward
{
  const s=slideBase("安全投影与下层奖励","SAC下层控制",C.teal);
  formula(s,170,"最小干预安全投影",
    "Fₜᵉˣᵉᶜ = Π𝒞₍ₛₜ₎(Fₜᵖʳᵒᵖ) = arg min |F − Fₜᵖʳᵒᵖ|，  F∈𝒞(sₜ)",
    "F_t^{exec}=\\Pi_{\\mathcal C(s_t)}(F_t^{prop})=\\arg\\min_{F\\in\\mathcal C(s_t)}|F-F_t^{prop}|",
    "安全集合包含限速、牵引功率、制动力、加速度和终点停车约束。",C.teal,140);
  formula(s,370,"四类奖励",
    "rₜᴸ = rₜᵗⁱᵐᵉ + rₜᵉⁿᵉʳᵍʸ + rₜᶜᵒᵐᶠᵒʳᵗ + rₜˢᵃᶠᵉᵗʸ",
    "r_t^L=r_t^{time}+r_t^{energy}+r_t^{comfort}+r_t^{safety}",
    "准点是主要目标。能耗和舒适性用于同等可行方案之间的优化。",C.teal,130);
  formula(s,560,"奖励分量示例",
    "rₜᵉⁿᵉʳᵍʸ = −wᴱ·ΔEₜ/Eʳᵉᶠ\nrₜᶜᵒᵐᶠᵒʳᵗ = −wᴶ|jₜ|Δt\nrₜˢᵃᶠᵉᵗʸ = −wˢ·𝟙(Fₜᵉˣᵉᶜ ≠ Fₜᵖʳᵒᵖ)",
    "r_t^{energy}=-w_E\\frac{\\Delta E_t}{E_{ref}},\\quad r_t^{comfort}=-w_J|j_t|\\Delta t,\\quad r_t^{safety}=-w_S\\mathbf1(F_t^{exec}\\ne F_t^{prop})",
    "安全层纠正动作后，惩罚归因于actor提出的原始动作。",C.teal,178);
  notes(s,["timetable/lower_control.py"]);
}

// 4 SAC
{
  const s=slideBase("Soft Actor-Critic更新公式","SAC下层控制",C.teal);
  formula(s,165,"策略重参数化",
    "zₜ = μθ(sₜ) + σθ(sₜ)⊙ε，  ε∼𝒩(0,I)，  aₜ = tanh(zₜ)",
    "z_t=\\mu_\\theta(s_t)+\\sigma_\\theta(s_t)\\odot\\epsilon,\\quad \\epsilon\\sim\\mathcal N(0,I),\\quad a_t=\\tanh(z_t)",
    "训练时高斯采样，评估时使用tanh(μ)。",C.teal,125);
  formula(s,345,"双Q目标值",
    "yₜ = rₜ + γ(1−dₜ)[minᵢQφ̄ᵢ(sₜ₊₁,aₜ₊₁) − αlog πθ(aₜ₊₁|sₜ₊₁)]",
    "y_t=r_t+\\gamma(1-d_t)[\\min_{i=1,2}Q_{\\bar\\phi_i}(s_{t+1},a_{t+1})-\\alpha\\log\\pi_\\theta(a_{t+1}|s_{t+1})]",
    "两个目标critic取较小值，降低Q值高估。",C.teal,140);
  formula(s,540,"critic与actor目标",
    "JQ(φᵢ)=𝔼[(Qφᵢ(sₜ,aₜ)−yₜ)²]\nJπ(θ)=𝔼[αlogπθ(aₜ|sₜ)−minᵢQφᵢ(sₜ,aₜ)]",
    "J_Q(\\phi_i)=\\mathbb E[(Q_{\\phi_i}(s_t,a_t)-y_t)^2],\\quad J_\\pi(\\theta)=\\mathbb E[\\alpha\\log\\pi_\\theta(a_t|s_t)-\\min_iQ_{\\phi_i}(s_t,a_t)]",
    "经验池保存屏蔽前的原始提出动作，actor和critic使用同一动作语义。",C.teal,175);
  notes(s,["timetable/lower_continuous.py"]);
}

// 5 upper action and interaction
{
  const s=slideBase("MAPPO上层动作与时刻传播","MAPPO上层规划",C.orange);
  formula(s,170,"上层动作映射",
    "ΔDᵢ,ₖ = Dₘₐₓ·tanh(uᵢ,ₖᵈʷᵉˡˡ)\nΔTᵢ,ₖ = Tₘₐₓ·tanh(uᵢ,ₖʳᵘⁿ)",
    "\\Delta D_{i,k}=D_{max}\\tanh(u_{i,k}^{dwell}),\\quad \\Delta T_{i,k}=T_{max}\\tanh(u_{i,k}^{run})",
    "当前课题设置Tₘₐₓ=0，上层主要优化停站时间。",C.orange,150);
  formula(s,380,"目标时刻与到达时刻",
    "Dᵢ,ₖ = Dᵢ,ₖ⁰ + ΔDᵢ,ₖ，    Tᵢ,ₖᵗᵃʳ = Tᵢ,ₖ⁰ + ΔTᵢ,ₖ\nAᵢ,ₖ₊₁ = Aᵢ,ₖ + Dᵢ,ₖ + Tᵢ,ₖᵃᶜᵗ",
    "D_{i,k}=D_{i,k}^0+\\Delta D_{i,k},\\quad T_{i,k}^{tar}=T_{i,k}^0+\\Delta T_{i,k},\\quad A_{i,k+1}=A_{i,k}+D_{i,k}+T_{i,k}^{act}",
    "实际运行时间由下层策略仿真返回，不直接用目标时间替代。",C.orange,160);
  formula(s,600,"上下层调用关系",
    "(Tᵃᶜᵗ,Eᵗʳ,Eʳᵉᵍ,χ,𝒳) = fθˢᴬᶜ(Tᵗᵃʳ,m,ℓ,d)",
    "(T^{act},E^{tr},E^{reg},\\chi,\\mathcal X)=f_\\theta^{SAC}(T^{tar},m,\\ell,d)",
    "下层向上层返回实际时间、牵引能耗、再生能、可行性和运行轨迹。",C.orange,130);
  notes(s,["timetable/online_hierarchical.py"]);
}

// 6 upper objective
{
  const s=slideBase("多列车节能与安全目标","MAPPO上层规划",C.orange);
  formula(s,170,"再生能匹配",
    "Eₘₐₜcₕ = Σz,b min(Ez,bᵗʳ, Ez,bʳᵉᵍ)，    ρʳᵉᵍ = Eₘₐₜcₕ/(Eʳᵉᵍ+ε)",
    "E_{match}=\\sum_{z,b}\\min(E_{z,b}^{tr},E_{z,b}^{reg}),\\quad \\rho_{reg}=\\frac{E_{match}}{E_{reg}+\\varepsilon}",
    "按供电分区和时间窗统计牵引需求与再生制动能量的重叠。",C.orange,140);
  formula(s,370,"追踪间隔违反量",
    "H = Σi,k max[0, hₘᵢₙ − (Aᵢ₊₁,ₖ − Aᵢ,ₖ)]",
    "H=\\sum_{i,k}\\max\\{0,h_{min}-(A_{i+1,k}-A_{i,k})\\}",
    "当相邻列车间隔低于最小追踪间隔时产生惩罚。",C.orange,130);
  formula(s,560,"上层总体目标",
    "min Jᵁ = wᴱ(Eᵗʳ/Eʳᵉᶠ) − wᴿρʳᵉᵍ + wᴴH + wᶠNⁱⁿᶠᵉᵃˢⁱᵇˡᵉ",
    "\\min J^U=w_E\\frac{E^{tr}}{E^{ref}}-w_R\\rho_{reg}+w_HH+w_FN_{infeasible}",
    "上层在满足可行性和追踪间隔的前提下减少牵引能耗并提高再生能利用率。",C.orange,150);
  notes(s,["timetable/online_hierarchical.py","timetable/mappo.py"]);
}

// 7 MAPPO
{
  const s=slideBase("MAPPO集中训练与分散执行","MAPPO上层规划",C.orange);
  formula(s,165,"CTDE结构",
    "aᵢ,ₖ ∼ πθᵢ(a|oᵢ,ₖ)，      Vφ = Vφ([o₁,ₖ,…,oN,ₖ])",
    "a_{i,k}\\sim\\pi_{\\theta_i}(a|o_{i,k}),\\quad V_\\phi=V_\\phi([o_{1,k},\\ldots,o_{N,k}])",
    "每列车actor只使用局部观测，集中critic使用所有列车状态。",C.orange,125);
  formula(s,345,"GAE优势估计",
    "δₖ = rₖ + γVφ(sₖ₊₁ᴳ) − Vφ(sₖᴳ)\nÂₖ = Σˡ(γλ)ˡδₖ₊ₗ",
    "\\delta_k=r_k+\\gamma V_\\phi(s_{k+1}^G)-V_\\phi(s_k^G),\\quad \\hat A_k=\\sum_l(\\gamma\\lambda)^l\\delta_{k+l}",
    "当前λ=0.95，用于平衡优势估计的偏差与方差。",C.orange,150);
  formula(s,555,"PPO裁剪目标",
    "Lᶜˡⁱᵖ = 𝔼[min(rₖ(θᵢ)Âₖ, clip(rₖ(θᵢ),1−ε,1+ε)Âₖ)]",
    "L^{clip}=\\mathbb E[\\min(r_k(\\theta_i)\\hat A_k,\\operatorname{clip}(r_k(\\theta_i),1-\\epsilon,1+\\epsilon)\\hat A_k)]",
    "当前ε=0.2，限制单次策略更新幅度。",C.orange,145);
  notes(s,["timetable/mappo.py"]);
}

// 8 summary
{
  const s=slideBase("上下层联合建模摘要","分层强化学习框架",C.blue);
  textbox(s,88,178,1424,72,"上层 MAPPO：时刻表规划",28,C.orange,true,CN,"left");
  textbox(s,88,250,1424,100,"uᵁ → {ΔD, ΔT} → Tᵗᵃʳ",36,C.navy,false,MATH,"center",C.orangeFill,"none");
  textbox(s,88,380,1424,72,"下层 SAC：安全节能驾驶",28,C.teal,true,CN,"left");
  textbox(s,88,452,1424,100,"(Tᵃᶜᵗ, E, χ, 𝒳) = fθˢᴬᶜ(Tᵗᵃʳ, m, ℓ, d)",36,C.navy,false,MATH,"center",C.tealFill,"none");
  textbox(s,88,585,1424,72,"联合优化目标",28,C.blue,true,CN,"left");
  textbox(s,88,657,1424,110,"min J = wT·Jᵗⁱᵐᵉ + wE·Jᵉⁿᵉʳᵍʸ + wH·Jʰᵉᵃᵈʷᵃʸ + wS·Jˢᵃᶠᵉᵗʸ",34,C.navy,false,MATH,"center",C.blueFill,"none");
  textbox(s,88,786,1424,40,"LaTeX：\\min J=w_TJ^{time}+w_EJ^{energy}+w_HJ^{headway}+w_SJ^{safety}",15,C.gray,false,MATH,"left",C.code,"none");
  notes(s,["timetable/online_hierarchical.py","timetable/lower_continuous.py","timetable/mappo.py"]);
}

// Previews and candidate
for (let i=0;i<deck.slides.items.length;i++){
  const sl=deck.slides.items[i];
  const png=await deck.export({slide:sl,format:"png",scale:1});
  await fs.writeFile(path.join(TMP_DIR,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
  const layout=await sl.export({format:"layout"});
  await fs.writeFile(path.join(TMP_DIR,`slide-${i+1}.layout.json`),await layout.text());
}

const candidatePath=path.join(TMP_DIR,"candidate_formula_library.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
const stagingDir=path.join(TMP_DIR,"finalizer");
await fs.mkdir(stagingDir,{recursive:true});
const finalized=await finalizePresentation({
  explicitTotalSlideCount:8,
  requiredNativeTableOwnerSlides:[], requiredNativeChartOwnerSlides:[],
  workspaceDir,candidatePath,finalPath:FINAL_PPTX,pythonExecutable:RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(SKILL_DIR,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","15240000,8572500","--validate-bullet-geometry","--validate-heading-fit"],
  fontPolicy:{basis:"design",families:[CN,MATH]},verifyArtifactToolImport:true,
  receiptPath:path.join(stagingDir,"formula_library.validation.json"),
});
console.log(JSON.stringify({FINAL_PPTX,finalized},null,2));

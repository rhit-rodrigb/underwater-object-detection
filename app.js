/* Submerged: static GitHub Pages demo using ONNX Runtime Web.
 * The four model variants and class order come from the team's TrashCan-code
 * archive (demo/predict.py and YOLO validation results).
 */

const MODEL_FILES = {
  baseline: "./models/baseline.onnx",
  builtin: "./models/builtin.onnx",
  weighted: "./models/weighted.onnx",
  offline: "./models/offline.onnx",
};

// This must match data/yolo_instance/data.yaml from the training run.
const CLASS_NAMES = [
  "rov", "plant", "animal_fish", "animal_starfish", "animal_shells",
  "animal_crab", "animal_eel", "animal_etc", "trash_clothing",
  "trash_pipe", "trash_bottle", "trash_bag", "trash_snack_wrapper",
  "trash_can", "trash_cup", "trash_container", "trash_unknown_instance",
  "trash_branch", "trash_wreckage", "trash_tarp", "trash_rope", "trash_net",
];

const IMAGE_SIZE = 640;
const MATCH_IOU = 0.50;
const NMS_IOU = 0.45;

const ui = {
  model: document.querySelector("#modelSelect"),
  source: document.querySelectorAll('input[name="source"]'),
  uploadControls: document.querySelector("#uploadControls"),
  datasetControls: document.querySelector("#datasetControls"),
  upload: document.querySelector("#imageUpload"),
  uploadZone: document.querySelector("#uploadZone"),
  dataset: document.querySelector("#datasetSelect"),
  sample: document.querySelector("#sampleSelect"),
  datasetHint: document.querySelector("#datasetHint"),
  confidence: document.querySelector("#confidenceSlider"),
  confidenceValue: document.querySelector("#confidenceValue"),
  run: document.querySelector("#runButton"),
  status: document.querySelector("#status"),
  inputCanvas: document.querySelector("#inputCanvas"),
  predictionCanvas: document.querySelector("#predictionCanvas"),
  inputPlaceholder: document.querySelector("#inputPlaceholder"),
  predictionPlaceholder: document.querySelector("#predictionPlaceholder"),
  inputCaption: document.querySelector("#inputCaption"),
  outputCaption: document.querySelector("#outputCaption"),
  resultModeTag: document.querySelector("#resultModeTag"),
  body: document.querySelector("#resultsBody"),
  objectCount: document.querySelector("#objectCount"),
};

let samples = [];
let currentImage = null;
let currentSample = null;
let currentSource = "upload";
let loadedModel = null;
let working = false;

function status(message, kind = "normal") {
  ui.status.textContent = message;
  ui.status.dataset.kind = kind;
}

function friendly(name) {
  return String(name ?? "Unknown").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function emptyResults(message = "Run detection to see per-object results.") {
  ui.body.replaceChildren();
  const tr = document.createElement("tr");
  const td = document.createElement("td");
  td.colSpan = 6;
  td.className = "empty-row";
  td.textContent = message;
  tr.append(td);
  ui.body.append(tr);
  ui.objectCount.textContent = "0 predictions";
}

function renderRows(rows, predictionCount) {
  ui.body.replaceChildren();
  ui.objectCount.textContent = `${predictionCount} prediction${predictionCount === 1 ? "" : "s"}`;
  if (!rows.length) {
    emptyResults("No objects detected above the selected confidence threshold.");
    return;
  }
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const tr = document.createElement("tr");
    const columns = [
      String(i + 1),
      row.predicted ? friendly(row.predicted) : "—",
      row.actual ? friendly(row.actual) : "—",
      row.confidence == null ? "—" : `${(row.confidence * 100).toFixed(1)}%`,
      row.iou == null ? "—" : row.iou.toFixed(2),
    ];
    for (const value of columns) {
      const cell = document.createElement("td");
      cell.textContent = value;
      tr.append(cell);
    }
    const statusCell = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `result-badge ${row.kind}`;
    badge.textContent = row.status;
    statusCell.append(badge);
    tr.append(statusCell);
    ui.body.append(tr);
  }
}

function drawCanvas(canvas, image, boxes = []) {
  const ctx = canvas.getContext("2d");
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(image, 0, 0, width, height);
  const fontSize = Math.max(13, Math.round(Math.min(width, height) / 34));
  ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
  ctx.lineWidth = Math.max(2, width / 350);
  for (const box of boxes) {
    const x1 = Math.max(0, box.x1), y1 = Math.max(0, box.y1);
    const x2 = Math.min(width, box.x2), y2 = Math.min(height, box.y2);
    const color = box.color || "#82e8da";
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, Math.max(0, x2 - x1), Math.max(0, y2 - y1));
    const text = box.label || "Object";
    const padding = 5;
    const textWidth = ctx.measureText(text).width;
    const labelY = y1 > fontSize + 13 ? y1 - fontSize - 12 : y1 + 1;
    ctx.fillStyle = color;
    ctx.fillRect(x1, labelY, Math.min(textWidth + padding * 2, width - x1), fontSize + 11);
    ctx.fillStyle = "#052331";
    ctx.fillText(text, x1 + padding, labelY + fontSize + 1, Math.max(10, width - x1 - padding));
  }
  canvas.classList.add("ready");
}

function groundTruthBoxes(sample) {
  return (sample?.annotations || []).map(a => ({
    x1: a.bbox[0], y1: a.bbox[1],
    x2: a.bbox[0] + a.bbox[2], y2: a.bbox[1] + a.bbox[3],
    label: friendly(a.class), color: "#8de5fa",
  }));
}

function setCurrentImage(image, sample = null, groundTruthImage = null) {
  currentImage = image;
  currentSample = sample;
  ui.inputPlaceholder.hidden = true;
  ui.predictionPlaceholder.hidden = true;
  drawCanvas(ui.inputCanvas, groundTruthImage || image,
    groundTruthImage ? [] : groundTruthBoxes(sample));
  drawCanvas(ui.predictionCanvas, image);
  ui.inputCaption.textContent = groundTruthImage
    ? "Ground truth (team-provided)"
    : sample?.annotations?.length ? "Ground truth (blue)" : "Uploaded image";
  ui.outputCaption.textContent = "Ready to detect";
  ui.resultModeTag.textContent = sample ? "DATASET IMAGE" : "PERSONAL UPLOAD";
  ui.run.disabled = false;
  emptyResults();
  status(sample?.previews
    ? "Image ready. Click Run detection to view the team's saved result; live inference needs ONNX files."
    : "Image ready. Click Run detection to load the selected model.");
}

function clearImage(message = "Choose an image to begin.") {
  currentImage = null;
  currentSample = null;
  ui.run.disabled = true;
  for (const [canvas, placeholder] of [
    [ui.inputCanvas, ui.inputPlaceholder],
    [ui.predictionCanvas, ui.predictionPlaceholder],
  ]) {
    canvas.classList.remove("ready");
    placeholder.hidden = false;
  }
  ui.resultModeTag.textContent = "WAITING FOR IMAGE";
  ui.inputCaption.textContent = "Original";
  ui.outputCaption.textContent = "Awaiting inference";
  emptyResults();
  status(message);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load the selected image."));
    image.src = url;
  });
}

async function useUploadedFile(file) {
  if (!file) return;
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) {
    status("Please select a JPG, PNG, or WebP image.", "error");
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    status("Please select an image under 25 MB.", "error");
    return;
  }
  const url = URL.createObjectURL(file);
  try {
    const image = await loadImage(url);
    if (currentSource === "upload") setCurrentImage(image);
  } catch (error) {
    status(error.message, "error");
  } finally {
    URL.revokeObjectURL(url);
  }
}

function matchingSamples() {
  return samples.filter(s => s.dataset === ui.dataset.value &&
    (s.split === "val" || s.split === "curated"));
}

function refreshSamples() {
  const chosen = matchingSamples();
  ui.sample.replaceChildren();
  if (!chosen.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No samples added yet";
    ui.sample.append(opt);
    ui.sample.disabled = true;
    if (currentSource === "dataset") clearImage(
      ui.dataset.value === "material"
        ? "No Material examples were provided yet. Select Instance or add Material samples."
        : "Add example images with tools/prepare_examples.py first.");
  } else {
    ui.sample.disabled = false;
    for (const sample of chosen) {
      const opt = document.createElement("option");
      opt.value = sample.id;
      opt.textContent = sample.label || sample.id;
      ui.sample.append(opt);
    }
    if (currentSource === "dataset") void chooseSample();
  }
  ui.datasetHint.textContent = ui.dataset.value === "material"
    ? "The current models predict Instance labels. Material ground truth cannot be used to grade them."
    : "Curated examples include saved predictions. Per-object grading requires original COCO annotations.";
}

async function chooseSample() {
  const sample = matchingSamples().find(s => s.id === ui.sample.value);
  if (!sample) return;
  try {
    const [image, reference] = await Promise.all([
      loadImage(new URL(sample.image, document.baseURI).href),
      sample.groundTruthImage
        ? loadImage(new URL(sample.groundTruthImage, document.baseURI).href)
        : Promise.resolve(null),
    ]);
    if (currentSource === "dataset" && sample.id === ui.sample.value) {
      setCurrentImage(image, sample, reference);
    }
  } catch (error) {
    status(`Sample unavailable: ${error.message}`, "error");
  }
}

async function loadSamples() {
  try {
    const response = await fetch("./samples/manifest.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    samples = Array.isArray(data) ? data : [];
  } catch {
    samples = [];
  }
  refreshSamples();
}

function prepareInput(image) {
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const scale = Math.min(IMAGE_SIZE / sourceWidth, IMAGE_SIZE / sourceHeight);
  const drawWidth = Math.round(sourceWidth * scale);
  const drawHeight = Math.round(sourceHeight * scale);
  const padX = Math.round((IMAGE_SIZE - drawWidth) / 2);
  const padY = Math.round((IMAGE_SIZE - drawHeight) / 2);
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "rgb(114,114,114)";
  context.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  context.drawImage(image, padX, padY, drawWidth, drawHeight);
  const rgba = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  const pixels = IMAGE_SIZE * IMAGE_SIZE;
  const channels = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i++) {
    channels[i] = rgba[i * 4] / 255;
    channels[i + pixels] = rgba[i * 4 + 1] / 255;
    channels[i + pixels * 2] = rgba[i * 4 + 2] / 255;
  }
  return {
    tensor: new ort.Tensor("float32", channels, [1, 3, IMAGE_SIZE, IMAGE_SIZE]),
    scale, padX, padY, width: sourceWidth, height: sourceHeight,
  };
}

function area(box) {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

function boxIoU(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = area(a) + area(b) - intersection;
  return union > 0 ? intersection / union : 0;
}

function nonMaximumSuppression(boxes, threshold = NMS_IOU) {
  const ordered = boxes.sort((a, b) => b.confidence - a.confidence).slice(0, 600);
  const keep = [];
  for (const box of ordered) {
    if (keep.some(prev => prev.classId === box.classId && boxIoU(prev, box) > threshold)) continue;
    keep.push(box);
    if (keep.length >= 100) break;
  }
  return keep;
}

function decodeOutput(output, threshold, transform) {
  const dimensions = output.dims.map(Number);
  const data = output.data;
  const found = [];
  const classCount = CLASS_NAMES.length;
  const channelCount = 4 + classCount;

  if ((dimensions.length === 3 && dimensions[2] === 6) ||
      (dimensions.length === 2 && dimensions[1] === 6)) {
    // Export with nms=True: rows are [x1, y1, x2, y2, score, class].
    const count = dimensions.length === 3 ? dimensions[1] : dimensions[0];
    for (let i = 0; i < count; i++) {
      const offset = i * 6;
      const score = Number(data[offset + 4]);
      const classId = Math.round(Number(data[offset + 5]));
      if (!(score >= threshold) || classId < 0 || classId >= classCount) continue;
      found.push({ x1:data[offset], y1:data[offset+1], x2:data[offset+2],
        y2:data[offset+3], confidence:score, classId });
    }
  } else if (dimensions.length === 3 && (dimensions[1] === channelCount || dimensions[2] === channelCount)) {
    // Standard raw YOLO11 detect output: [1, 4+C, N] or [1, N, 4+C].
    const channelsFirst = dimensions[1] === channelCount;
    const count = channelsFirst ? dimensions[2] : dimensions[1];
    const at = (channel, index) => channelsFirst
      ? Number(data[channel * count + index])
      : Number(data[index * channelCount + channel]);
    for (let i = 0; i < count; i++) {
      let bestScore = 0, bestClass = -1;
      for (let cls = 0; cls < classCount; cls++) {
        const score = at(4 + cls, i);
        if (score > bestScore) { bestScore = score; bestClass = cls; }
      }
      if (bestScore < threshold) continue;
      const cx = at(0,i), cy = at(1,i), w = at(2,i), h = at(3,i);
      found.push({ x1:cx-w/2, y1:cy-h/2, x2:cx+w/2, y2:cy+h/2,
        confidence:bestScore, classId:bestClass });
    }
  } else {
    throw new Error(
      `Unsupported model output shape [${dimensions.join(", ")}]. ` +
      "Please export an Ultralytics YOLO11 detection model at 640 px and check its ONNX output format."
    );
  }

  const selected = nonMaximumSuppression(found);
  return selected.map(box => ({
    ...box,
    x1: Math.max(0, Math.min(transform.width, (box.x1 - transform.padX) / transform.scale)),
    y1: Math.max(0, Math.min(transform.height, (box.y1 - transform.padY) / transform.scale)),
    x2: Math.max(0, Math.min(transform.width, (box.x2 - transform.padX) / transform.scale)),
    y2: Math.max(0, Math.min(transform.height, (box.y2 - transform.padY) / transform.scale)),
    className: CLASS_NAMES[box.classId],
  })).filter(box => area(box) > 1);
}

function evaluateDetections(detections, sample) {
  const comparisonEnabled = sample?.dataset === "instance" &&
    Array.isArray(sample.annotations) && sample.annotations.length > 0;
  if (!comparisonEnabled) {
    const rows = detections.map(d => ({
      predicted:d.className, actual:null, confidence:d.confidence, iou:null,
      status:sample ? "Not comparable" : "Not available", kind:"unknown",
    }));
    return { rows, annotations:detections.map(d => ({
      ...d, label:`${friendly(d.className)} ${(d.confidence*100).toFixed(0)}%`,color:"#7ee4d5",
    })) };
  }

  const gt = groundTruthBoxes(sample).map((box, i) => ({...box, className:sample.annotations[i].class}));
  const candidates = [];
  for (let p=0; p<detections.length; p++) {
    for (let g=0; g<gt.length; g++) {
      const overlap = boxIoU(detections[p],gt[g]);
      if (overlap >= MATCH_IOU) candidates.push({p,g,overlap});
    }
  }
  candidates.sort((a,b) => b.overlap - a.overlap);
  const usedP = new Set(), usedG = new Set(), pairs = [];
  for (const pair of candidates) {
    if (usedP.has(pair.p) || usedG.has(pair.g)) continue;
    usedP.add(pair.p); usedG.add(pair.g); pairs.push(pair);
  }

  const rows = [];
  const annotations = detections.map(d => ({
    ...d, color:"#ff9e96", label:`${friendly(d.className)} ${(d.confidence*100).toFixed(0)}%`,
  }));
  for (const pair of pairs) {
    const pred = detections[pair.p], actual = gt[pair.g];
    const correct = pred.className === actual.className;
    rows.push({ predicted:pred.className, actual:actual.className,
      confidence:pred.confidence, iou:pair.overlap,
      status:correct ? "Correct" : "Wrong class", kind:correct ? "correct":"incorrect" });
    annotations[pair.p].color = correct ? "#7ee4ba" : "#fbc67a";
  }
  for (let g=0; g<gt.length; g++) {
    if (!usedG.has(g)) rows.push({ predicted:null, actual:gt[g].className,
      confidence:null, iou:null,status:"Missed",kind:"missed" });
  }
  for (let p=0; p<detections.length; p++) {
    if (!usedP.has(p)) rows.push({ predicted:detections[p].className, actual:null,
      confidence:detections[p].confidence, iou:null,status:"Extra detection",kind:"extra" });
  }
  return { rows, annotations };
}

async function getModelSession(key) {
  if (loadedModel?.key === key) return loadedModel.session;
  const url = new URL(MODEL_FILES[key], document.baseURI).href;
  const response = await fetch(url, { method:"HEAD" });
  if (!response.ok) throw new Error(
    `Model file is missing (${MODEL_FILES[key]}). Export the team's trained ${key} checkpoint to ONNX first.`
  );
  if (!window.ort) throw new Error("ONNX Runtime did not load. Check your internet connection or script blocker.");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
  const session = await ort.InferenceSession.create(url, { executionProviders:["wasm"], graphOptimizationLevel:"all" });
  if (loadedModel?.session?.release) await loadedModel.session.release();
  loadedModel = { key, session };
  return session;
}

async function showSavedPrediction(sample, modelKey) {
  const previewPath = sample?.previews?.[modelKey];
  if (!previewPath) return false;
  const annotated = await loadImage(new URL(previewPath, document.baseURI).href);
  if (currentSample !== sample || ui.model.value !== modelKey) return false;
  drawCanvas(ui.predictionCanvas, annotated);
  ui.outputCaption.textContent = "Saved prediction (provided by team)";
  ui.resultModeTag.textContent = "SAVED RESULT";
  emptyResults(
    "This is a saved prediction image, not a new model run. The ZIP contains annotated JPGs but no machine-readable per-object detections, so a correctness table cannot be calculated yet."
  );
  status(
    "Showing the team's saved prediction. Confidence-slider changes do not affect saved JPGs; live inference needs the ONNX file.",
    "success"
  );
  return true;
}

async function runDetection() {
  if (!currentImage || working) return;
  working = true;
  ui.run.disabled = true;
  ui.run.firstChild.textContent = "Running model ";
  status("Loading model and running object detection. First load may take a while…");
  try {
    const selectedModel = ui.model.value;
    let session;
    try {
      session = await getModelSession(selectedModel);
    } catch (error) {
      if (String(error.message).includes("Model file is missing") &&
          await showSavedPrediction(currentSample, selectedModel)) return;
      throw error;
    }
    const input = prepareInput(currentImage);
    const output = await session.run({ [session.inputNames[0]]:input.tensor });
    const detections = decodeOutput(output[session.outputNames[0]], Number(ui.confidence.value)/100, input);
    const evaluation = evaluateDetections(detections,currentSample);
    drawCanvas(ui.predictionCanvas,currentImage,evaluation.annotations);
    renderRows(evaluation.rows,detections.length);
    ui.outputCaption.textContent = "Predictions (colored boxes)";
    ui.resultModeTag.textContent = `${detections.length} DETECTION${detections.length===1?"":"S"}`;
    status(`Finished: ${detections.length} object${detections.length===1?"":"s"} detected.`,"success");
  } catch (error) {
    console.error(error);
    status(error.message || "Model inference failed.","error");
    ui.outputCaption.textContent = "Inference unavailable";
    emptyResults("No inference results. Read the message under the Detect button.");
  } finally {
    working = false;
    ui.run.disabled = !currentImage;
    ui.run.firstChild.textContent = "Run detection ";
  }
}

ui.upload.addEventListener("change", event => void useUploadedFile(event.target.files[0]));
ui.uploadZone.addEventListener("dragover", event => { event.preventDefault();ui.uploadZone.classList.add("drag-over"); });
ui.uploadZone.addEventListener("dragleave", () => ui.uploadZone.classList.remove("drag-over"));
ui.uploadZone.addEventListener("drop", event => {
  event.preventDefault(); ui.uploadZone.classList.remove("drag-over");
  const file = event.dataTransfer?.files?.[0];
  if (file) void useUploadedFile(file);
});
for (const input of ui.source) input.addEventListener("change", () => {
  currentSource = document.querySelector('input[name="source"]:checked').value;
  ui.uploadControls.hidden = currentSource !== "upload";
  ui.datasetControls.hidden = currentSource !== "dataset";
  clearImage();
  if (currentSource === "dataset") refreshSamples();
  else if (ui.upload.files[0]) void useUploadedFile(ui.upload.files[0]);
});
ui.dataset.addEventListener("change",refreshSamples);
ui.sample.addEventListener("change",() => void chooseSample());
ui.confidence.addEventListener("input", () => ui.confidenceValue.textContent = `${ui.confidence.value}%`);
ui.model.addEventListener("change", () => {
  if (!currentImage) return;
  drawCanvas(ui.predictionCanvas, currentImage);
  ui.outputCaption.textContent = "Awaiting selected model";
  ui.resultModeTag.textContent = "MODEL CHANGED";
  emptyResults();
  status("Model changed. Click Run detection to view its result.");
});
ui.run.addEventListener("click",() => void runDetection());

void loadSamples();

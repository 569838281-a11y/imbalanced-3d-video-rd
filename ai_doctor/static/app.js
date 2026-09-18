(() => {
  const form = document.getElementById("upload-form");
  const input = document.getElementById("file-input");
  const dropInner = form.querySelector(".drop-inner");
  const fileList = document.getElementById("file-list");
  const submitBtn = document.getElementById("submit-btn");
  const statusEl = document.getElementById("status");
  const resultStack = document.getElementById("result-stack");
  const diseaseLabel = document.getElementById("disease-label");
  const riskTier = document.getElementById("risk-tier");
  const summaryEl = document.getElementById("summary");
  const pipelineEl = document.getElementById("pipeline");
  const metricsEl = document.getElementById("metrics");
  const evidenceGrid = document.getElementById("evidence-grid");
  const findingsEl = document.getElementById("findings");
  const impressionEl = document.getElementById("impression");
  const adviceList = document.getElementById("advice-list");
  const disclaimerEl = document.getElementById("disclaimer");

  let files = [];

  function setStatus(text, isError = false) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("is-error", !!isError);
  }

  function refreshFileList() {
    if (!files.length) {
      fileList.hidden = true;
      fileList.textContent = "";
      submitBtn.disabled = true;
      return;
    }
    fileList.hidden = false;
    fileList.textContent = files.map((f) => f.name).join(" · ");
    submitBtn.disabled = false;
  }

  function takeFiles(fileListLike) {
    files = Array.from(fileListLike || []);
    refreshFileList();
    resultStack.hidden = true;
    setStatus(files.length ? `已选择 ${files.length} 个文件` : "");
  }

  dropInner.addEventListener("click", () => input.click());
  form.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => takeFiles(input.files));

  ["dragenter", "dragover"].forEach((evt) => {
    form.addEventListener(evt, (e) => {
      e.preventDefault();
      form.classList.add("is-drag");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    form.addEventListener(evt, (e) => {
      e.preventDefault();
      form.classList.remove("is-drag");
    });
  });
  form.addEventListener("drop", (e) => takeFiles(e.dataTransfer.files));

  function toneClass(code) {
    if (code === "non_rd") return "is-ok";
    if (code && code.includes("intact")) return "is-warn";
    return "is-danger";
  }

  function pct(v) {
    if (v == null) return "—";
    return `${(Number(v) * 100).toFixed(1)}%`;
  }

  function tierLabel(t) {
    const map = { low: "低风险", mid: "中风险", high: "高风险" };
    return map[t] || t || "—";
  }

  function subtypeProbLabel(result) {
    if (!result.stage3_ran || result.subtype_probability == null) return "—";
    const pos = result.subtype_positive_class || "正类";
    return `${pct(result.subtype_probability)}（P(${pos})）`;
  }

  function renderResult(result) {
    diseaseLabel.textContent = result.disease_label_zh;
    diseaseLabel.className = `disease ${toneClass(result.disease_code)}`;
    riskTier.textContent = `风险分层 · ${tierLabel(result.risk_tier)} (${result.risk_tier || "—"})`;
    summaryEl.textContent = result.summary_zh;

    pipelineEl.innerHTML = "";
    (result.pipeline || []).forEach((step) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="stage">阶段 ${step.stage}</span><span class="name">${step.name}</span><span class="out">${step.result}</span>`;
      pipelineEl.appendChild(li);
    });

    const rows = [
      ["一阶段 RD 概率", pct(result.rd_probability)],
      ["二阶段黄斑完整概率", pct(result.macula_intact_probability)],
      ["三阶段亚型", result.subtype_label_zh || "—"],
      ["三阶段正类概率", subtypeProbLabel(result)],
      ["病例 ID", result.case_result?.case_id || "—"],
      ["来源", result.source || "—"],
    ];
    metricsEl.innerHTML = rows
      .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
      .join("");

    evidenceGrid.innerHTML = "";
    const overlays = result.evidence_overlays || [];
    if (!overlays.length) {
      evidenceGrid.innerHTML = `<p class="panel-note">暂无依据预览（CAM 不可用或未启用）。</p>`;
    } else {
      overlays.forEach((ev) => {
        const card = document.createElement("figure");
        card.className = "evidence-card";
        card.innerHTML = `
          <img src="${ev.png}" alt="依据 ${ev.id} 帧 ${ev.frame_idx}" />
          <figcaption class="cap"><strong>${ev.id}</strong> · 帧 ${ev.frame_idx} · ${ev.source}<br/>${ev.text}</figcaption>
        `;
        evidenceGrid.appendChild(card);
      });
    }

    findingsEl.textContent = result.findings || "";
    impressionEl.textContent = result.impression || "";
    adviceList.innerHTML = "";
    (result.advice || []).forEach((a) => {
      const li = document.createElement("li");
      li.className = a.level || "";
      li.textContent = a.text;
      adviceList.appendChild(li);
    });
    disclaimerEl.textContent = result.disclaimer_zh || "";

    resultStack.hidden = false;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!files.length) return;

    const body = new FormData();
    files.forEach((f) => body.append("files", f, f.name));

    submitBtn.disabled = true;
    setStatus("正在分析：三级级联 + Grad-CAM 依据引擎……");

    try {
      const res = await fetch("/api/diagnose", { method: "POST", body });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.detail || "诊断失败");
      }
      renderResult(data.result);
      setStatus("诊断完成（含结论卡 / 依据区 / 报告）");
    } catch (err) {
      setStatus(err.message || String(err), true);
      resultStack.hidden = true;
    } finally {
      submitBtn.disabled = !files.length;
    }
  });
})();

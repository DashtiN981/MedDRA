const els = {
  input: document.getElementById("ae-input"),
  predictBtn: document.getElementById("predict-btn"),
  exampleGrid: document.getElementById("example-grid"),
  results: document.getElementById("results"),
  status: document.getElementById("status-pill"),
  lltTerm: document.getElementById("llt-term"),
  lltCode: document.getElementById("llt-code"),
  ptTerm: document.getElementById("pt-term"),
  ptCode: document.getElementById("pt-code"),
  socTerm: document.getElementById("soc-term"),
  socCode: document.getElementById("soc-code"),
  latency: document.getElementById("latency"),
  fallback: document.getElementById("fallback"),
  top5: document.getElementById("top5-inline"),
  candidateBody: document.getElementById("candidate-body"),
  hierarchy: document.getElementById("hierarchy"),
};

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = `status ${cls}`;
}

function valOrDash(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

function buildHierarchy(data) {
  const p = data.prediction || {};
  const socAll = (p.soc_codes_all || []).join(", ") || "-";
  const missing = p.primary_soc_missing ? "<span class='warn'>yes</span>" : "<span class='ok'>no</span>";

  return [
    "$ meddra-hierarchy --inspect",
    "",
    `AE_TEXT      : ${data.input_text}`,
    `LLT_TERM     : ${valOrDash(p.llt_term)}`,
    `LLT_CODE     : ${valOrDash(p.llt_code)}`,
    "    |",
    `    +-- PT_TERM   : ${valOrDash(p.pt_term)}`,
    `    +-- PT_CODE   : ${valOrDash(p.pt_code)}`,
    "            |",
    `            +-- SOC_TERM        : ${valOrDash(p.soc_term)}`,
    `            +-- SOC_CODE        : ${valOrDash(p.soc_code)}`,
    `            +-- ALL_SOC_CODES   : ${socAll}`,
    `            +-- PRIMARY_MISSING : ${missing}`,
  ].join("\n");
}

function renderCandidates(rows) {
  els.candidateBody.innerHTML = "";

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${valOrDash(row.rank)}</td>
      <td>${valOrDash(row.llt_term)}</td>
      <td>${valOrDash(row.llt_code)}</td>
      <td class="score-cell">
        <div class="score-track"><div class="score-fill" data-score="${row.score}"></div></div>
        <div class="score-label">${row.score}%</div>
      </td>
      <td>${valOrDash(row.pt_code)}</td>
      <td>${valOrDash(row.soc_code)}</td>
    `;
    els.candidateBody.appendChild(tr);
  });

  requestAnimationFrame(() => {
    document.querySelectorAll(".score-fill").forEach((bar) => {
      bar.style.width = `${bar.dataset.score}%`;
    });
  });
}

function renderPrediction(data) {
  const p = data.prediction || {};
  els.lltTerm.textContent = valOrDash(p.llt_term);
  els.lltCode.textContent = valOrDash(p.llt_code);
  els.ptTerm.textContent = valOrDash(p.pt_term);
  els.ptCode.textContent = valOrDash(p.pt_code);
  els.socTerm.textContent = valOrDash(p.soc_term);
  els.socCode.textContent = valOrDash(p.soc_code);
  els.latency.textContent = valOrDash(data.debug?.latency_ms);
  els.fallback.textContent = data.debug?.used_fallback ? "Yes" : "No";
  els.top5.textContent = (data.top5_preview || []).join(" | ") || "-";
  renderCandidates(data.candidates || []);
  els.hierarchy.innerHTML = buildHierarchy(data);
  els.results.classList.remove("hidden");
}

async function predict() {
  const text = els.input.value.trim();
  if (!text) {
    setStatus("No Input", "error");
    return;
  }

  setStatus("Predicting", "loading");
  els.predictBtn.disabled = true;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, top_k: 100 }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    renderPrediction(data);
    setStatus("Done", "done");
  } catch (err) {
    setStatus("Error", "error");
    els.hierarchy.innerHTML = `$ error\n${err.message || String(err)}`;
    els.results.classList.remove("hidden");
  } finally {
    els.predictBtn.disabled = false;
  }
}

els.predictBtn.addEventListener("click", predict);
els.input.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    predict();
  }
});

els.exampleGrid.addEventListener("click", (event) => {
  const target = event.target.closest(".example-card");
  if (!target) return;
  els.input.value = target.dataset.example || "";
  els.input.focus();
});

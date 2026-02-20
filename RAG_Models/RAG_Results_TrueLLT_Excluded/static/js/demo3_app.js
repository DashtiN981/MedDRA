const state = {
  jobId: null,
  runPollTimer: null,
};

const el = {
  status: document.getElementById("status-pill"),
  fileInput: document.getElementById("file-input"),
  aeColumn: document.getElementById("ae-column"),
  runBtn: document.getElementById("run-btn"),
  runProgressWrap: document.getElementById("run-progress-wrap"),
  runProgressText: document.getElementById("run-progress-text"),
  runElapsed: document.getElementById("run-elapsed"),
  runProgressFill: document.getElementById("run-progress-fill"),
  fileMeta: document.getElementById("file-meta"),
  previewHead: document.getElementById("preview-head"),
  previewBody: document.getElementById("preview-body"),

  reviewSection: document.getElementById("review-section"),
  progress: document.getElementById("progress-label"),
  aeText: document.getElementById("ae-text"),
  predLlt: document.getElementById("pred-llt"),
  predLltCode: document.getElementById("pred-llt-code"),
  predPt: document.getElementById("pred-pt"),
  predPtCode: document.getElementById("pred-pt-code"),
  predSoc: document.getElementById("pred-soc"),
  predSocCode: document.getElementById("pred-soc-code"),
  latency: document.getElementById("latency"),
  fallback: document.getElementById("fallback"),
  decision: document.getElementById("decision"),
  top5Select: document.getElementById("top5-select"),
  manualTerm: document.getElementById("manual-term"),
  hierarchy: document.getElementById("hierarchy-box"),
  candidateBody: document.getElementById("candidate-body"),
  reviewBody: document.getElementById("review-body"),

  prevBtn: document.getElementById("prev-btn"),
  nextBtn: document.getElementById("next-btn"),
  acceptBtn: document.getElementById("accept-btn"),
  rejectBtn: document.getElementById("reject-btn"),
  exportBtn: document.getElementById("export-btn"),
};

function setStatus(text, cls = "") {
  el.status.textContent = text;
  el.status.className = cls ? `status ${cls}` : "status";
}

function setRunProgress(percent, text, elapsedMs) {
  const safePct = Math.max(0, Math.min(100, Number(percent || 0)));
  el.runProgressWrap.classList.remove("hidden");
  el.runProgressFill.style.width = `${safePct}%`;
  el.runProgressText.textContent = text;
  el.runElapsed.textContent = `${(Number(elapsedMs || 0) / 1000).toFixed(1)}s`;
}

function stopRunPolling() {
  if (state.runPollTimer) {
    clearTimeout(state.runPollTimer);
    state.runPollTimer = null;
  }
}

async function pollRunStatus() {
  if (!state.jobId) return;

  try {
    const status = await api(`/api/run_status/${state.jobId}`);
    const total = Number(status.total_rows || 0);
    const done = Number(status.processed_rows || 0);
    const pct = Number(status.progress_pct || 0);
    const text = total > 0 ? `Processing ${done}/${total} rows` : "Preparing predictions...";
    setRunProgress(pct, text, status.elapsed_ms || 0);

    if (status.status === "completed") {
      setRunProgress(100, `Completed ${done}/${total} rows`, status.elapsed_ms || 0);
      if (status.result) {
        renderPayload(status.result);
      }
      setStatus("Ready", "done");
      el.runBtn.disabled = false;
      stopRunPolling();
      return;
    }

    if (status.status === "error") {
      setRunProgress(pct, "Processing failed", status.elapsed_ms || 0);
      setStatus("Run Error", "error");
      el.runBtn.disabled = false;
      stopRunPolling();
      alert(status.error || "Prediction failed.");
      return;
    }

    state.runPollTimer = setTimeout(pollRunStatus, 500);
  } catch (err) {
    setStatus("Run Error", "error");
    el.runBtn.disabled = false;
    stopRunPolling();
  }
}

function safe(v) {
  if (v === null || v === undefined || v === "") return "-";
  return String(v);
}

function renderTableHead(target, columns) {
  target.innerHTML = "";
  if (!columns || !columns.length) return;
  const tr = document.createElement("tr");
  columns.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    tr.appendChild(th);
  });
  target.appendChild(tr);
}

function renderPreview(previewRows) {
  el.previewBody.innerHTML = "";
  if (!previewRows || !previewRows.length) return;
  const columns = Object.keys(previewRows[0]);
  renderTableHead(el.previewHead, columns);
  previewRows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((c) => {
      const td = document.createElement("td");
      td.textContent = row[c] ?? "";
      tr.appendChild(td);
    });
    el.previewBody.appendChild(tr);
  });
}

function fillSelect(select, items, selected = "") {
  select.innerHTML = "";
  items.forEach((it) => {
    const opt = document.createElement("option");
    opt.value = it;
    opt.textContent = it || "(empty)";
    if (it === selected) opt.selected = true;
    select.appendChild(opt);
  });
}

function renderHierarchy(current) {
  if (!current) {
    el.hierarchy.textContent = "$ no current item";
    return;
  }

  const text = [
    "$ meddra review --tree",
    "",
    `row_id        : ${safe(current.row_id)}`,
    `ae_text       : ${safe(current.ae_text)}`,
    `LLT           : ${safe(current.pred_LLT_term)} (${safe(current.pred_LLT_Code)})`,
    `  -> PT code  : ${safe(current.pred_PT_Code)}`,
    `  -> SOC      : ${safe(current.pred_SOC_Term)} (${safe(current.pred_SOC_Code)})`,
    `decision      : ${safe(current.decision)}`,
    `selection src : ${safe(current.selection_source)}`,
    `selected LLT  : ${safe(current.selected_LLT_term)} (${safe(current.selected_LLT_Code)})`,
  ].join("\n");

  el.hierarchy.textContent = text;
}

function renderCandidates(rows) {
  el.candidateBody.innerHTML = "";
  (rows || []).forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${safe(row.rank)}</td>
      <td>${safe(row.llt_term)}</td>
      <td>${safe(row.llt_code)}</td>
      <td>
        <div class="score-track"><div class="score-fill" data-score="${row.score}"></div></div>
        <div class="score-label">${row.score}%</div>
      </td>
      <td>${safe(row.pt_code)}</td>
      <td>${safe(row.soc_code)}</td>
    `;
    el.candidateBody.appendChild(tr);
  });

  requestAnimationFrame(() => {
    document.querySelectorAll(".score-fill").forEach((bar) => {
      bar.style.width = `${bar.dataset.score}%`;
    });
  });
}

function renderReviewTable(rows) {
  el.reviewBody.innerHTML = "";
  (rows || []).forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${safe(row.row_id)}</td>
      <td>${safe(row.AE_text)}</td>
      <td>${safe(row.pred_LLT_term)}</td>
      <td>${safe(row.selected_LLT_term)}</td>
      <td>${safe(row.selected_LLT_Code)}</td>
      <td>${safe(row.selection_source)}</td>
      <td>${safe(row.decision)}</td>
    `;
    el.reviewBody.appendChild(tr);
  });
}

function renderPayload(payload) {
  el.reviewSection.classList.remove("hidden");
  el.progress.textContent = payload.progress || "Item 0 / 0";

  const current = payload.current;
  if (!current) {
    el.aeText.textContent = "No row selected.";
    return;
  }

  el.aeText.textContent = safe(current.ae_text);
  el.predLlt.textContent = safe(current.pred_LLT_term);
  el.predLltCode.textContent = `Code: ${safe(current.pred_LLT_Code)}`;
  el.predPt.textContent = safe(current.pred_PT_Code);
  el.predPtCode.textContent = `Code: ${safe(current.pred_PT_Code)}`;
  el.predSoc.textContent = safe(current.pred_SOC_Term);
  el.predSocCode.textContent = `Code: ${safe(current.pred_SOC_Code)}`;
  el.latency.textContent = safe(current.latency_ms);
  el.fallback.textContent = current.used_fallback ? "Yes" : "No";
  el.decision.textContent = safe(current.decision);

  const top5 = payload.top5_choices || [];
  fillSelect(el.top5Select, top5, top5.length ? top5[0] : "");
  renderHierarchy(current);
  renderCandidates(payload.candidates || []);
  renderReviewTable(payload.table || []);

  el.manualTerm.value = "";
}

async function api(path, method = "GET", body = null) {
  const options = { method, headers: {} };
  if (body) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }

  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

async function uploadSelectedFile() {
  const file = el.fileInput.files[0];
  if (!file) {
    setStatus("No File", "error");
    throw new Error("No file selected.");
  }

  setStatus("Uploading", "loading");
  const fd = new FormData();
  fd.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    state.jobId = data.job_id;
    stopRunPolling();
    el.runProgressWrap.classList.add("hidden");
    el.runProgressFill.style.width = "0%";
    fillSelect(el.aeColumn, data.columns, data.default_column);
    renderPreview(data.preview || []);
    el.fileMeta.textContent = `${data.file_name} loaded (${data.row_count} rows).`;
    setStatus("File Loaded", "done");
    return data;
  } catch (err) {
    setStatus("Upload Error", "error");
    el.fileMeta.textContent = err.message || String(err);
    throw err;
  }
}

el.fileInput.addEventListener("change", async () => {
  if (!el.fileInput.files[0]) return;
  try {
    await uploadSelectedFile();
  } catch (_) {
    // message already shown in UI
  }
});

el.runBtn.addEventListener("click", async () => {
  setStatus("Running", "loading");
  el.runBtn.disabled = true;
  stopRunPolling();
  setRunProgress(0, "Starting prediction...", 0);

  try {
    if (!state.jobId) {
      await uploadSelectedFile();
    }
    await api("/api/run", "POST", {
      job_id: state.jobId,
      ae_column: el.aeColumn.value,
    });
    state.runPollTimer = setTimeout(pollRunStatus, 300);
  } catch (err) {
    setStatus("Run Error", "error");
    el.runBtn.disabled = false;
    stopRunPolling();
    alert(err.message || String(err));
  }
});

el.prevBtn.addEventListener("click", async () => {
  if (!state.jobId) return;
  try {
    const payload = await api("/api/nav", "POST", { job_id: state.jobId, delta: -1 });
    renderPayload(payload);
  } catch (err) {
    setStatus("Nav Error", "error");
  }
});

el.nextBtn.addEventListener("click", async () => {
  if (!state.jobId) return;
  try {
    const payload = await api("/api/nav", "POST", { job_id: state.jobId, delta: 1 });
    renderPayload(payload);
  } catch (err) {
    setStatus("Nav Error", "error");
  }
});

el.acceptBtn.addEventListener("click", async () => {
  if (!state.jobId) return;
  try {
    const payload = await api("/api/accept", "POST", { job_id: state.jobId });
    renderPayload(payload);
    setStatus("Accepted", "done");
  } catch (err) {
    setStatus("Accept Error", "error");
  }
});

el.rejectBtn.addEventListener("click", async () => {
  if (!state.jobId) return;
  try {
    const payload = await api("/api/reject", "POST", {
      job_id: state.jobId,
      chosen_term: el.top5Select.value,
      manual_term: el.manualTerm.value,
    });
    renderPayload(payload);
    setStatus("Rejected", "done");
  } catch (err) {
    setStatus("Reject Error", "error");
  }
});

el.exportBtn.addEventListener("click", () => {
  if (!state.jobId) return;
  window.location.href = `/api/export/${state.jobId}`;
});

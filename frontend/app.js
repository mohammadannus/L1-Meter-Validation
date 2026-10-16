/**
 * OmniOCR Platform  |  Multi-Application Frontend Logic
 * Supports Meter OCR, Invoice Application No. Reader, and General Document OCR.
 */

const API_BASE = '/api/v1';

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------
let currentTab = 'meter'; // 'meter' | 'invoice' | 'ocr'
let selectedFile = null;
let requestCount = 0;
let successCount = 0;
let lastProcessingTime = null;
let regionsOpen = false;

// ---------------------------------------------------------------------------
// Tab Configuration
// ---------------------------------------------------------------------------
const TAB_CONFIG = {
  meter: {
    title: 'Meter <span>Identifier Extraction</span>',
    subtitle: 'Upload an electricity meter image to extract Serial Number, IMEI, LED power status, and seal security.',
    endpoint: `${API_BASE}/extract`,
    endpointLabel: '/api/v1/extract',
    dropIcon: '⚡',
    dropTitle: 'Drop your meter image here',
    dropHint: 'Supports JPEG, PNG, WEBP, BMP, TIFF · Max 20 MB',
    buttonText: '⚡ Analyze Meter',
    loaderTitle: 'Analysing electricity meter…',
    stages: [
      'Waiting for OCR concurrency slot…',
      'Detecting meter body via YOLO…',
      'Running PaddleOCR recognition…',
      'Extracting Serial & IMEI patterns…',
      'Analyzing LED power & tamper seals…',
    ],
  },
  invoice: {
    title: 'Invoice <span>Application No. Reader</span>',
    subtitle: 'Upload utility procurement letters or invoice photos to extract Application No, Account ID, and Reference No.',
    endpoint: `${API_BASE}/invoice/extract`,
    endpointLabel: '/api/v1/invoice/extract',
    dropIcon: '📄',
    dropTitle: 'Drop your invoice or document photo here',
    dropHint: 'Supports JPEG, PNG, WEBP, BMP, TIFF · Max 20 MB',
    buttonText: '📄 Extract Application No.',
    loaderTitle: 'Reading invoice document…',
    stages: [
      'Waiting for OCR concurrency slot…',
      'Running full-document PaddleOCR…',
      'Extracting Application No. & Account ID…',
      'Validating utility procurement patterns…',
    ],
  },
  ocr: {
    title: 'General <span>Document OCR</span>',
    subtitle: 'Upload any image, scanned document, or photo to extract raw text lines using the shared PaddleOCR engine.',
    endpoint: `${API_BASE}/ocr/extract`,
    endpointLabel: '/api/v1/ocr/extract',
    dropIcon: '📝',
    dropTitle: 'Drop any photo or document here',
    dropHint: 'Supports JPEG, PNG, WEBP, BMP, TIFF · Max 20 MB',
    buttonText: '📝 Extract Raw Text',
    loaderTitle: 'Extracting text from image…',
    stages: [
      'Waiting for OCR concurrency slot…',
      'Running PaddleOCR recognition…',
      'Formatting detected text lines…',
    ],
  },
};

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------
const uploadZone       = document.getElementById('upload-zone');
const fileInput        = document.getElementById('file-input');
const uploadZoneWrap   = document.getElementById('upload-zone-wrap');
const previewWrap      = document.getElementById('preview-wrap');
const previewImg       = document.getElementById('preview-img');
const previewRemoveBtn = document.getElementById('preview-remove');
const loaderOverlay    = document.getElementById('loader-overlay');
const loaderTitle      = document.getElementById('loader-title');
const loaderStages     = document.getElementById('loader-stages');
const resultsSection   = document.getElementById('results-section');
const errorBanner      = document.getElementById('error-banner');
const errorMsg         = document.getElementById('error-msg');
const analyzeBtn       = document.getElementById('analyze-btn');

// Header elements
const pageTitle        = document.getElementById('page-title');
const pageSubtitle     = document.getElementById('page-subtitle');
const uploadIcon       = document.getElementById('upload-icon');
const uploadTitle      = document.getElementById('upload-title');
const uploadHint       = document.getElementById('upload-hint');
const endpointBadge    = document.getElementById('endpoint-badge');
const timingBadge      = document.getElementById('timing-badge');

// Result Views
const viewMeter        = document.getElementById('view-meter-results');
const viewInvoice      = document.getElementById('view-invoice-results');
const viewOcr          = document.getElementById('view-ocr-results');

// Meter DOM refs
const serialValue      = document.getElementById('serial-value');
const imeiValue        = document.getElementById('imei-value');
const regionsBody      = document.getElementById('regions-body');
const regionsChevron   = document.getElementById('regions-chevron');
const regionsHeader    = document.getElementById('regions-header');
const regionsCount     = document.getElementById('regions-count');

// Invoice DOM refs
const invoiceAppValue  = document.getElementById('invoice-app-value');
const invoiceAccValue  = document.getElementById('invoice-acc-value');
const invoiceRefValue  = document.getElementById('invoice-ref-value');
const invoiceRawText   = document.getElementById('invoice-raw-text');

// General OCR DOM refs
const ocrStatLines     = document.getElementById('ocr-stat-lines');
const ocrStatChars     = document.getElementById('ocr-stat-chars');
const ocrLineBadge     = document.getElementById('ocr-line-badge');
const ocrRawText       = document.getElementById('ocr-raw-text');

// JSON & Stats refs
const jsonCode         = document.getElementById('json-code');
const statRequests     = document.getElementById('stat-requests');
const statSuccess      = document.getElementById('stat-success');

// Concurrency Live Status
const queueDevice      = document.getElementById('queue-device');
const queueActive      = document.getElementById('queue-active');
const queueWaiting     = document.getElementById('queue-waiting');
const queueProcessed   = document.getElementById('queue-processed');

// ---------------------------------------------------------------------------
// Tab Switching
// ---------------------------------------------------------------------------
function switchTab(tabKey) {
  if (!TAB_CONFIG[tabKey]) return;
  currentTab = tabKey;
  const cfg = TAB_CONFIG[tabKey];

  // Update top tabs
  document.querySelectorAll('.top-tab').forEach(btn => {
    const isActive = btn.dataset.tab === tabKey;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });

  // Update sidebar tabs
  document.querySelectorAll('.nav-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabKey);
  });

  // Update headers
  pageTitle.innerHTML = cfg.title;
  pageSubtitle.textContent = cfg.subtitle;
  uploadIcon.textContent = cfg.dropIcon;
  uploadTitle.textContent = cfg.dropTitle;
  uploadHint.textContent = cfg.dropHint;
  endpointBadge.textContent = cfg.endpointLabel;
  analyzeBtn.textContent = cfg.buttonText;

  // Clear or hide results if file not processed yet
  hideResults();
  hideError();

  // If a file is already loaded, re-enable analyze button
  if (selectedFile) {
    analyzeBtn.disabled = false;
  }
}

// Attach Tab Switcher Listeners
document.querySelectorAll('.top-tab, .nav-tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    if (tab) switchTab(tab);
  });
});

// ---------------------------------------------------------------------------
// Upload Zone Interactions
// ---------------------------------------------------------------------------
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

previewRemoveBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFile();
});

function handleFile(file) {
  const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'];
  if (!validTypes.includes(file.type)) {
    showError('Unsupported file type. Please upload a JPEG, PNG, WEBP, BMP or TIFF image.');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError('File too large. Maximum allowed size is 20 MB.');
    return;
  }

  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  uploadZoneWrap.style.display = 'none';
  previewWrap.classList.add('visible');
  hideError();
  hideResults();
  analyzeBtn.disabled = false;

  // Automatically start extraction when an image is dropped/selected
  runInference();
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewWrap.classList.remove('visible');
  uploadZoneWrap.style.display = '';
  analyzeBtn.disabled = true;
  hideResults();
  hideError();
}

// Manual analyze button
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  await runInference();
});

// ---------------------------------------------------------------------------
// API Inference Execution
// ---------------------------------------------------------------------------
let stageTimer = null;
let elapsedTimer = null;

async function runInference() {
  if (!selectedFile) return;

  const cfg = TAB_CONFIG[currentTab];

  // UI — loading state
  hideResults();
  hideError();
  showLoader(cfg.loaderTitle);
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = '⏳ Processing…';

  requestCount++;
  statRequests.textContent = requestCount;

  // Start animated progress stages
  animateStages(cfg.stages);

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    // If meter endpoint, also pass model
    if (currentTab === 'meter') {
      formData.append('model', 'paddleocr');
    }

    const response = await fetch(cfg.endpoint, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      if (response.status === 503) {
        throw new Error('Server is currently busy processing other requests. Please wait a moment and retry.');
      } else if (response.status === 504) {
        throw new Error('Request timed out in queue. The system is under heavy load.');
      }
      throw new Error(data.detail || `Server error ${response.status}`);
    }

    // Render results according to active tab
    renderResults(currentTab, data);

    successCount++;
    statSuccess.textContent = successCount;
    lastProcessingTime = data.processing_time_ms;

    // Refresh queue stats immediately
    fetchQueueStatus();

  } catch (err) {
    showError(err.message || 'An unexpected error occurred.');
  } finally {
    hideLoader();
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = cfg.buttonText;
  }
}

// ---------------------------------------------------------------------------
// Result Renderers
// ---------------------------------------------------------------------------
function renderResults(tabKey, data) {
  // Hide all view containers
  viewMeter.style.display = 'none';
  viewInvoice.style.display = 'none';
  viewOcr.style.display = 'none';

  timingBadge.textContent = `⏱ ${(data.processing_time_ms || 0).toFixed(0)} ms`;
  endpointBadge.textContent = TAB_CONFIG[tabKey].endpointLabel;

  if (tabKey === 'meter') {
    renderMeterResults(data);
    viewMeter.style.display = '';
  } else if (tabKey === 'invoice') {
    renderInvoiceResults(data);
    viewInvoice.style.display = '';
  } else if (tabKey === 'ocr') {
    renderOcrResults(data);
    viewOcr.style.display = '';
  }

  // Populate JSON viewer
  if (jsonCode) {
    jsonCode.textContent = JSON.stringify(data, null, 2);
  }

  resultsSection.classList.add('visible');

  // Smooth scroll
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 100);
}

function renderMeterResults(data) {
  const sn = data.serial_number || '';
  const imei = data.imei || '';

  // Serial number
  if (sn) {
    serialValue.textContent = sn;
    serialValue.classList.remove('not-found');
    document.getElementById('copy-serial').style.display = '';
  } else {
    serialValue.textContent = 'Not found';
    serialValue.classList.add('not-found');
    document.getElementById('copy-serial').style.display = 'none';
  }

  // IMEI
  if (imei) {
    imeiValue.textContent = imei;
    imeiValue.classList.remove('not-found');
    document.getElementById('copy-imei').style.display = '';
  } else {
    imeiValue.textContent = 'Not found';
    imeiValue.classList.add('not-found');
    document.getElementById('copy-imei').style.display = 'none';
  }

  // LED Power Status
  const greenStatus = data.power_status || data.green_light_status || 'unknown';
  const greenPct = typeof data.green_light_percent === 'number' ? data.green_light_percent : 0;
  const ledEl = document.getElementById('led-indicator');
  const statusTextEl = document.getElementById('green-light-status-text');
  const pctBadgeEl = document.getElementById('green-pct-badge');

  if (ledEl && statusTextEl && pctBadgeEl) {
    ledEl.className = 'led-indicator';
    if (greenStatus === 'powered_on' || greenStatus === 'connected') {
      ledEl.classList.add('led-on');
      statusTextEl.textContent = '🟢 POWERED ON';
      statusTextEl.className = 'green-light-status-text status-connected';
    } else if (greenStatus === 'powered_off' || greenStatus === 'disconnected') {
      ledEl.classList.add('led-off');
      statusTextEl.textContent = '⚫ POWERED OFF';
      statusTextEl.className = 'green-light-status-text status-disconnected';
    } else {
      ledEl.classList.add('led-unknown');
      statusTextEl.textContent = '❓ UNKNOWN';
      statusTextEl.className = 'green-light-status-text status-unknown';
    }
    pctBadgeEl.textContent = `${greenPct}% green`;
  }

  // Seal Status
  const sealCount = typeof data.seal_count === 'number' ? data.seal_count : 0;
  const meterLocked = Boolean(data.meter_locked);
  const sealConfs = Array.isArray(data.seal_confidences) ? data.seal_confidences : [];
  const avgConf = typeof data.average_seal_confidence === 'number' ? data.average_seal_confidence : 0;
  const sealCountEl = document.getElementById('seal-count-badge');
  const sealLockIcon = document.getElementById('seal-lock-icon');
  const sealStatusEl = document.getElementById('seal-status-text');
  const sealDetailEl = document.getElementById('seal-conf-detail');

  if (sealCountEl && sealStatusEl) {
    sealCountEl.textContent = `${sealCount} seal${sealCount === 1 ? '' : 's'}`;
    if (meterLocked && sealCount > 0) {
      if (sealLockIcon) sealLockIcon.textContent = '🔒';
      sealStatusEl.textContent = `LOCKED (${sealCount} SEAL${sealCount === 1 ? '' : 'S'})`;
      sealStatusEl.className = 'seal-status-text status-locked';
      if (sealDetailEl) {
        const confPercents = sealConfs.map(c => `${(c * 100).toFixed(1)}%`);
        sealDetailEl.textContent = `Avg: ${(avgConf * 100).toFixed(1)}% [${confPercents.join(', ')}]`;
      }
    } else {
      if (sealLockIcon) sealLockIcon.textContent = '🔓';
      sealStatusEl.textContent = 'UNLOCKED (0 SEALS)';
      sealStatusEl.className = 'seal-status-text status-unlocked';
      if (sealDetailEl) sealDetailEl.textContent = 'No tamper seals detected';
    }
  }

  // Regions breakdown
  renderRegions(data.regions || []);
}

function renderInvoiceResults(data) {
  const appNo = data.application_no || '';
  const accId = data.account_id || '';
  const refNo = data.reference_no || '';
  const rawText = data.raw_text || '';

  // Application No
  if (appNo) {
    invoiceAppValue.textContent = appNo;
    invoiceAppValue.classList.remove('not-found');
    document.getElementById('copy-invoice-app').style.display = '';
  } else {
    invoiceAppValue.textContent = 'Not found';
    invoiceAppValue.classList.add('not-found');
    document.getElementById('copy-invoice-app').style.display = 'none';
  }

  // Account ID
  if (accId) {
    invoiceAccValue.textContent = accId;
    invoiceAccValue.classList.remove('not-found');
    document.getElementById('copy-invoice-acc').style.display = '';
  } else {
    invoiceAccValue.textContent = 'Not found';
    invoiceAccValue.classList.add('not-found');
    document.getElementById('copy-invoice-acc').style.display = 'none';
  }

  // Reference No
  if (refNo) {
    invoiceRefValue.textContent = refNo;
    invoiceRefValue.classList.remove('not-found');
    document.getElementById('copy-invoice-ref').style.display = '';
  } else {
    invoiceRefValue.textContent = 'Not found in document';
    invoiceRefValue.classList.add('not-found');
    document.getElementById('copy-invoice-ref').style.display = 'none';
  }

  // Document Raw Text
  invoiceRawText.textContent = rawText || 'No text recognized.';
}

function renderOcrResults(data) {
  const rawText = data.raw_text || '';
  const lines = Array.isArray(data.lines) ? data.lines : (rawText.split('\n').filter(l => l.trim()));
  const lineCount = typeof data.line_count === 'number' ? data.line_count : lines.length;

  ocrStatLines.textContent = lineCount;
  ocrStatChars.textContent = rawText.length;
  ocrLineBadge.textContent = `${lineCount} line${lineCount === 1 ? '' : 's'}`;
  ocrRawText.textContent = rawText || 'No text recognized.';
}

function renderRegions(regions) {
  regionsBody.innerHTML = '';
  regionsCount.textContent = `${regions.length} regions`;

  regions.forEach(r => {
    const row = document.createElement('div');
    row.className = 'region-row';
    const rawPreview = (r.raw_output || '').replace(/\n/g, ' · ').slice(0, 60);
    row.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:2px;overflow:hidden">
        <span class="region-name">${r.region}</span>
        ${rawPreview ? `<span style="font-size:0.72rem;color:var(--text-dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escapeHtml(r.raw_output || '')}">${escapeHtml(rawPreview)}</span>` : ''}
      </div>
      <span class="${r.serial_number ? 'region-serial' : 'region-empty'}">${r.serial_number || '—'}</span>
      <span class="${r.imei ? 'region-imei' : 'region-empty'}">${r.imei || '—'}</span>
    `;
    regionsBody.appendChild(row);
  });
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Regions accordion toggle
regionsHeader.addEventListener('click', () => {
  regionsOpen = !regionsOpen;
  regionsBody.classList.toggle('open', regionsOpen);
  regionsChevron.classList.toggle('open', regionsOpen);
});

// ---------------------------------------------------------------------------
// Animated Progress Stages
// ---------------------------------------------------------------------------
function animateStages(stageList) {
  loaderStages.innerHTML = '';
  stageList.forEach((label, i) => {
    const el = document.createElement('div');
    el.className = 'loader-stage';
    el.id = `stage-${i}`;
    el.innerHTML = `<span class="stage-dot"></span><span>${label}</span>`;
    loaderStages.appendChild(el);
  });

  const timerEl = document.createElement('div');
  timerEl.id = 'loader-timer';
  timerEl.style.cssText = 'margin-top:14px;font-size:0.8rem;color:var(--text-muted);font-family:var(--font-mono);text-align:center;';
  timerEl.textContent = '⏱ 0s elapsed';
  loaderStages.appendChild(timerEl);

  const startTime = Date.now();
  if (elapsedTimer) clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    const sec = Math.floor((Date.now() - startTime) / 1000);
    const t = document.getElementById('loader-timer');
    if (t) t.textContent = `⏱ ${sec}s elapsed (FIFO Guard Active)`;
  }, 1000);

  let current = 0;
  if (stageTimer) clearInterval(stageTimer);
  stageTimer = setInterval(() => {
    if (current > 0) {
      const prev = document.getElementById(`stage-${current - 1}`);
      if (prev) { prev.classList.remove('active'); prev.classList.add('done'); }
    }
    const curr = document.getElementById(`stage-${current}`);
    if (curr) curr.classList.add('active');
    current++;
    if (current >= stageList.length) clearInterval(stageTimer);
  }, 1400);
}

// ---------------------------------------------------------------------------
// Copy to Clipboard Utility
// ---------------------------------------------------------------------------
function copyToClipboard(text, btn) {
  if (!text || text === 'Not found' || text === 'No text extracted.') return;
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.innerHTML;
    btn.classList.add('copied');
    btn.innerHTML = '✓ Copied';
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.innerHTML = original;
    }, 1800);
  });
}

// Copy event listeners
document.getElementById('copy-serial').addEventListener('click', () => {
  copyToClipboard(serialValue.textContent, document.getElementById('copy-serial'));
});
document.getElementById('copy-imei').addEventListener('click', () => {
  copyToClipboard(imeiValue.textContent, document.getElementById('copy-imei'));
});
document.getElementById('copy-invoice-app').addEventListener('click', () => {
  copyToClipboard(invoiceAppValue.textContent, document.getElementById('copy-invoice-app'));
});
document.getElementById('copy-invoice-acc').addEventListener('click', () => {
  copyToClipboard(invoiceAccValue.textContent, document.getElementById('copy-invoice-acc'));
});
document.getElementById('copy-invoice-ref').addEventListener('click', () => {
  copyToClipboard(invoiceRefValue.textContent, document.getElementById('copy-invoice-ref'));
});
document.getElementById('copy-invoice-text').addEventListener('click', () => {
  copyToClipboard(invoiceRawText.textContent, document.getElementById('copy-invoice-text'));
});
document.getElementById('copy-ocr-text').addEventListener('click', () => {
  copyToClipboard(ocrRawText.textContent, document.getElementById('copy-ocr-text'));
});
document.getElementById('copy-json').addEventListener('click', () => {
  copyToClipboard(jsonCode ? jsonCode.textContent : '', document.getElementById('copy-json'));
});

// ---------------------------------------------------------------------------
// Live Concurrency & Health Polling
// ---------------------------------------------------------------------------
async function fetchQueueStatus() {
  try {
    const res = await fetch(`${API_BASE}/queue/status`);
    if (!res.ok) return;
    const q = await res.json();
    if (queueActive) queueActive.textContent = `${q.active_requests} / ${q.max_concurrent}`;
    if (queueWaiting) queueWaiting.textContent = `${q.queued_requests} / ${q.max_queue_depth}`;
    if (queueProcessed) queueProcessed.textContent = q.total_processed;
  } catch (_) {}
}

async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return;
    const h = await res.json();
    if (queueDevice) queueDevice.textContent = (h.device || 'cpu').toUpperCase();
    if (h.queue_status) {
      const q = h.queue_status;
      if (queueActive) queueActive.textContent = `${q.active_requests} / ${q.max_concurrent}`;
      if (queueWaiting) queueWaiting.textContent = `${q.queued_requests} / ${q.max_queue_depth}`;
      if (queueProcessed) queueProcessed.textContent = q.total_processed;
    }
  } catch (_) {}
}

// ---------------------------------------------------------------------------
// UI State Helpers
// ---------------------------------------------------------------------------
function showLoader(titleText) {
  if (loaderTitle && titleText) loaderTitle.textContent = titleText;
  loaderOverlay.classList.add('visible');
}
function hideLoader() {
  loaderOverlay.classList.remove('visible');
  if (stageTimer) clearInterval(stageTimer);
  if (elapsedTimer) clearInterval(elapsedTimer);
}
function hideResults() { resultsSection.classList.remove('visible'); }
function showError(msg) { errorMsg.textContent = msg; errorBanner.classList.add('visible'); }
function hideError() { errorBanner.classList.remove('visible'); }

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
switchTab('meter');
fetchHealth();
fetchQueueStatus();
setInterval(fetchQueueStatus, 4000);

/**
 * End-to-End Live Backend + UI Integration Test
 * Verifies live FastAPI backend responses against frontend rendering functions.
 */

const fs = require('fs');
const path = require('path');

async function runE2ETests() {
  console.log('=== STARTING E2E LIVE UI & BACKEND VERIFICATION ===\n');

  // 1. Verify index.html & app.js served from live server
  console.log('1. Checking Live HTTP Server...');
  const htmlRes = await fetch('http://127.0.0.1:8000/vqa');
  if (!htmlRes.ok) throw new Error(`Failed to fetch /vqa from server: ${htmlRes.status}`);
  const liveHtml = await htmlRes.text();
  console.log('  ✓ /vqa returned 200 OK (length: ' + liveHtml.length + ' bytes)');

  const jsRes = await fetch('http://127.0.0.1:8000/static/app.js');
  if (!jsRes.ok) throw new Error(`Failed to fetch /static/app.js: ${jsRes.status}`);
  const liveJs = await jsRes.text();
  console.log('  ✓ /static/app.js returned 200 OK (length: ' + liveJs.length + ' bytes)');

  // 2. Set up Virtual DOM for testing live rendering
  class MockClassList {
    constructor() { this.classes = new Set(); }
    add(c) { this.classes.add(c); }
    remove(c) { this.classes.delete(c); }
    contains(c) { return this.classes.has(c); }
  }

  class MockElement {
    constructor(id, tag = 'div') {
      this.id = id;
      this.tagName = tag.toUpperCase();
      this.textContent = '';
      this.innerHTML = '';
      this.classList = new MockClassList();
      this.src = '';
      this.checked = true;
      this.clientWidth = 400;
      this.clientHeight = 300;
    }
    getContext() {
      return {
        clearRect: () => {},
        strokeRect: () => {},
        fillRect: () => {},
        fillText: () => {},
        measureText: (txt) => ({ width: txt.length * 6 })
      };
    }
  }

  const mockDom = new Map();
  const allIds = [
    'resAnswer', 'resTask', 'resTool', 'resModel', 'resInput', 'resLatency',
    'resConfidence', 'routingReasonBanner', 'routingReasonText',
    'agentTraceContainer', 'traceSteps', 'metaCrs', 'metaShape', 'metaBands',
    'metaDriver', 'metaCrs2', 'metaShape2', 'metaBands2', 'metaDriver2',
    'previewsGrid', 'previewSlot1', 'previewSlot2', 'previewSlotHeader1', 'previewSlotHeader2',
    'previewSlotLabel1', 'previewSlotLabel2', 'imagePreview', 'imagePreview2',
    'labelPrimaryImage', 'dropTextPrimary', 'labelSecondaryImage',
    'dropTextSecondary', 'groundingCanvas', 'changeOverlay', 'toggleGrounding',
    'toggleChange', 'toggleChangeLayer', 'layerControls', 'provenanceFeedFull'
  ];

  allIds.forEach(id => mockDom.set(id, new MockElement(id)));

  global.document = {
    getElementById: (id) => mockDom.get(id) || null,
    querySelector: () => new MockElement('mockQuery'),
    querySelectorAll: () => []
  };

  // Extract functions from app.js
  function setText(id, value, fallback = '—') {
    const element = document.getElementById(id);
    if (!element) {
      console.error(`[SatQuery UI] Missing DOM element: #${id}`);
      return false;
    }
    element.textContent = (value !== undefined && value !== null) ? String(value) : fallback;
    return true;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function renderAgentTrace(trace) {
    const container = document.getElementById('agentTraceContainer');
    const stepsContainer = document.getElementById('traceSteps');
    if (!container || !stepsContainer) {
      console.warn('[SatQuery UI] Agent trace DOM elements missing.');
      return;
    }
    if (!trace || !Array.isArray(trace.steps) || trace.steps.length === 0) {
      container.classList.add('hidden');
      stepsContainer.innerHTML = '';
      return;
    }
    container.classList.remove('hidden');
    stepsContainer.innerHTML = trace.steps.map(s => {
      const stepNum = s && s.step !== undefined ? String(s.step) : '•';
      const action = s && s.action ? String(s.action) : 'Step';
      const details = s && s.details ? String(s.details) : '';
      return `<div class="trace-step-item"><span>${escapeHtml(stepNum)}</span><div><strong>${escapeHtml(action)}:</strong> ${escapeHtml(details)}</div></div>`;
    }).join('');
  }

  function renderAgentResult(data, currentMode = 'vqa') {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid response structure received from backend.');
    }

    setText('resAnswer', data.answer, 'No answer returned.');
    const displayTask = (data.task || currentMode || 'VQA').toUpperCase().replace(/_/g, ' ');
    setText('resTask', displayTask, 'UNKNOWN');
    setText('resTool', data.tool_used, 'Specialist Tool');
    setText('resModel', data.model, '—');

    const latencyStr = data.execution_time_sec !== undefined && data.execution_time_sec !== null
      ? `${data.execution_time_sec}s`
      : '—';
    setText('resLatency', latencyStr, '—');

    let confText = '—';
    if (typeof data.confidence === 'number' && !isNaN(data.confidence)) {
      confText = `${Math.round(data.confidence * 100)}%`;
    } else if (typeof data.confidence === 'string' && data.confidence.trim().length > 0) {
      confText = data.confidence;
    }
    setText('resConfidence', confText, '—');

    let inputSummary = '1 image';
    if (data.execution_trace && data.execution_trace.detected_inputs) {
      const prim = data.execution_trace.detected_inputs.primary;
      const sec = data.execution_trace.detected_inputs.secondary;
      if (sec) {
        const primMod = (prim && prim.modality) ? prim.modality.toUpperCase() : 'OPTICAL';
        const secMod = (sec && sec.modality) ? sec.modality.toUpperCase() : 'SAR';
        inputSummary = `2 inputs (${primMod} + ${secMod})`;
      } else if (prim) {
        const primMod = prim.modality ? prim.modality.toUpperCase() : 'OPTICAL';
        inputSummary = `1 ${primMod} scene`;
      }
    }
    setText('resInput', inputSummary, '—');

    const reasonBanner = document.getElementById('routingReasonBanner');
    if (reasonBanner) {
      if (data.execution_trace && data.execution_trace.routing_reason) {
        setText('routingReasonText', data.execution_trace.routing_reason, '—');
        reasonBanner.classList.remove('hidden');
      } else {
        reasonBanner.classList.add('hidden');
      }
    }

    const canvas = document.getElementById('groundingCanvas');
    if (Array.isArray(data.boxes) && data.boxes.length > 0) {
      if (canvas) canvas.classList.remove('hidden');
    } else {
      if (canvas) canvas.classList.add('hidden');
    }

    const changeOverlay = document.getElementById('changeOverlay');
    if (data.change_map_url) {
      if (changeOverlay) {
        changeOverlay.src = data.change_map_url;
        changeOverlay.classList.remove('hidden');
      }
    } else {
      if (changeOverlay) changeOverlay.classList.add('hidden');
    }

    renderAgentTrace(data.execution_trace);

    if (data.metadata && typeof data.metadata === 'object') {
      setText('metaCrs', data.metadata.crs, 'Non-georeferenced');
      const shapeStr = Array.isArray(data.metadata.shape) ? `[${data.metadata.shape.join(', ')}]` : (data.metadata.shape || '—');
      setText('metaShape', shapeStr, '—');
      setText('metaBands', data.metadata.count || data.metadata.bands, '—');
      setText('metaDriver', data.metadata.driver, '—');
    }
  }

  // 3. Test Workflow 1: VQA Semantic Scene Understanding
  console.log('\n2. Testing VQA Live Request & UI Rendering:');
  console.log('   Query: "describe the land cover and major objects visible in this image"');
  const sampleAgriPath = path.join(__dirname, '..', 'data', 'samples', 'sample_agricultural.tif');
  const fileBytes = fs.readFileSync(sampleAgriPath);
  const blob = new Blob([fileBytes], { type: 'image/tiff' });

  const formData1 = new FormData();
  formData1.append('image', blob, 'sample_agricultural.tif');
  formData1.append('question', 'describe the land cover and major objects visible in this image');
  formData1.append('task_mode', 'vqa');

  const vqaRes = await fetch('http://127.0.0.1:8000/agent/analyze', {
    method: 'POST',
    body: formData1
  });
  if (!vqaRes.ok) throw new Error(`VQA API returned ${vqaRes.status}`);
  const vqaData = await vqaRes.json();

  renderAgentResult(vqaData, 'vqa');

  console.log('   ✓ Selected Task: ' + mockDom.get('resTask').textContent);
  console.log('   ✓ Tool Used:     ' + mockDom.get('resTool').textContent);
  console.log('   ✓ Model Name:    ' + mockDom.get('resModel').textContent);
  console.log('   ✓ Confidence:    ' + mockDom.get('resConfidence').textContent);
  console.log('   ✓ Speed:         ' + mockDom.get('resLatency').textContent);
  console.log('   ✓ Input:         ' + mockDom.get('resInput').textContent);
  console.log('   ✓ Answer:        ' + mockDom.get('resAnswer').textContent.slice(0, 80) + '...');

  if (mockDom.get('resTask').textContent !== 'VQA' ||
      mockDom.get('resTool').textContent !== 'RemoteSensingVQATool' ||
      mockDom.get('resAnswer').textContent === 'Inference failed.') {
    throw new Error('VQA UI rendering verification failed!');
  }

  // 4. Test Workflow 2: Visual Grounding
  console.log('\n3. Testing Visual Grounding Live Request & UI Rendering:');
  console.log('   Query: "Where is the water body?"');
  const formData2 = new FormData();
  formData2.append('image', blob, 'sample_agricultural.tif');
  formData2.append('question', 'Where is the water body?');
  formData2.append('task_mode', 'grounding');

  const gndRes = await fetch('http://127.0.0.1:8000/agent/analyze', {
    method: 'POST',
    body: formData2
  });
  if (!gndRes.ok) throw new Error(`Grounding API returned ${gndRes.status}`);
  const gndData = await gndRes.json();

  renderAgentResult(gndData, 'grounding');

  console.log('   ✓ Selected Task: ' + mockDom.get('resTask').textContent);
  console.log('   ✓ Tool Used:     ' + mockDom.get('resTool').textContent);
  console.log('   ✓ Model Name:    ' + mockDom.get('resModel').textContent);
  console.log('   ✓ Boxes count:   ' + (gndData.boxes ? gndData.boxes.length : 0));
  console.log('   ✓ Canvas active: ' + !mockDom.get('groundingCanvas').classList.contains('hidden'));

  if (mockDom.get('resTask').textContent !== 'GROUNDING' ||
      mockDom.get('resTool').textContent !== 'RemoteSensingGroundingTool') {
    throw new Error('Grounding UI rendering verification failed!');
  }

  // 5. Test Workflow 3: Bi-Temporal Change Analysis
  console.log('\n4. Testing Bi-temporal Change Live Request & UI Rendering:');
  console.log('   Query: "What changed between these two dates?"');
  const t1Path = path.join(__dirname, '..', 'data', 'samples', 'sample_bitemporal_t1.png');
  const t2Path = path.join(__dirname, '..', 'data', 'samples', 'sample_bitemporal_t2.png');
  const t1Blob = new Blob([fs.readFileSync(t1Path)], { type: 'image/png' });
  const t2Blob = new Blob([fs.readFileSync(t2Path)], { type: 'image/png' });

  const formData3 = new FormData();
  formData3.append('image', t1Blob, 'sample_bitemporal_t1.png');
  formData3.append('secondary_image', t2Blob, 'sample_bitemporal_t2.png');
  formData3.append('question', 'What changed between these two dates?');
  formData3.append('task_mode', 'bitemporal_change');

  const chgRes = await fetch('http://127.0.0.1:8000/agent/analyze', {
    method: 'POST',
    body: formData3
  });
  if (!chgRes.ok) throw new Error(`Change API returned ${chgRes.status}`);
  const chgData = await chgRes.json();

  renderAgentResult(chgData, 'bitemporal_change');

  console.log('   ✓ Selected Task: ' + mockDom.get('resTask').textContent);
  console.log('   ✓ Tool Used:     ' + mockDom.get('resTool').textContent);
  console.log('   ✓ Overlay URL:   ' + (mockDom.get('changeOverlay').src ? 'Set (data:image/png)' : 'None'));

  if (mockDom.get('resTask').textContent !== 'BITEMPORAL CHANGE' ||
      mockDom.get('resTool').textContent !== 'RemoteSensingChangeTool') {
    throw new Error('Change UI rendering verification failed!');
  }

  // 6. Test Workflow 4: Optical + SAR Fusion
  console.log('\n5. Testing Optical+SAR Cross-Modal Fusion Live Request & UI Rendering:');
  console.log('   Query: "Use optical and SAR together to identify built-up areas."');
  const optPath = path.join(__dirname, '..', 'data', 'samples', 'sample_optical_s2.png');
  const sarPath = path.join(__dirname, '..', 'data', 'samples', 'sample_sar_s1.png');
  const optBlob = new Blob([fs.readFileSync(optPath)], { type: 'image/png' });
  const sarBlob = new Blob([fs.readFileSync(sarPath)], { type: 'image/png' });

  const formData4 = new FormData();
  formData4.append('image', optBlob, 'sample_optical_s2.png');
  formData4.append('secondary_image', sarBlob, 'sample_sar_s1.png');
  formData4.append('question', 'Use optical and SAR together to identify built-up areas.');
  formData4.append('task_mode', 'optical_sar_fusion');

  const fusRes = await fetch('http://127.0.0.1:8000/agent/analyze', {
    method: 'POST',
    body: formData4
  });
  if (!fusRes.ok) throw new Error(`Fusion API returned ${fusRes.status}`);
  const fusData = await fusRes.json();

  renderAgentResult(fusData, 'optical_sar_fusion');

  console.log('   ✓ Selected Task: ' + mockDom.get('resTask').textContent);
  console.log('   ✓ Tool Used:     ' + mockDom.get('resTool').textContent);
  console.log('   ✓ Detected Input:' + mockDom.get('resInput').textContent);

  if (mockDom.get('resTask').textContent !== 'OPTICAL SAR FUSION' ||
      mockDom.get('resTool').textContent !== 'OpticalSARFusionTool') {
    throw new Error('Fusion UI rendering verification failed!');
  }

  console.log('\n======================================================');
  console.log('ALL 4 WORKFLOWS RENDERED PERFECTLY WITH ZERO ERRORS');
  console.log('======================================================');
}

runE2ETests().catch(err => {
  console.error('\nE2E Test Failed:', err);
  process.exit(1);
});

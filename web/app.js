/**
 * SatQuery.ai — Interactive Frontend, Multi-Route SPA Router, & Three.js Celestial Background
 */

// ============================================================================
// 1. Three.js Celestial Background Animation
// ============================================================================
(function initCelestialBackground() {
  const canvas = document.getElementById('sky');
  if (!canvas || typeof THREE === 'undefined') {
    console.warn('Three.js or #sky canvas not found.');
    return;
  }

  const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);
  camera.position.set(0, 14, 34);
  camera.lookAt(0, 0, 0);

  // Starfield
  const starGeo = new THREE.BufferGeometry();
  const starCount = 900;
  const positions = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 160;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 160;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 160;
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.18, transparent: true, opacity: 0.7 });
  scene.add(new THREE.Points(starGeo, starMat));

  const solarSystem = new THREE.Group();
  scene.add(solarSystem);

  // Glow Sprite behind Sun
  function makeGlowTexture() {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
    g.addColorStop(0, 'rgba(255,180,90,0.9)');
    g.addColorStop(0.4, 'rgba(255,120,40,0.35)');
    g.addColorStop(1, 'rgba(255,120,40,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);
    return new THREE.CanvasTexture(c);
  }

  const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: makeGlowTexture(), transparent: true, depthWrite: false }));
  glow.scale.set(18, 18, 1);
  solarSystem.add(glow);

  // Sun
  const sun = new THREE.Mesh(
    new THREE.SphereGeometry(2.1, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0xffb35c })
  );
  solarSystem.add(sun);
  const sunLight = new THREE.PointLight(0xffcf9e, 2.2, 100);
  solarSystem.add(sunLight);
  scene.add(new THREE.AmbientLight(0x404050, 0.6));

  const planetDefs = [
    { r: 0.35, dist: 4.2,  speed: 1.4,  color: 0x9aa4b8 },
    { r: 0.55, dist: 6.4,  speed: 1.0,  color: 0x5eead4 },
    { r: 0.5,  dist: 8.8,  speed: 0.72, color: 0x4f8cff },
    { r: 0.75, dist: 11.6, speed: 0.5,  color: 0xe8622c },
    { r: 0.9,  dist: 14.8, speed: 0.34, color: 0xd8c39a }
  ];

  const planets = planetDefs.map(def => {
    const orbitPts = [];
    const seg = 128;
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      orbitPts.push(new THREE.Vector3(Math.cos(a) * def.dist, 0, Math.sin(a) * def.dist));
    }
    const orbitGeo = new THREE.BufferGeometry().setFromPoints(orbitPts);
    const orbitMat = new THREE.LineBasicMaterial({ color: 0x3a4157, transparent: true, opacity: 0.5 });
    solarSystem.add(new THREE.LineLoop(orbitGeo, orbitMat));

    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(def.r, 24, 24),
      new THREE.MeshStandardMaterial({ color: def.color, roughness: 0.6, metalness: 0.1 })
    );
    solarSystem.add(mesh);
    return { mesh, dist: def.dist, speed: def.speed, angle: Math.random() * Math.PI * 2 };
  });

  solarSystem.rotation.x = 0.35;
  const clock = new THREE.Clock();

  function render() {
    const dt = clock.getDelta();
    planets.forEach(p => {
      p.angle += dt * p.speed * 0.35;
      p.mesh.position.set(Math.cos(p.angle) * p.dist, 0, Math.sin(p.angle) * p.dist);
      p.mesh.rotation.y += dt * 0.6;
    });
    sun.rotation.y += dt * 0.15;

    renderer.render(scene, camera);
    requestAnimationFrame(render);
  }
  render();

  function onScroll() {
    const max = document.body.scrollHeight - window.innerHeight;
    const t = max > 0 ? window.scrollY / max : 0;
    solarSystem.rotation.y = t * Math.PI * 1.4;
    solarSystem.rotation.z = t * 0.25;
    camera.position.y = 14 - t * 10;
    camera.position.z = 34 - t * 14;
    camera.lookAt(0, 0, 0);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
})();


// ============================================================================
// 2. SatQuery AI Multi-Route SPA, Safe DOM Utilities, & Agentic Logic
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  let currentMode = 'vqa'; // 'vqa', 'grounding', 'bitemporal_change', 'optical_sar_fusion'
  let cachedBoxes = [];
  let executionHistory = [];

  // ==========================================================================
  // Safe DOM Helper Functions
  // ==========================================================================
  function getElement(id) {
    const el = document.getElementById(id);
    if (!el) {
      console.warn(`[SatQuery UI] Element #${id} not found in DOM.`);
      return null;
    }
    return el;
  }

  function setText(id, value, fallback = '—') {
    const element = document.getElementById(id);
    if (!element) {
      console.error(`[SatQuery UI] Missing DOM element: #${id}`);
      return false;
    }
    element.textContent = (value !== undefined && value !== null) ? String(value) : fallback;
    return true;
  }

  function setHtml(id, htmlString) {
    const element = document.getElementById(id);
    if (!element) {
      console.error(`[SatQuery UI] Missing DOM element: #${id}`);
      return false;
    }
    element.innerHTML = htmlString !== undefined && htmlString !== null ? String(htmlString) : '';
    return true;
  }

  function toggleHidden(id, shouldHide) {
    const element = document.getElementById(id);
    if (!element) {
      console.warn(`[SatQuery UI] Missing DOM element to toggle visibility: #${id}`);
      return false;
    }
    if (shouldHide) {
      element.classList.add('hidden');
    } else {
      element.classList.remove('hidden');
    }
    return true;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function showAlert(msg, isError = true) {
    const alertBox = document.getElementById('alertBox');
    if (!alertBox) return;
    alertBox.textContent = msg;
    alertBox.className = `alert-box ${isError ? 'error' : ''}`;
    alertBox.classList.remove('hidden');
  }

  function clearAlert() {
    const alertBox = document.getElementById('alertBox');
    if (!alertBox) return;
    alertBox.classList.add('hidden');
    alertBox.textContent = '';
  }

  // DOM Form & Input Elements
  const vqaForm = document.getElementById('vqaForm');
  const imageInput = document.getElementById('imageInput');
  const secondaryImageInput = document.getElementById('secondaryImageInput');
  const dropZone = document.getElementById('dropZone');
  const dropZone2 = document.getElementById('dropZone2');
  const questionInput = document.getElementById('questionInput');
  const btnAnalyse = document.getElementById('btnAnalyse');
  const spinner = document.getElementById('spinner');
  const btnClearFile = document.getElementById('btnClearFile');
  const btnClearFile2 = document.getElementById('btnClearFile2');
  const toggleGrounding = document.getElementById('toggleGrounding');
  const toggleChange = document.getElementById('toggleChange');
  const btnRefreshProv = document.getElementById('btnRefreshProv');
  const btnExportJson = document.getElementById('btnExportJson');

  // Studio Profiles per Route
  const studioProfiles = {
    vqa: {
      badge: 'Ask Questions (VQA)',
      title: 'Ask Anything About a Satellite Image',
      subtitle: 'Upload any satellite image or GeoTIFF and ask questions in plain English.',
      panelTitle: 'Choose Your Image',
      panelSubtitle: 'Upload a satellite picture or select one of the sample images.',
      presets: [
        { label: 'What is here?', q: 'What land types and objects are visible in this image?' },
        { label: 'Any Water?', q: 'Are there any water bodies, lakes, or rivers here?' },
        { label: 'Buildings & Cities', q: 'Are there buildings, roads, or cities present?' },
        { label: 'Farms & Crops', q: 'Describe the farms, crops, and greenery in this scene.' }
      ]
    },
    grounding: {
      badge: 'Find Objects',
      title: 'Locate & Highlight Specific Areas',
      subtitle: 'Find specific features on the ground with glowing highlight boxes.',
      panelTitle: 'Choose Your Image',
      panelSubtitle: 'Upload an image where you want to find and mark specific things.',
      presets: [
        { label: 'Find Water', q: 'Find and highlight the water body in this image.' },
        { label: 'Find Buildings', q: 'Draw boxes around the buildings and urban areas.' },
        { label: 'Find Farms', q: 'Highlight the farm fields and crops.' },
        { label: 'Find Roads', q: 'Locate the roads and transportation lines.' }
      ]
    },
    bitemporal_change: {
      badge: 'Before & After Change',
      title: 'Spot Changes Over Time',
      subtitle: 'Compare two satellite images of the same area taken on different dates.',
      panelTitle: 'Upload Both Images',
      panelSubtitle: 'Upload the "before" image on the left and the "after" image on the right.',
      presets: [
        { label: 'Flood Damage', q: 'What flood or water changes happened between these two dates?' },
        { label: 'New Construction', q: 'Did the built-up city area grow or change between these dates?' },
        { label: 'Forest & Trees', q: 'What changes happened to the trees, plants, or greenery?' },
        { label: 'All Changes', q: 'What changed between these two pictures and where did it happen?' }
      ]
    },
    optical_sar_fusion: {
      badge: 'See Through Clouds',
      title: 'Combine Normal Photos with Radar',
      subtitle: 'Use radar to see ground details clearly even when thick clouds block normal cameras.',
      panelTitle: 'Upload Camera & Radar Images',
      panelSubtitle: 'Upload the cloudy optical picture on the left and the radar image on the right.',
      presets: [
        { label: 'Find Buildings in Clouds', q: 'Use both images together to find buildings through the clouds.' },
        { label: 'Map Water', q: 'Combine color and radar data to accurately map water bodies.' },
        { label: 'Check Farm Soil', q: 'Use optical color and radar roughness to inspect farm fields.' },
        { label: 'Full Summary', q: 'Give a complete summary using both the optical photo and radar data.' }
      ]
    }
  };

  // ==========================================================================
  // Router Implementation
  // ==========================================================================
  function navigateToRoute(route, pushState = true) {
    let cleanRoute = (route || '').replace(/^\//, '').trim().toLowerCase();
    if (!cleanRoute || cleanRoute === 'index.html') cleanRoute = 'overview';

    // Hide all views
    toggleHidden('viewOverview', true);
    toggleHidden('viewWorkspace', true);
    toggleHidden('viewProvenance', true);

    // Update Nav Link Active States
    document.querySelectorAll('.nav-link').forEach(link => {
      const linkRoute = (link.dataset.route || '').toLowerCase();
      if (linkRoute === cleanRoute || (cleanRoute === 'overview' && linkRoute === '')) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    if (cleanRoute === 'overview') {
      toggleHidden('viewOverview', false);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else if (cleanRoute === 'provenance') {
      toggleHidden('viewProvenance', false);
      loadProvenanceFull();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      // Workspace Studios: vqa, grounding, change, fusion
      toggleHidden('viewWorkspace', false);

      if (cleanRoute === 'vqa') currentMode = 'vqa';
      else if (cleanRoute === 'grounding') currentMode = 'grounding';
      else if (cleanRoute === 'change' || cleanRoute === 'bitemporal_change') currentMode = 'bitemporal_change';
      else if (cleanRoute === 'fusion' || cleanRoute === 'optical_sar_fusion') currentMode = 'optical_sar_fusion';
      else currentMode = 'vqa';

      updateModeUI();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    if (pushState) {
      const targetUrl = cleanRoute === 'overview' ? '/' : `/${cleanRoute}`;
      history.pushState({ route: cleanRoute }, '', targetUrl);
    }
  }

  // Intercept Nav Links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const route = link.dataset.route || link.getAttribute('href');
      navigateToRoute(route, true);
    });
  });

  window.addEventListener('popstate', () => {
    const currentPath = window.location.pathname.replace(/^\//, '') || 'overview';
    navigateToRoute(currentPath, false);
  });

  function updateModeUI() {
    clearAlert();
    clearGroundingCanvas();
    toggleHidden('changeOverlay', true);
    toggleHidden('toggleChangeLayer', true);

    const profile = studioProfiles[currentMode] || studioProfiles.vqa;

    setText('workspaceBadge', profile.badge);
    setText('workspaceTitle', profile.title);
    setText('workspaceSubtitle', profile.subtitle);
    setText('panelTitle', profile.panelTitle);
    setText('panelSubtitle', profile.panelSubtitle);

    if (currentMode === 'bitemporal_change') {
      toggleHidden('secondaryGroup', false);
      setText('labelPrimaryImage', 'First Image (Before Date)');
      setText('dropTextPrimary', 'Click or drop the Before image here');
      setText('labelSecondaryImage', 'Second Image (After Date)');
      setText('dropTextSecondary', 'Click or drop the After image here');
    } else if (currentMode === 'optical_sar_fusion') {
      toggleHidden('secondaryGroup', false);
      setText('labelPrimaryImage', 'Cloudy Camera Image (Optical)');
      setText('dropTextPrimary', 'Click or drop the Optical image here');
      setText('labelSecondaryImage', 'Radar Image (SAR)');
      setText('dropTextSecondary', 'Click or drop the Radar image here');
    } else {
      toggleHidden('secondaryGroup', true);
      setText('labelPrimaryImage', 'Satellite Image (GeoTIFF, PNG, JPEG)');
      setText('dropTextPrimary', 'Click or drop your satellite image here');
    }

    // Populate Presets
    const presets = profile.presets || [];
    const quickQuestionsEl = document.getElementById('quickQuestions');
    if (quickQuestionsEl) {
      quickQuestionsEl.innerHTML = '<span class="quick-title">Quick Examples:</span>' + presets.map(p => `
        <button type="button" class="chip" data-q="${escapeHtml(p.q)}">${escapeHtml(p.label)}</button>
      `).join('');

      quickQuestionsEl.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
          if (questionInput) {
            questionInput.value = chip.dataset.q;
            questionInput.focus();
          }
        });
      });
    }

    if (presets.length > 0 && questionInput) {
      questionInput.placeholder = presets[0].q;
    }
  }

  // Fetch Health & Active Model
  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        const activeModel = data.active_model || 'Ready';
        const modelLabel = activeModel.includes('mock') ? `${activeModel} (Mock)` : activeModel;
        setText('modelBadge', modelLabel);
      } else {
        setText('modelBadge', 'Backend Offline');
      }
    } catch (e) {
      setText('modelBadge', 'Offline');
    }
  }

  // Fetch Provenance Feed
  async function loadProvenance() {
    try {
      const res = await fetch('/executions?limit=5');
      if (res.ok) {
        const data = await res.json();
        executionHistory = data;
      }
    } catch (e) {
      console.warn('Failed to load provenance feed', e);
    }
  }

  // Fullscreen Provenance Feed
  async function loadProvenanceFull() {
    try {
      const res = await fetch('/executions?limit=50');
      if (res.ok) {
        const data = await res.json();
        executionHistory = data;
        const container = document.getElementById('provenanceFeedFull');
        if (container) {
          if (!Array.isArray(data) || data.length === 0) {
            container.innerHTML = '<div class="prov-empty">No executions recorded yet. Launch a studio above to run queries!</div>';
            return;
          }
          container.innerHTML = data.map((item, idx) => `
            <div class="prov-item">
              <div>
                <div><strong>#${data.length - idx} [${escapeHtml((item.task || 'vqa').toUpperCase())}]</strong> — <em>${escapeHtml(item.model || '—')}</em></div>
                <div style="color: #c5cedd; margin-top: 4px;"><strong>Query:</strong> "${escapeHtml(item.question || '')}"</div>
                <div style="color: #9aa4b8; font-size: 12px; margin-top: 2px;"><strong>Files:</strong> ${escapeHtml(item.input || '—')}</div>
                <div style="color: #5eead4; margin-top: 4px;"><strong>Result:</strong> ${escapeHtml(item.output || '—')}</div>
              </div>
              <div style="text-align: right; flex-shrink: 0;">
                <div style="color: #ff9a56; font-weight: 600;">${item.execution_time_sec !== undefined ? item.execution_time_sec + 's' : '—'}</div>
                <div style="font-size: 11px; color: #6f7a8c; margin-top: 4px;">${item.timestamp ? item.timestamp.replace('T', ' ').split('.')[0] : ''}</div>
              </div>
            </div>
          `).join('');
        }
      }
    } catch (e) {
      console.warn('Failed to load full provenance', e);
    }
  }

  // Export JSON functionality
  if (btnExportJson) {
    btnExportJson.addEventListener('click', () => {
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(executionHistory, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", "satquery_executions_ledger.json");
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    });
  }

  // Handle File Upload & Server-side Preview
  async function handleFile(file, isSecondary = false) {
    clearAlert();
    if (!file) return;

    if (!isSecondary) {
      setText('fileName', `${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
      toggleHidden('fileInfo', false);
    } else {
      setText('fileName2', `${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
      toggleHidden('fileInfo2', false);
    }

    const imagePreview = document.getElementById('imagePreview');
    const previewPlaceholder = document.getElementById('previewPlaceholder');

    if (!isSecondary) {
      const isTiff = file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');
      if (!isTiff && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          if (imagePreview) {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
          }
          if (previewPlaceholder) previewPlaceholder.classList.add('hidden');
        };
        reader.readAsDataURL(file);
      } else {
        if (imagePreview) imagePreview.classList.add('hidden');
        if (previewPlaceholder) {
          previewPlaceholder.classList.remove('hidden');
          const spanText = previewPlaceholder.querySelector('span');
          if (spanText) spanText.textContent = `Normalizing GeoTIFF bands for ${file.name}...`;
        }
      }
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('/preview', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        if (!isSecondary && data.preview_url && imagePreview) {
          imagePreview.src = data.preview_url;
          imagePreview.classList.remove('hidden');
          if (previewPlaceholder) previewPlaceholder.classList.add('hidden');
        }
        if (!isSecondary && data.metadata) {
          setText('metaCrs', data.metadata.crs, 'Local / None');
          const shapeVal = Array.isArray(data.metadata.shape) ? `[${data.metadata.shape.join(', ')}]` : (data.metadata.shape || '—');
          setText('metaShape', shapeVal);
          setText('metaBands', data.metadata.bands || data.metadata.count, '—');
          setText('metaDriver', data.metadata.driver, 'Raster');
        }
      }
    } catch (err) {
      console.warn('Failed to generate GeoTIFF preview', err);
    }
  }

  function resetFileSelection(isSecondary = false) {
    if (!isSecondary) {
      if (imageInput) imageInput.value = '';
      toggleHidden('fileInfo', true);
      toggleHidden('imagePreview', true);
      clearGroundingCanvas();
      toggleHidden('changeOverlay', true);
      toggleHidden('previewPlaceholder', false);
      const spanText = document.querySelector('#previewPlaceholder span');
      if (spanText) spanText.textContent = 'Upload or select a scene to preview raster and extract coordinates';
      setText('metaCrs', '—');
      setText('metaShape', '—');
      setText('metaBands', '—');
      setText('metaDriver', '—');
    } else {
      if (secondaryImageInput) secondaryImageInput.value = '';
      toggleHidden('fileInfo2', true);
    }
  }

  // Draw Visual Grounding Bounding Boxes & Segmentation Masks
  let cachedMaskUrl = null;

  function drawGroundingBoxes(boxes, maskUrl = null) {
    const canvas = document.getElementById('groundingCanvas');
    const imagePreview = document.getElementById('imagePreview');
    const layerControls = document.getElementById('layerControls');
    const toggleGrounding = document.getElementById('toggleGrounding');

    if (!canvas) return;

    if ((!Array.isArray(boxes) || boxes.length === 0) && !maskUrl) {
      clearGroundingCanvas();
      if (layerControls) layerControls.classList.add('hidden');
      return;
    }

    cachedBoxes = boxes || [];
    if (maskUrl !== undefined) cachedMaskUrl = maskUrl;
    if (layerControls) layerControls.classList.remove('hidden');
    canvas.classList.remove('hidden');

    const w = (imagePreview && imagePreview.clientWidth) || 300;
    const h = (imagePreview && imagePreview.clientHeight) || 240;

    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);

    if (toggleGrounding && !toggleGrounding.checked) return;

    // Draw segmentation mask overlay if available
    if (cachedMaskUrl) {
      const maskImg = new Image();
      maskImg.onload = () => {
        if (toggleGrounding && !toggleGrounding.checked) return;
        ctx.drawImage(maskImg, 0, 0, w, h);
        renderBoxOutlines(ctx, cachedBoxes, w, h);
      };
      maskImg.src = cachedMaskUrl;
    } else {
      renderBoxOutlines(ctx, cachedBoxes, w, h);
    }
  }

  function renderBoxOutlines(ctx, boxes, w, h) {
    if (!Array.isArray(boxes)) return;
    boxes.forEach((box) => {
      if (!box) return;
      const xmin = box.xmin !== undefined ? box.xmin : 0;
      const ymin = box.ymin !== undefined ? box.ymin : 0;
      const xmax = box.xmax !== undefined ? box.xmax : 1;
      const ymax = box.ymax !== undefined ? box.ymax : 1;

      const bx = xmin * w;
      const by = ymin * h;
      const bw = (xmax - xmin) * w;
      const bh = (ymax - ymin) * h;

      // Glow outline
      ctx.shadowColor = '#5eead4';
      ctx.shadowBlur = 10;
      ctx.strokeStyle = '#5eead4';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(bx, by, bw, bh);

      // Translucent fill
      ctx.fillStyle = 'rgba(94, 234, 212, 0.10)';
      ctx.fillRect(bx, by, bw, bh);

      // Label badge
      ctx.shadowBlur = 0;
      const confStr = typeof box.confidence === 'number' ? ` (${Math.round(box.confidence * 100)}%)` : '';
      const covStr = typeof box.mask_coverage_pct === 'number' ? ` [Mask: ${box.mask_coverage_pct}%]` : (typeof box.bbox_coverage_pct === 'number' ? ` [BBox: ${box.bbox_coverage_pct}%]` : '');
      const label = `${box.label || 'Target'}${confStr}${covStr}`;
      ctx.font = '11px JetBrains Mono, monospace';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
      ctx.fillRect(bx, Math.max(0, by - 20), textWidth + 12, 18);
      ctx.strokeStyle = '#5eead4';
      ctx.lineWidth = 1;
      ctx.strokeRect(bx, Math.max(0, by - 20), textWidth + 12, 18);

      ctx.fillStyle = '#5eead4';
      ctx.fillText(label, bx + 6, Math.max(13, by - 6));
    });
  }

  function clearGroundingCanvas() {
    cachedBoxes = [];
    cachedMaskUrl = null;
    const canvas = document.getElementById('groundingCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.classList.add('hidden');
  }

  if (toggleGrounding) {
    toggleGrounding.addEventListener('change', () => {
      drawGroundingBoxes(cachedBoxes, cachedMaskUrl);
    });
  }

  if (toggleChange) {
    toggleChange.addEventListener('change', () => {
      const changeOverlay = document.getElementById('changeOverlay');
      if (changeOverlay) {
        if (toggleChange.checked) {
          changeOverlay.classList.remove('hidden');
        } else {
          changeOverlay.classList.add('hidden');
        }
      }
    });
  }

  // Render Observable Execution Trace
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
      const stepNum = s && (s.step_id !== undefined ? s.step_id : (s.step !== undefined ? s.step : '•'));
      const action = s && (s.name || s.action || 'Step');
      const details = s && (s.description || (typeof s.details === 'string' ? s.details : (s.details ? JSON.stringify(s.details) : '')) || '');
      const toolBadge = s && s.tool ? `<span class="badge-tag">${escapeHtml(s.tool)}</span>` : '';
      const provBadge = s && s.provenance ? `<span class="badge-tag prov">${escapeHtml(String(s.provenance).toUpperCase())}</span>` : '';
      return `
        <div class="trace-step-item">
          <span class="trace-step-num">${escapeHtml(String(stepNum))}</span>
          <div class="trace-step-content">
            <strong>${escapeHtml(action)}</strong> ${toolBadge} ${provBadge}
            <div class="trace-step-desc">${escapeHtml(details)}</div>
          </div>
        </div>
      `;
    }).join('');
  }

  // ==========================================================================
  // Render Result Payload Safely
  // ==========================================================================
  function renderAgentResult(data) {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid response structure received from backend.');
    }

    // 1. Text & Tags Display
    setText('resAnswer', data.answer, 'No answer returned.');
    const displayTask = (data.task || currentMode || 'VQA').toUpperCase().replace(/_/g, ' ');
    setText('resTask', displayTask, 'UNKNOWN');
    setText('resTool', data.tool_used, 'Specialist Tool');
    
    // Model & Provenance Tags
    const modelStr = data.model || '—';
    setText('resModel', modelStr, '—');

    const modeStr = data.model_mode || (data.execution_trace && data.execution_trace.model_mode) || '—';
    setText('resMode', modeStr.toUpperCase(), '—');

    const adaptStr = data.rs_adaptation || (data.execution_trace && data.execution_trace.rs_adaptation) || 'NOT LOADED';
    setText('resAdaptation', adaptStr, 'NOT LOADED');

    const evidenceStr = data.evidence_used || (data.execution_trace && data.execution_trace.evidence_used) || 'NONE';
    setText('resEvidence', evidenceStr, 'NONE');

    // Latency
    const latencyStr = data.execution_time_sec !== undefined && data.execution_time_sec !== null
      ? `${data.execution_time_sec}s`
      : '—';
    setText('resLatency', latencyStr, '—');

    // Confidence - formatted defensively (never fake confidence)
    let confText = '—';
    if (typeof data.confidence === 'number' && !isNaN(data.confidence)) {
      confText = `${Math.round(data.confidence * 100)}%`;
    } else if (typeof data.confidence === 'string' && data.confidence.trim().length > 0) {
      confText = data.confidence;
    }
    setText('resConfidence', confText, '—');

    // Input Summary
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
    } else if (secondaryImageInput && secondaryImageInput.files && secondaryImageInput.files.length > 0) {
      inputSummary = '2 inputs';
    }
    setText('resInput', inputSummary, '—');

    // Routing Reason Banner
    const reasonBanner = document.getElementById('routingReasonBanner');
    if (reasonBanner) {
      if (data.execution_trace && data.execution_trace.routing_reason) {
        setText('routingReasonText', data.execution_trace.routing_reason, '—');
        reasonBanner.classList.remove('hidden');
      } else {
        reasonBanner.classList.add('hidden');
      }
    }

    // Visual Grounding Bounding Boxes & Segmentation Masks
    if ((Array.isArray(data.boxes) && data.boxes.length > 0) || data.grounding_mask_url) {
      setTimeout(() => drawGroundingBoxes(data.boxes, data.grounding_mask_url), 100);
    } else {
      clearGroundingCanvas();
    }

    // Change Heatmap Overlay
    const changeOverlay = document.getElementById('changeOverlay');
    const toggleChangeLayer = document.getElementById('toggleChangeLayer');
    const layerControls = document.getElementById('layerControls');
    if (data.change_map_url) {
      if (changeOverlay) {
        changeOverlay.src = data.change_map_url;
        changeOverlay.classList.remove('hidden');
      }
      if (toggleChangeLayer) toggleChangeLayer.classList.remove('hidden');
      if (layerControls) layerControls.classList.remove('hidden');
    } else {
      if (changeOverlay) changeOverlay.classList.add('hidden');
      if (toggleChangeLayer) toggleChangeLayer.classList.add('hidden');
    }

    // Observable Agent Execution Trace
    renderAgentTrace(data.execution_trace);

    // Display Metadata
    if (data.metadata && typeof data.metadata === 'object') {
      setText('metaCrs', data.metadata.crs, 'Non-georeferenced');
      const shapeStr = Array.isArray(data.metadata.shape) ? `[${data.metadata.shape.join(', ')}]` : (data.metadata.shape || '—');
      setText('metaShape', shapeStr, '—');
      setText('metaBands', data.metadata.count || data.metadata.bands, '—');
      setText('metaDriver', data.metadata.driver, '—');
    }

    loadProvenance();
  }

  // Event Listeners for File Selection
  if (imageInput) {
    imageInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFile(e.target.files[0], false);
      }
    });
  }

  if (secondaryImageInput) {
    secondaryImageInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFile(e.target.files[0], true);
      }
    });
  }

  if (btnClearFile) {
    btnClearFile.addEventListener('click', () => resetFileSelection(false));
  }
  if (btnClearFile2) {
    btnClearFile2.addEventListener('click', () => resetFileSelection(true));
  }

  // Drag & Drop Handling
  if (dropZone) {
    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        if (imageInput) imageInput.files = e.dataTransfer.files;
        handleFile(e.dataTransfer.files[0], false);
      }
    });
  }

  if (dropZone2) {
    ['dragenter', 'dragover'].forEach(name => {
      dropZone2.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone2.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropZone2.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone2.classList.remove('drag-over');
      });
    });

    dropZone2.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        if (secondaryImageInput) secondaryImageInput.files = e.dataTransfer.files;
        handleFile(e.dataTransfer.files[0], true);
      }
    });
  }

  // Form Submission Pipeline with Structured Error Handling
  if (vqaForm) {
    vqaForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      clearAlert();
      clearGroundingCanvas();
      toggleHidden('changeOverlay', true);
      toggleHidden('toggleChangeLayer', true);

      if (!imageInput || !imageInput.files || imageInput.files.length === 0) {
        showAlert('Please select or upload a primary satellite image.');
        return;
      }

      if ((currentMode === 'bitemporal_change' || currentMode === 'optical_sar_fusion') && (!secondaryImageInput || !secondaryImageInput.files || secondaryImageInput.files.length === 0)) {
        showAlert(`Please upload the secondary scene required for ${currentMode.replace('_', ' ').toUpperCase()} analysis.`);
        return;
      }

      const question = questionInput ? questionInput.value.trim() : '';
      if (!question) {
        showAlert('Please enter a natural language question.');
        return;
      }

      const formData = new FormData();
      formData.append('image', imageInput.files[0]);
      formData.append('question', question);
      formData.append('task_mode', currentMode);

      if (secondaryImageInput && secondaryImageInput.files && secondaryImageInput.files.length > 0) {
        formData.append('secondary_image', secondaryImageInput.files[0]);
      }

      // UI Loading state - reset all previous result state cleanly
      if (btnAnalyse) btnAnalyse.disabled = true;
      if (spinner) spinner.classList.remove('hidden');
      setText('resAnswer', 'Agentic orchestrator is classifying intent and executing specialist remote-sensing tools...');
      setText('resTask', '—');
      setText('resTool', '—');
      setText('resModel', '—');
      setText('resMode', '—');
      setText('resAdaptation', '—');
      setText('resEvidence', '—');
      setText('resConfidence', '—');
      setText('resLatency', '—');
      setText('routingReasonText', '—');
      toggleHidden('routingReasonBanner', true);
      toggleHidden('agentTraceContainer', true);

      let responseData = null;
      let requestError = null;

      try {
        let response;
        try {
          response = await fetch('/agent/analyze', {
            method: 'POST',
            body: formData,
          });
        } catch (netErr) {
          throw new Error(`NETWORK_ERROR: ${netErr.message || 'Could not connect to SatQuery backend.'}`);
        }

        try {
          responseData = await response.json();
        } catch (jsonErr) {
          throw new Error(`API_ERROR: Invalid JSON response from server (${response.status} ${response.statusText})`);
        }

        if (!response.ok) {
          const detail = responseData && responseData.detail ? responseData.detail : `HTTP ${response.status} ${response.statusText}`;
          throw new Error(`API_ERROR: ${detail}`);
        }
      } catch (err) {
        requestError = err;
      }

      if (requestError) {
        const fullMsg = requestError.message || 'Unknown network error';
        if (fullMsg.startsWith('NETWORK_ERROR:')) {
          const cleanMsg = fullMsg.replace('NETWORK_ERROR:', '').trim();
          showAlert(cleanMsg, true);
          setText('resAnswer', 'Could not connect to SatQuery backend. Ensure the server is running on http://127.0.0.1:8000.');
        } else if (fullMsg.startsWith('API_ERROR:')) {
          const cleanMsg = fullMsg.replace('API_ERROR:', '').trim();
          showAlert(cleanMsg, true);
          setText('resAnswer', `Inference failed: ${cleanMsg}`);
        } else {
          showAlert(fullMsg, true);
          setText('resAnswer', `Inference failed: ${fullMsg}`);
        }
      } else {
        // Backend inference succeeded; render results safely
        try {
          renderAgentResult(responseData);
        } catch (renderErr) {
          console.error('[SatQuery UI] Failed to render result:', renderErr);
          showAlert('Backend inference succeeded, but the result could not be rendered.', true);
          setText('resAnswer', 'Backend inference succeeded, but the result could not be rendered.');
        }
      }

      if (btnAnalyse) btnAnalyse.disabled = false;
      if (spinner) spinner.classList.add('hidden');
    });
  }

  if (btnRefreshProv) {
    btnRefreshProv.addEventListener('click', () => {
      loadProvenance();
      loadProvenanceFull();
    });
  }

  // Initial Route Hydration from URL
  const initialPath = window.location.pathname.replace(/^\//, '') || 'overview';
  navigateToRoute(initialPath, false);

  checkHealth();
  loadProvenance();
});


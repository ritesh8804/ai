// VSales – LinkedIn Lead Helper (Popup Logic)
// Ensure manifest.json has permissions: ["storage","activeTab","tabs"]
// and host_permissions: ["https://www.linkedin.com/*","https://script.google.com/*","https://script.googleusercontent.com/*"]

// ========= CONFIG =========
// Replace with your deployed Apps Script Web App URL (Deploy > Web app)
const APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbzhYOl_Hpa7lub2dTZN1lM2SltUXLr0kMYeRT8VAUPgW_6PAffMjcKpxQ85cMaVSHxW/exec';

// ========= STATE =========
let scraped = { url: '', name: '', jobTitle: '', companyName: '' };
let sheet = { url: '', name: '', jobTitle: '', companyName: '', phone: '', email: '' };

// ========= DOM =========
let urlInput, searchBtn,
  nameInput, jobInput, companyInput, phoneInput, emailInput,
  namePlus, jobPlus, companyPlus, phonePlus, emailPlus,
  nameStatus, jobStatus, companyStatus, phoneStatus, emailStatus,
  banner, updateBtn, statusEl, hintEl,
  lastUpdatedValueEl, refreshBtn, metricsCard, metricsBody;

// ========= HELPERS =========
function sel(id) { return document.getElementById(id); }

function setStatus(msg, ok = true) {
  if (!statusEl) return;
  statusEl.textContent = msg || '';
  statusEl.style.color = ok ? '#0d6b3a' : '#a11622';
  if (!msg) return;
  setTimeout(() => (statusEl.textContent = ''), 3000);
}

function setBanner(text, type) {
  if (!banner) return;
  banner.textContent = text || '';
  banner.className = 'banner' + (type ? ' ' + type : '');
}

function validUrl(u) {
  return /linkedin\.com\/in\//i.test(u || '');
}

function computeMismatches() {
  const mm = {
    name: !!(scraped.name && sheet.name && scraped.name.trim() !== sheet.name.trim()),
    job: !!(scraped.jobTitle && sheet.jobTitle && scraped.jobTitle.trim() !== sheet.jobTitle.trim()),
    company: !!(scraped.companyName && sheet.companyName && scraped.companyName.trim() !== sheet.companyName.trim())
  };
  const any = mm.name || mm.job || mm.company;
  return { ...mm, any };
}

function renderIndicators() {
  const mm = computeMismatches();

  // Name
  if (!sheet.name) { nameStatus.textContent = '❗'; nameStatus.className = 'affix status-dot warn'; namePlus.classList.remove('hidden'); }
  else if (mm.name) { nameStatus.textContent = '❗'; nameStatus.className = 'affix status-dot warn'; namePlus.classList.remove('hidden'); }
  else { nameStatus.textContent = '✅'; nameStatus.className = 'affix status-dot ok'; namePlus.classList.add('hidden'); }

  // Job
  if (!sheet.jobTitle) { jobStatus.textContent = '❗'; jobStatus.className = 'affix status-dot warn'; jobPlus.classList.remove('hidden'); }
  else if (mm.job) { jobStatus.textContent = '❗'; jobStatus.className = 'affix status-dot warn'; jobPlus.classList.remove('hidden'); }
  else { jobStatus.textContent = '✅'; jobStatus.className = 'affix status-dot ok'; jobPlus.classList.add('hidden'); }

  // Company
  if (!sheet.companyName) { companyStatus.textContent = '❗'; companyStatus.className = 'affix status-dot warn'; companyPlus.classList.remove('hidden'); }
  else if (mm.company) { companyStatus.textContent = '❗'; companyStatus.className = 'affix status-dot warn'; companyPlus.classList.remove('hidden'); }
  else { companyStatus.textContent = '✅'; companyStatus.className = 'affix status-dot ok'; companyPlus.classList.add('hidden'); }

  // Phone & Email presence
  phoneStatus.textContent = phoneInput.value ? '✅' : '❗';
  phoneStatus.className = 'affix status-dot ' + (phoneInput.value ? 'ok' : 'warn');
  emailStatus.textContent = emailInput.value ? '✅' : '❗';
  emailStatus.className = 'affix status-dot ' + (emailInput.value ? 'ok' : 'warn');

  if (mm.any) setBanner('Update Database', 'danger');
  else setBanner('In sync with Database', 'ok');
}

function fillInputsFromSheet() {
  nameInput.value = sheet.name || '';
  jobInput.value = sheet.jobTitle || '';
  companyInput.value = sheet.companyName || '';
  phoneInput.value = sheet.phone || '';
  emailInput.value = sheet.email || '';
  renderIndicators();
}

function applyScrapedToInputsIfEmpty() {
  if (!nameInput.value && scraped.name) nameInput.value = scraped.name;
  if (!jobInput.value && scraped.jobTitle) jobInput.value = scraped.jobTitle;
  if (!companyInput.value && scraped.companyName) companyInput.value = scraped.companyName;
  renderIndicators();
}

function formatDate(value) {
  if (!value) return '-';
  const d = new Date(value);
  if (isNaN(d.getTime())) return String(value);
  // dd/mm/yyyy like the screenshot
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' });
}

function renderLastUpdated(rec) {
  if (!lastUpdatedValueEl) return;
  const v = rec?.updated_at || rec?.updatedAt || '';
  lastUpdatedValueEl.textContent = formatDate(v);
}

function renderMetrics(data) {
  if (!metricsBody || !metricsCard) return;
  
  console.log('Rendering metrics with data:', data);
  
  let html = '';
  
  if (data && data.metrics && Array.isArray(data.metrics)) {
    // Use the metrics array from the JSON response
    data.metrics.forEach(metric => {
      html += `
        <tr>
          <td>${metric.source}</td>
          <td>${metric.bounce}</td>
          <td>${metric.unsubscribed}</td>
          <td>${metric.clicked}</td>
          <td>${metric.enquired}</td>
        </tr>
      `;
    });
  } else {
    // Default empty data
    html = `
      <tr>
        <td>Zobble</td>
        <td>0</td>
        <td>No</td>
        <td>0</td>
        <td>0</td>
      </tr>
      <tr>
        <td>Violet</td>
        <td>0</td>
        <td>No</td>
        <td>0</td>
        <td>0</td>
      </tr>
      <tr>
        <td>WOM</td>
        <td>0</td>
        <td>No</td>
        <td>0</td>
        <td>0</td>
      </tr>
    `;
  }
  
  metricsBody.innerHTML = html;
  metricsCard.style.display = 'block';
}

// ========= APPS SCRIPT API =========
async function fetchSheetRecord(linkedinUrl) {
  if (!APPS_SCRIPT_URL || APPS_SCRIPT_URL.includes('YOUR_GOOGLE_APPS_SCRIPT_WEB_APP_URL_HERE')) {
    if (hintEl) hintEl.textContent = 'Configure your Apps Script Web App URL in popup.js (APPS_SCRIPT_URL).';
    return null;
  }
  const url = APPS_SCRIPT_URL + '?action=getbyurl&url=' + encodeURIComponent(linkedinUrl);
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) throw new Error('Sheet GET failed: ' + res.status);
  const json = await res.json();
  return json && json.data ? json.data : null;
}

async function upsertSheetRecord(payload) {
  if (!APPS_SCRIPT_URL || APPS_SCRIPT_URL.includes('YOUR_GOOGLE_APPS_SCRIPT_WEB_APP_URL_HERE')) {
    throw new Error('Apps Script Web App URL not configured');
  }
  const res = await fetch(APPS_SCRIPT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'upsert', data: payload })
  });
  if (!res.ok) throw new Error('Sheet POST failed: ' + res.status);
  const json = await res.json();
  return json;
}

function buildPayload() {
  return {
    linkedin_url: (urlInput.value || '').trim(),
    name: (nameInput.value || '').trim(),
    job_title: (jobInput.value || '').trim(),
    company_name: (companyInput.value || '').trim(),
    phone: (phoneInput.value || '').trim(),
    email: (emailInput.value || '').trim()
  };
}

// ========= LINKEDIN SCRAPE INTEGRATION =========
async function detectActiveLinkedInUrl() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url && validUrl(tab.url)) {
      urlInput.value = tab.url;
      scraped.url = tab.url;
      return tab.url;
    }
  } catch (e) {}
  return '';
}

async function scrapeFromActiveTab() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !validUrl(tab.url)) return null;
    return await new Promise((resolve) => {
      chrome.tabs.sendMessage(tab.id, { action: 'scrapeProfile' }, (response) => {
        if (chrome.runtime.lastError) return resolve(null);
        if (!response || !response.success) return resolve(null);
        resolve(response.data);
      });
    });
  } catch (e) {
    return null;
  }
}

function deriveJobInfo(profile) {
  let jobTitle = profile.currentPosition || '';
  let companyName = profile.currentCompany || '';
  if (!jobTitle || !companyName) {
    const exp = Array.isArray(profile.experience) ? profile.experience : [];
    if (exp.length) {
      jobTitle = jobTitle || (exp[0] && exp[0].title) || '';
      companyName = companyName || (exp[0] && exp[0].company) || '';
    }
  }
  return { jobTitle, companyName };
}

// ========= EVENTS =========
function attachEvents() {
  searchBtn.addEventListener('click', async () => {
    const url = (urlInput.value || '').trim();
    if (!validUrl(url)) { setStatus('Enter a valid LinkedIn profile URL', false); return; }
    setStatus('Searching...');
    try {
      const response = await fetchSheetRecord(url);
      if (response) {
        sheet = { url, name: response.name || '', jobTitle: response.job_title || '', companyName: response.company_name || '', phone: response.phone || '', email: response.email || '' };
        fillInputsFromSheet();
        renderLastUpdated(response);
        renderMetrics(response);
        setStatus('Loaded from Google Sheet');
      } else {
        sheet = { url, name: '', jobTitle: '', companyName: '', phone: '', email: '' };
        setStatus('No record in Sheet');
      }
      applyScrapedToInputsIfEmpty();
      renderIndicators(); // Show comparison after search
    } catch (e) {
      setStatus(e.message || 'Lookup failed', false);
    }
  });

  updateBtn.addEventListener('click', async () => {
    const url = (urlInput.value || '').trim();
    if (!validUrl(url)) { setStatus('Enter a valid LinkedIn profile URL', false); return; }
    try {
      const payload = buildPayload();
      await upsertSheetRecord(payload);
      setStatus('Database updated');
      sheet = { url, name: payload.name, jobTitle: payload.job_title, companyName: payload.company_name, phone: payload.phone, email: payload.email };
      renderIndicators();
    } catch (e) {
      setStatus(e.message || 'Update failed', false);
    }
  });

  namePlus.addEventListener('click', async () => { if (!scraped.name) return; nameInput.value = scraped.name; updateBtn.click(); });
  jobPlus.addEventListener('click', async () => { if (!scraped.jobTitle) return; jobInput.value = scraped.jobTitle; updateBtn.click(); });
  companyPlus.addEventListener('click', async () => { if (!scraped.companyName) return; companyInput.value = scraped.companyName; updateBtn.click(); });
  phonePlus.addEventListener('click', async () => { updateBtn.click(); });
  emailPlus.addEventListener('click', async () => { updateBtn.click(); });

  refreshBtn?.addEventListener('click', async () => {
    const url = (urlInput.value || '').trim();
    if (!validUrl(url)) { setStatus('Enter a valid LinkedIn profile URL', false); return; }
    try {
      const rec = await fetchSheetRecord(url);
      sheet = { url, name: rec?.name || '', jobTitle: rec?.job_title || '', companyName: rec?.company_name || '', phone: rec?.phone || '', email: rec?.email || '' };
      fillInputsFromSheet();
      renderLastUpdated(rec);
      renderMetrics(rec);
      setStatus('Refreshed');
    } catch (e) {
      setStatus(e.message || 'Refresh failed', false);
    }
  });

  [nameInput, jobInput, companyInput, phoneInput, emailInput].forEach(el => el.addEventListener('input', renderIndicators));
}

// ========= INIT =========
async function init() {
  urlInput = sel('urlInput');
  searchBtn = sel('searchBtn');
  nameInput = sel('nameInput');
  jobInput = sel('jobInput');
  companyInput = sel('companyInput');
  phoneInput = sel('phoneInput');
  emailInput = sel('emailInput');
  namePlus = sel('namePlus');
  jobPlus = sel('jobPlus');
  companyPlus = sel('companyPlus');
  phonePlus = sel('phonePlus');
  emailPlus = sel('emailPlus');
  nameStatus = sel('nameStatus');
  jobStatus = sel('jobStatus');
  companyStatus = sel('companyStatus');
  phoneStatus = sel('phoneStatus');
  emailStatus = sel('emailStatus');
  banner = sel('banner');
  updateBtn = sel('updateBtn');
  statusEl = sel('status');
  hintEl = sel('hint');
  lastUpdatedValueEl = sel('lastUpdatedValue');
  refreshBtn = sel('refreshBtn');
  metricsCard = sel('metricsCard');
  metricsBody = sel('metricsBody');

  attachEvents();

  // 1) Autofill URL from active tab
  await detectActiveLinkedInUrl();

  // 2) Scrape from page
  const data = await scrapeFromActiveTab();
  if (data) {
    scraped.name = data.name || '';
    const dj = deriveJobInfo(data);
    scraped.jobTitle = dj.jobTitle || '';
    scraped.companyName = dj.companyName || '';
  }

  // 3) If URL present, try sheet lookup
  if (scraped.url) {
    try {
      const response = await fetchSheetRecord(scraped.url);
      if (response) {
        sheet = { url: scraped.url, name: response.name || '', jobTitle: response.job_title || '', companyName: response.company_name || '', phone: response.phone || '', email: response.email || '' };
        renderLastUpdated(response);
        renderMetrics(response);
      }
    } catch (e) {}
  }

  // 4) Fill inputs and show indicators
  fillInputsFromSheet();
  applyScrapedToInputsIfEmpty();
  renderIndicators(); // Show comparison indicators

  // 5) Hints
  if (!APPS_SCRIPT_URL || APPS_SCRIPT_URL.includes('YOUR_GOOGLE_APPS_SCRIPT_WEB_APP_URL_HERE')) {
    hintEl.textContent = 'Tip: Deploy the Apps Script Web App and paste its URL in popup.js (APPS_SCRIPT_URL) to enable Sheet sync.';
  } else {
    hintEl.textContent = '';
  }
}

document.addEventListener('DOMContentLoaded', init);

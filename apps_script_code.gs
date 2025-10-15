/**
 * LinkedIn Extension Sheet API (CORS-friendly for Chrome Extensions)
 * - GET/POST endpoints for lookup and updates
 * - Accepts JSON and application/x-www-form-urlencoded
 * - JSONP support via callback=... on GET
 * - Normalized LinkedIn URL matching
 * - Optional Gemini job-title prediction (fixes API key usage)
 *
 * Replace SPREADSHEET_ID and (optionally) GEMINI_API_KEY before deploying.
 */

// ---------- CONFIG ----------
const SPREADSHEET_ID = '11ELmuu-XV1QNvdKy8dxhsAEIv7bozoNHvQscNlL_OEE'; // <<< your spreadsheet ID
const SHEET_NAME = 'Leads';

// Gemini REST config (optional)
const GEMINI_API_KEY = 'REPLACE_WITH_YOUR_GEMINI_API_KEY';
const GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateText';

// Canonical field keys we expect to read/write
const KNOWN_KEYS = ['linkedin_url', 'name', 'job_title', 'company_name', 'phone', 'email', 'updated_at'];

// ---------- ENTRY POINTS ----------
function doOptions(e) {
  // Browser may attempt preflight OPTIONS. Apps Script Web App doesn't natively support CORS headers.
  // Return a lightweight JSON body. Your Chrome extension should prefer form-encoded POST (no preflight)
  // or JSONP GET (callback) when needed.
  return jsonOut({ ok: true });
}

function doGet(e) {
  try {
    const params = e.parameter || {};
    const action = (params.action || params.mode || 'ping').toLowerCase();
    const callback = params.callback || '';

    if (action === 'getbyurl') {
      const url = params.url || params.q || '';
      if (!url) return jsonOut({ error: 'missing url' }, 400, callback);
      const rec = findRowByUrl(url);
      return jsonOut({ data: rec }, 200, callback);
    }

    // Health check
    return jsonOut({ message: 'LinkedIn Sheet API running' }, 200, callback);
  } catch (err) {
    return jsonOut({ error: String(err) }, 500);
  }
}

function doPost(e) {
  try {
    const ct = (e.postData && e.postData.type) || '';
    let body = {};

    if (ct.indexOf('application/x-www-form-urlencoded') === 0) {
      // Form-encoded -> values are in e.parameter
      const p = e.parameter || {};
      if (p.data) {
        try { body = JSON.parse(p.data); } catch (e) { body = {}; }
      }
      // top-level mode/action can be passed as either body or form field
      if (!body.action && p.action) body.action = p.action;
      if (!body.mode && p.mode) body.mode = p.mode;
      if (!body.url && p.url) body.url = p.url;
      if (!body.field && p.field) body.field = p.field;
      if (!body.value && p.value) body.value = p.value;
    } else {
      // JSON
      body = e.postData && e.postData.contents ? JSON.parse(e.postData.contents) : {};
    }

    const action = (body.action || body.mode || 'unknown').toLowerCase();

    if (action === 'predicttitle') {
      const profile = body.profile || {};
      const predicted = predictTitleWithGemini(profile);
      return jsonOut({ predictedTitle: predicted });
    }

    if (action === 'updatefield') {
      const url = body.url || '';
      const field = body.field || '';
      const value = body.value !== undefined ? body.value : '';
      if (!url || !field) return jsonOut({ success: false, error: 'missing url or field' }, 400);

      const r = findRowByUrlRaw(url);
      if (!r) return jsonOut({ success: false, error: 'row not found' }, 404);

      const col = columnIndexByHeaderNameOrKey(field);
      if (!col) return jsonOut({ success: false, error: 'unknown field' }, 400);

      const sh = getSheet();
      sh.getRange(r.rowIndex, col).setValue(value);
      // Touch updated_at if exists
      const updCol = columnIndexByHeaderNameOrKey('updated_at');
      if (updCol) sh.getRange(r.rowIndex, updCol).setValue(new Date());

      return jsonOut({ success: true });
    }

    if (action === 'updatefull' || action === 'upsert') {
      const url = body.url || body.linkedin_url || '';
      const data = body.data || body.record || body.payload || body;
      if (!url && !data.linkedin_url) data.linkedin_url = url; // ensure key present
      if (!data || !(data.linkedin_url || url)) return jsonOut({ success: false, error: 'missing url or data' }, 400);

      const result = upsertRecord(data);
      return jsonOut({ success: true, result: result });
    }

    return jsonOut({ success: false, error: 'unknown action' }, 400);
  } catch (err) {
    return jsonOut({ success: false, error: String(err) }, 500);
  }
}

// ---------- RESPONSE HELPERS ----------
function jsonOut(obj, statusCode, callback) {
  statusCode = statusCode || 200;

  if (callback) {
    // JSONP response (no CORS needed for GET)
    const text = callback + '(' + JSON.stringify(obj) + ')';
    return ContentService.createTextOutput(text).setMimeType(ContentService.MimeType.JAVASCRIPT);
  }

  // Normal JSON response. Note: Apps Script Web App doesn't allow setting CORS headers here.
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

// ---------- SHEET HELPERS ----------
function getSheet() {
  const ss = SpreadsheetApp.openById(SPREADSHEET_ID);
  let sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) sh = ss.insertSheet(SHEET_NAME);
  ensureHeaders(sh);
  return sh;
}

function ensureHeaders(sh) {
  const lastCol = Math.max(1, sh.getLastColumn());
  const row = sh.getRange(1, 1, 1, lastCol).getValues()[0];
  let headers = row.map(v => String(v || '').trim());
  if (!headers.length || headers.every(h => !h)) {
    sh.getRange(1, 1, 1, KNOWN_KEYS.length).setValues([KNOWN_KEYS]);
    return;
  }
  // Append missing headers to the right
  const toAdd = KNOWN_KEYS.filter(h => headers.indexOf(h) < 0);
  if (toAdd.length) {
    sh.getRange(1, headers.length + 1, 1, toAdd.length).setValues([toAdd]);
  }
}

function getHeaders() {
  const sh = getSheet();
  const lastCol = Math.max(1, sh.getLastColumn());
  const row = sh.getRange(1, 1, 1, lastCol).getValues()[0];
  return row.map(v => String(v || '').trim());
}

function headersMap() {
  const map = {};
  getHeaders().forEach((h, i) => { if (h) map[h] = i + 1; });
  return map;
}

function columnIndexByHeaderNameOrKey(nameOrKey) {
  const name = String(nameOrKey || '').trim();
  const map = headersMap();
  // Exact header name match
  if (map[name]) return map[name];
  // Try normalized key match
  const key = normalizeHeaderToKey(name);
  const headers = getHeaders();
  for (let i = 0; i < headers.length; i++) {
    if (normalizeHeaderToKey(headers[i]) === key) return i + 1;
  }
  return null;
}

function normalizeHeaderToKey(header) {
  return header.toString().trim()
    .replace(/\s+/g, ' ')
    .replace(/[^\w\s]/g, '')
    .split(' ')
    .map((s, i) => i === 0 ? s.toLowerCase() : (s.charAt(0).toUpperCase() + s.slice(1).toLowerCase()))
    .join('');
}

// ---------- ROW LOOKUP / UPSERT ----------
function normalizeUrl(u) {
  try {
    const s = String(u || '');
    return s.split('?')[0].replace(/\/+$/, '').toLowerCase();
  } catch (e) {
    return String(u || '').toLowerCase();
  }
}

function findRowByUrlRaw(url) {
  const sh = getSheet();
  const values = sh.getDataRange().getValues();
  if (!values || values.length < 2) return null;

  // Identify LinkedIn URL column; try exact header or heuristics
  const headers = values[0].map(h => String(h || '').trim().toLowerCase());
  let urlColIndex = headers.indexOf('linkedin_url');
  if (urlColIndex < 0) {
    const candidates = ['linkedin url', 'linkedinurl', 'linkedin', 'profile url', 'url'];
    for (let i = 0; i < headers.length; i++) {
      if (candidates.indexOf(headers[i]) >= 0) { urlColIndex = i; break; }
    }
  }
  if (urlColIndex < 0) urlColIndex = headers.length - 1; // fallback to last column

  const target = normalizeUrl(url);
  for (let i = 1; i < values.length; i++) {
    const cell = String(values[i][urlColIndex] || '').trim();
    if (!cell) continue;
    if (normalizeUrl(cell) === target) {
      return { rowIndex: i + 1, rowValues: values[i] };
    }
  }
  return null;
}

function findRowByUrl(url) {
  const r = findRowByUrlRaw(url);
  if (!r) return null;
  const headers = getHeaders();
  const obj = {};
  headers.forEach((hdr, i) => {
    const key = normalizeHeaderToKey(hdr);
    obj[key] = r.rowValues[i] !== undefined ? r.rowValues[i] : '';
  });
  return obj;
}

function upsertRecord(data) {
  // data keys can be header names or normalized keys
  const sh = getSheet();
  const map = headersMap();

  const url = data.linkedin_url || data.linkedinUrl || data.url || '';
  if (!url) throw new Error('linkedin_url is required');
  const found = findRowByUrlRaw(url);

  // Construct row array with existing values when updating
  const rowLen = Math.max(sh.getLastColumn(), KNOWN_KEYS.length);
  let rowVals = new Array(rowLen).fill('');

  if (found) {
    rowVals = sh.getRange(found.rowIndex, 1, 1, rowLen).getValues()[0];
  }

  // Ensure headers include all KNOWN_KEYS so we can map deterministically
  ensureHeaders(sh);
  const headers = getHeaders();

  function setVal(name, value) {
    const col = columnIndexByHeaderNameOrKey(name);
    if (!col) return;
    rowVals[col - 1] = value;
  }

  // Always set linkedin_url
  setVal('linkedin_url', url);

  // Copy present fields from data (both header names and key aliases supported)
  const fieldMap = {
    name: ['name'],
    job_title: ['job_title', 'jobTitle'],
    company_name: ['company_name', 'companyName'],
    phone: ['phone', 'mobile', 'mobile_number', 'mobileNumber'],
    email: ['email', 'email_id', 'emailId']
  };

  Object.keys(fieldMap).forEach(headerName => {
    const keys = fieldMap[headerName];
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      if (data[k] !== undefined) {
        setVal(headerName, data[k]);
        break;
      }
    }
  });

  // Touch updated_at
  setVal('updated_at', new Date());

  if (found) {
    sh.getRange(found.rowIndex, 1, 1, rowVals.length).setValues([rowVals]);
    return { action: 'updated', row: found.rowIndex };
  } else {
    const appendRow = sh.getLastRow() + 1;
    sh.getRange(appendRow, 1, 1, rowVals.length).setValues([rowVals]);
    return { action: 'inserted', row: appendRow };
  }
}

// ---------- GEMINI INTEGRATION (optional) ----------
function predictTitleWithGemini(profile) {
  try {
    if (!GEMINI_API_KEY || GEMINI_API_KEY.indexOf('REPLACE_WITH_') === 0) return '';

    const headline = profile.headline || profile.headlineText || '';
    const experience = profile.experienceText || (Array.isArray(profile.experience)
      ? profile.experience.map(e => (e.title || '') + ' ' + (e.companyName || e.company || '') + ' ' + (e.dateRange || e.duration || '')).join('\n')
      : '');
    const name = profile.name || '';

    const prompt = 'You are an HR assistant. Given the profile data below, return the most appropriate current job title as a short phrase (2-6 words). Return only the job title and nothing else.\n\n'
      + 'Name: ' + name + '\n'
      + 'Headline: ' + headline + '\n'
      + 'Experience excerpt: ' + experience + '\n';

    // For Gemini APIs, include API key as ?key=...
    const url = GEMINI_URL + '?key=' + encodeURIComponent(GEMINI_API_KEY);
    const requestBody = {
      prompt: prompt,
      max_output_tokens: 64,
      temperature: 0.2
    };

    const options = {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify(requestBody),
      muteHttpExceptions: true
    };

    const resp = UrlFetchApp.fetch(url, options);
    const text = resp.getContentText();

    try {
      const parsed = JSON.parse(text);
      if (parsed.candidates && parsed.candidates[0]) {
        const cand = parsed.candidates[0];
        if (cand.content) {
          if (Array.isArray(cand.content)) {
            let joined = '';
            cand.content.forEach(c => {
              if (c.text) joined += c.text + ' ';
              if (c.parts) c.parts.forEach(p => { if (p.text) joined += p.text + ' '; });
            });
            if (joined.trim()) return joined.trim();
          } else if (cand.content.text) return cand.content.text.trim();
        }
        if (cand.outputText) return cand.outputText.trim();
      }
      // fallback to a short string anywhere
      for (const k in parsed) {
        if (typeof parsed[k] === 'string' && parsed[k].length < 200) return parsed[k].trim();
      }
    } catch (e) {
      const raw = text.trim();
      if (raw) return raw;
    }
    return '';
  } catch (err) {
    Logger.log('Gemini call error: ' + err);
    return '';
  }
}

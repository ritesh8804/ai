/**
 * LinkedIn Extension Sheet API (CORS-friendly for Chrome Extensions)
 * - Supports GET/POST from Chrome Extension
 * - Normalized layout: One record per platform (Zobble, Violet, WOM)
 * - Keeps LinkedIn scraping logic intact
 */

const SPREADSHEET_ID = '11ELmuu-XV1QNvdKy8dxhsAEIv7bozoNHvQscNlL_OEE';
const SHEET_NAME = 'Leads';

// New normalized headers (one row per platform)
const HEADERS = [
  'linkedin_url',
  'name',
  'job_title',
  'company_name',
  'phone',
  'email',
  'platform',       // <-- new column
  'Bounce',
  'Unsubscribed',
  'Clicked',
  'Enquired',
  'updated_at'
];

const METRIC_SOURCES = ['Zobble', 'Violet', 'WOM'];

// ---------- ENTRY POINTS ----------

function doOptions(e) {
  return jsonOut({ ok: true });
}

function doGet(e) {
  try {
    const params = e.parameter || {};
    const action = (params.action || params.mode || 'ping').toLowerCase();
    const callback = params.callback || '';

    if (action === 'getbyurl') {
      const url = params.url || params.q || params.linkedin_url || '';
      if (!url) return jsonOut({ error: 'missing url' }, 400, callback);

      const rows = findRowsByUrl(url);
      if (!rows.length) return jsonOut({ error: 'User not found' }, 404, callback);

      // Convert multiple platform rows into structured data
      const first = rows[0];
      const metrics = rows.map(r => ({
        source: r.platform,
        bounce: r.Bounce,
        unsubscribed: r.Unsubscribed,
        clicked: r.Clicked,
        enquired: r.Enquired
      }));

      const userData = {
        name: first.name,
        job_title: first.job_title,
        company_name: first.company_name,
        phone: first.phone,
        email: first.email,
        updated_at: first.updated_at,
        metrics
      };

      return jsonOut({ success: true, data: userData }, 200, callback);
    }

    return jsonOut({ message: 'LinkedIn Sheet API running (normalized layout)' }, 200, callback);
  } catch (err) {
    return jsonOut({ error: String(err) }, 500);
  }
}

function doPost(e) {
  try {
    const { body, action } = parseBody(e);

    if (action === 'updatefield') {
      const url = body.url || '';
      const field = body.field || '';
      const value = body.value !== undefined ? body.value : '';
      const platform = body.platform || ''; // optional

      if (!url || !field)
        return jsonOut({ success: false, error: 'missing url or field' }, 400);

      const sh = getSheet();
      const rows = findRowsByUrl(url);
      if (!rows.length)
        return jsonOut({ success: false, error: 'row not found' }, 404);

      // If platform is provided, target that row; otherwise update all
      for (const r of rows) {
        if (!platform || r.platform === platform) {
          const rowIndex = r.rowIndex;
          const col = columnIndexByHeaderNameOrKey(field);
          if (col) {
            sh.getRange(rowIndex, col).setValue(value);
            const updCol = columnIndexByHeaderNameOrKey('updated_at');
            if (updCol)
              sh.getRange(rowIndex, updCol).setValue(new Date());
          }
        }
      }

      return jsonOut({ success: true });
    }

    if (action === 'updatefull' || action === 'upsert') {
      const url = body.url || body.linkedin_url || '';
      const data = body.data || body.record || body.payload || body;
      if (!url && !data.linkedin_url) data.linkedin_url = url;
      if (!data || !(data.linkedin_url || url))
        return jsonOut({ success: false, error: 'missing url or data' }, 400);

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
  if (callback) {
    const text = callback + '(' + JSON.stringify(obj) + ')';
    return ContentService.createTextOutput(text)
      .setMimeType(ContentService.MimeType.JAVASCRIPT);
  }
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

function parseBody(e) {
  const ct = (e.postData && e.postData.type) || '';
  let body = {};
  if (ct.indexOf('application/x-www-form-urlencoded') === 0) {
    const p = e.parameter || {};
    if (p.data) {
      try { body = JSON.parse(p.data); } catch (err) { body = {}; }
    }
    if (!body.action && p.action) body.action = p.action;
  } else {
    body = e.postData && e.postData.contents ? JSON.parse(e.postData.contents) : {};
  }
  const action = (body.action || body.mode || 'unknown').toLowerCase();
  return { body, action };
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
  const row = sh.getRange(1, 1, 1, sh.getLastColumn() || HEADERS.length).getValues()[0];
  const headers = row.map(v => String(v || '').trim());
  if (!headers.length || headers.every(h => !h)) {
    sh.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]);
    return;
  }
  const toAdd = HEADERS.filter(h => headers.indexOf(h) < 0);
  if (toAdd.length)
    sh.getRange(1, headers.length + 1, 1, toAdd.length).setValues([toAdd]);
}

function getHeaders() {
  const sh = getSheet();
  const row = sh.getRange(1, 1, 1, sh.getLastColumn()).getValues()[0] || [];
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
  if (map[name]) return map[name];
  const headers = getHeaders();
  for (let i = 0; i < headers.length; i++) {
    if (headers[i].toLowerCase() === name.toLowerCase()) return i + 1;
  }
  return null;
}

function normalizeUrl(u) {
  try {
    const s = String(u || '');
    return s.split('?')[0].replace(/\/+$/, '').toLowerCase();
  } catch (e) {
    return String(u || '').toLowerCase();
  }
}

function findRowsByUrl(url) {
  const sh = getSheet();
  const values = sh.getDataRange().getValues();
  const headers = getHeaders();
  const urlCol = columnIndexByHeaderNameOrKey('linkedin_url') - 1;
  const out = [];

  for (let i = 1; i < values.length; i++) {
    const row = values[i];
    const cellUrl = normalizeUrl(row[urlCol] || '');
    if (cellUrl === normalizeUrl(url)) {
      const obj = {};
      headers.forEach((hdr, j) => obj[hdr] = row[j]);
      obj.rowIndex = i + 1;
      out.push(obj);
    }
  }
  return out;
}

// ---------- UPSERT LOGIC (normalized format) ----------

function upsertRecord(data) {
  const sh = getSheet();
  const url = data.linkedin_url || data.linkedinUrl || data.url || '';
  if (!url) throw new Error('linkedin_url is required');

  const existingRows = findRowsByUrl(url);
  const now = new Date();
  
  const coreFields = {
    linkedin_url: url,
    name: data.name,
    job_title: data.job_title || data.jobTitle,
    company_name: data.company_name || data.companyName,
    phone: data.phone,
    email: data.email
  };

  if (existingRows.length) {
    // Existing user - only update fields that have new non-empty values
    let hasChanges = false;
    
    existingRows.forEach(row => {
      // Only update fields that are provided and different
      Object.keys(coreFields).forEach(field => {
        if (field !== 'linkedin_url' && data[field] !== undefined && data[field] !== '') {
          const currentValue = String(row[field] || '').trim();
          const newValue = String(coreFields[field] || '').trim();
          
          if (currentValue !== newValue) {
            const col = columnIndexByHeaderNameOrKey(field);
            if (col) {
              sh.getRange(row.rowIndex, col).setValue(newValue);
              hasChanges = true;
            }
          }
        }
      });
    });
    
    // Update timestamp only if changes were made
    if (hasChanges) {
      const updCol = columnIndexByHeaderNameOrKey('updated_at');
      if (updCol) {
        existingRows.forEach(row => {
          sh.getRange(row.rowIndex, updCol).setValue(now);
        });
      }
    }
    
    return { action: hasChanges ? 'updated' : 'no_changes', rows: existingRows.length, changes: hasChanges };
  } else {
    // New user - insert all platform rows
    const campaigns = METRIC_SOURCES.map(s => ({
      source: s, bounce: '', unsubscribed: '', clicked: '', enquired: ''
    }));

    campaigns.forEach(c => {
      const row = [
        coreFields.linkedin_url,
        coreFields.name,
        coreFields.job_title,
        coreFields.company_name,
        coreFields.phone,
        coreFields.email,
        c.source,
        c.bounce,
        c.unsubscribed,
        c.clicked,
        c.enquired,
        now
      ];
      sh.appendRow(row);
    });

    return { action: 'inserted', rows: campaigns.length };
  }
}

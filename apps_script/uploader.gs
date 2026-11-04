/**
 * MS-COCOAI checkpoint uploader — Google Apps Script, zero admin / zero API keys.
 *
 * Setup (5 minutes, no Google Cloud project, no OAuth client, no service account):
 *   1. Go to https://script.google.com while logged into your university Google account
 *      and click "New project". Delete the sample code and paste this whole file.
 *   2. Set SECRET to a long random string (e.g. `openssl rand -hex 24`).
 *      Keep PARENT_FOLDER_NAME as is or change it to any Drive folder name.
 *   3. Click "Deploy" -> "New deployment" -> gear icon -> type "Web app".
 *        - Description: mscocoai uploader
 *        - Execute as:  Me (your university account)
 *        - Who has access: "Anyone" (or "Anyone with Google account" if your
 *          Workspace forbids "Anyone"; the client works the same).
 *   4. Click "Deploy", authorize the requested Drive scopes (you will see a
 *      warning that the app isn't verified — this is your own script; allow it),
 *      and copy the "Web app URL".
 *   5. On Kaggle / lab PC export:
 *        export MSCOCOAI_APPS_SCRIPT_URL="https://script.google.com/macros/s/AKfy.../exec"
 *        export MSCOCOAI_APPS_SCRIPT_TOKEN="<the SECRET string>"
 *      then:  python -m mscocoai.drive_upload --action upload --file outputs/run/last_weights.pt --run myrun
 *      Checkpoint parts appear under Drive: My Drive/MscocoAI-Weights/myrun/
 *   6. On a fresh Kaggle session restore them:
 *        python -m mscocoai.drive_upload --action download-latest --run myrun
 *
 * The script only touches the fold it creates ("MscocoAI-Weights/<run>/").
 * Delete the deployment anytime via script.google.com -> Deploy -> Manage deployments.
 */

const SECRET = 'PASTE_A_LONG_RANDOM_STRING_HERE';      // must match MSCOCOAI_APPS_SCRIPT_TOKEN
const PARENT_FOLDER_NAME = 'MscocoAI-Weights';         // created automatically in My Drive
const MAX_PARTS = 512;                                  // sanity limit (512 * 24MB = 12 GB)

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents);
    if (!body.token || body.token !== SECRET) return _out({ error: 'unauthorized' });

    const parent = _parentFolder();
    const run = String(body.run || 'default').replace(/[\\/]/g, '_').substring(0, 128);
    const folder = _child(parent, run);

    switch (body.action) {
      case 'purge': {
        _purgeName(folder, String(body.name));
        return _out({ ok: true, purged: body.name });
      }
      case 'store': {
        const idx = parseInt(body.index, 10);
        const total = parseInt(body.total, 10);
        if (isNaN(idx) || isNaN(total) || idx < 0 || total < 1 || total > MAX_PARTS || idx >= total) {
          return _out({ error: 'bad index/total' });
        }
        const bytes = Utilities.base64Decode(body.data);
        const part = _partName(String(body.name), idx);
        _replaceBlob(folder, part, Utilities.newBlob(bytes, 'application/octet-stream', part));
        return _out({ ok: true, stored: part, index: idx, total: total });
      }
      case 'finalize': {
        const total = parseInt(body.total, 10);
        const name = String(body.name);
        if (isNaN(total) || total < 1 || total > MAX_PARTS) return _out({ error: 'bad total' });
        const parts = [];
        let size = 0;
        const missing = [];
        for (let i = 0; i < total; i++) {
          const p = _find(folder, _partName(name, i));
          if (!p) { missing.push(_partName(name, i)); } else { parts.push(p.getName()); size += p.getSize(); }
        }
        if (missing.length) return _out({ error: 'missing parts', missing: missing });
        const manifest = {
          name: name, run: run, parts: total, partNames: parts, size: size,
          updated: new Date().toISOString(),
        };
        _replaceBlob(folder, name + '.manifest.json',
                     Utilities.newBlob(JSON.stringify(manifest, null, 2), 'application/json', name + '.manifest.json'));
        return _out({ ok: true, name: name, parts: total, size: size });
      }
      case 'list': {
        const files = folder.getFiles();
        const groups = {};
        const plain = [];
        while (files.hasNext()) {
          const f = files.next();
          const n = f.getName();
          const m = n.match(/^(.*)\.part-(\d{4})$/);
          if (m) {
            const base = m[1];
            groups[base] = groups[base] || { size: 0, count: 0 };
            groups[base].size += f.getSize();
            groups[base].count += 1;
            continue;
          }
          if (n.endsWith('.manifest.json')) continue;
          plain.push({ name: n, size: f.getSize(), parts: 1 });
        }
        for (const base of Object.keys(groups)) {
          plain.push({ name: base, size: groups[base].size, parts: groups[base].count });
        }
        return _out({ files: plain });
      }
      case 'part': {
        const idx = parseInt(body.index, 10);
        const name = String(body.name);
        if (isNaN(idx) || idx < 0) return _out({ error: 'bad index' });
        const f = _find(folder, _partName(name, idx));
        if (!f) return _out({ error: 'part not found: ' + _partName(name, idx) });
        return _out({ ok: true, data: Utilities.base64Encode(f.getBlob().getBytes()) });
      }
      case 'delete': {
        _purgeName(folder, String(body.name));
        return _out({ ok: true });
      }
      default:
        return _out({ error: 'unknown action' });
    }
  } catch (err) {
    return _out({ error: String(err) });
  }
}

function _parentFolder() {
  const it = DriveApp.getFoldersByName(PARENT_FOLDER_NAME);
  return it.hasNext() ? it.next() : DriveApp.createFolder(PARENT_FOLDER_NAME);
}
function _child(parent, name) {
  const it = parent.getFoldersByName(name);
  return it.hasNext() ? it.next() : parent.createFolder(name);
}
function _find(folder, name) {
  const it = folder.getFilesByName(name);
  return it.hasNext() ? it.next() : null;
}
function _partName(name, i) {
  return name + '.part-' + String(i).padStart(4, '0');
}
function _replaceBlob(folder, name, blob) {
  const old = _find(folder, name);
  if (old) old.setTrashed(true);
  folder.createFile(blob);
}
function _purgeName(folder, name) {
  const prefixes = [name + '.part-', name + '.manifest.json', name + '.'];
  const it = folder.getFiles();
  const doomed = [];
  while (it.hasNext()) {
    const f = it.next();
    const n = f.getName();
    if (n === name || n.startsWith(name + '.')) doomed.push(f);
  }
  doomed.forEach(function (f) { f.setTrashed(true); });
}
function _out(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

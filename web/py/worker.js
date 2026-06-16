/* Pyodide worker: runs the pabench pipeline in WebAssembly so the static site can launch
 * evaluations and compute custom metrics with no backend. It loads the pabench sources emitted
 * by build_web_data.py (web/py/) and dispatches /api/* requests to pabench.browser_api.
 *
 * Protocol (postMessage):
 *   main → worker: {type:'init', baseUrl, files, legacyReport, legacyIndex, customSpecs}
 *                  {type:'request', id, method, path, query, body}
 *   worker → main: {type:'ready'} | {type:'error', message}
 *                  {type:'response', id, status, body}
 */
const PYODIDE_VERSION = "v0.27.2";
importScripts(`https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/pyodide.js`);

let pyodide = null;
let dispatch = null;  // python _dispatch(method, path, query_json, body_json) -> "[status, obj]" JSON

async function init(msg) {
  pyodide = await loadPyodide({
    indexURL: `https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/`,
  });
  await pyodide.loadPackage(["numpy"]);

  // write the pabench sources into the in-memory FS, then make them importable
  for (const rel of msg.files) {
    const text = await (await fetch(msg.baseUrl + "py/" + rel)).text();
    const path = "/lib/" + rel;
    const dir = path.slice(0, path.lastIndexOf("/"));
    pyodide.FS.mkdirTree(dir);
    pyodide.FS.writeFile(path, text);
  }

  // seed the in-browser backend: import, build, load the static demo run + saved custom metrics
  pyodide.globals.set("_legacy_report", JSON.stringify(msg.legacyReport || null));
  pyodide.globals.set("_legacy_index", JSON.stringify(msg.legacyIndex || null));
  pyodide.globals.set("_custom_specs", JSON.stringify(msg.customSpecs || []));
  pyodide.runPython(`
import sys, json as _json
sys.path.insert(0, "/lib")
import pabench.browser_api as _ba
_backend = _ba.BrowserBackend()
_rep = _json.loads(_legacy_report)
_idx = _json.loads(_legacy_index)
if _rep is not None and _idx is not None:
    _backend.load_legacy(_rep, _idx)
_backend.seed_custom(_json.loads(_custom_specs))

def _dispatch(method, path, query_json, body_json):
    q = _json.loads(query_json) if query_json else {}
    b = _json.loads(body_json) if body_json else None
    status, obj = _backend.handle(method, path, q, b)
    return _json.dumps([status, obj])
`);
  dispatch = pyodide.globals.get("_dispatch");
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    try { await init(msg); self.postMessage({ type: "ready" }); }
    catch (err) { self.postMessage({ type: "error", message: String(err && err.message || err) }); }
    return;
  }
  if (msg.type === "request") {
    try {
      const out = dispatch(msg.method, msg.path,
        msg.query ? JSON.stringify(msg.query) : "",
        msg.body ? JSON.stringify(msg.body) : "");
      const [status, obj] = JSON.parse(out);
      self.postMessage({ type: "response", id: msg.id, status, body: obj });
    } catch (err) {
      self.postMessage({ type: "response", id: msg.id, status: 500,
        body: { detail: String(err && err.message || err) } });
    }
  }
};
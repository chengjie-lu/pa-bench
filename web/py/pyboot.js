/* Boots the Pyodide worker and installs a fetch interceptor so the existing server-mode UI
 * (run wizard, custom-metric registration, results, episode browser) works against an in-browser
 * Python backend on the static GitHub Pages deploy — no server required.
 *
 * Exposes bootPyodideBackend(): Promise<boolean>. On success, all /api/* fetches are routed to the
 * worker; on failure the app stays in plain static (view-only) mode.
 */
const BASE = new URL("../", import.meta.url).href;     // site root, e.g. https://…/pa-bench/
const CUSTOM_KEY = "pabench-custom-metrics";

let worker = null;
let bootPromise = null;
let reqSeq = 0;
const pending = new Map();

function workerRequest(method, path, query, body) {
  return new Promise((resolve, reject) => {
    const id = ++reqSeq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: "request", id, method, path, query, body });
  });
}

async function loadJSON(url) {
  try { const r = await _origFetch(url); return r.ok ? await r.json() : null; }
  catch { return null; }
}

const _origFetch = self.fetch.bind(self);

function jsonResponse(status, obj) {
  return new Response(JSON.stringify(obj), {
    status, headers: { "Content-Type": "application/json" } });
}

function urlOf(input) {
  return typeof input === "string" ? input : (input && input.url) || "";
}

function pathAndQuery(rawUrl) {
  const u = new URL(rawUrl, location.href);
  const query = Object.fromEntries(u.searchParams.entries());
  return { path: u.pathname.replace(/^.*?(\/api\/)/, "/api/"), query };
}

function installFetchInterceptor() {
  self.fetch = async (input, opts = {}) => {
    const raw = urlOf(input);
    if (!/\/api\//.test(raw)) return _origFetch(input, opts);
    const { path, query } = pathAndQuery(raw);
    const method = (opts.method || (typeof input !== "string" && input.method) || "GET").toUpperCase();
    let body = null;
    if (opts.body) { try { body = JSON.parse(opts.body); } catch { body = null; } }

    let { status, body: out } = await workerRequest(method, path, query, body);

    // legacy demo episodes live as static files, not in worker memory → fall back on 404
    if (status === 404 && /^\/api\/episodes\/[^/]+$/.test(path)) {
      const id = decodeURIComponent(path.split("/").pop());
      const r = await _origFetch(`${BASE}data/episodes/${encodeURIComponent(id)}.json`);
      if (r.ok) return r;
    }
    // persist custom-metric changes so they survive a page reload
    if (path === "/api/custom-metrics" && method === "POST" && status === 201) await saveCustom();
    if (/^\/api\/custom-metrics\//.test(path) && method === "DELETE" && status === 200) await saveCustom();

    return jsonResponse(status, out);
  };
}

async function saveCustom() {
  const r = await workerRequest("GET", "/api/custom-metrics", {}, null);
  if (r.status === 200) localStorage.setItem(CUSTOM_KEY, JSON.stringify(r.body.metrics || []));
}

function loadSavedCustom() {
  try { return JSON.parse(localStorage.getItem(CUSTOM_KEY) || "[]"); } catch { return []; }
}

export function bootPyodideBackend() {
  if (bootPromise) return bootPromise;
  bootPromise = (async () => {
    if (typeof Worker === "undefined" || typeof WebAssembly === "undefined") return false;
    const [manifest, legacyReport, legacyIndex] = await Promise.all([
      loadJSON(`${BASE}py/manifest.json`),
      loadJSON(`${BASE}data/report.json`),
      loadJSON(`${BASE}data/index.json`),
    ]);
    if (!manifest || !manifest.files) return false;

    worker = new Worker(new URL("./worker.js", import.meta.url));
    const ready = new Promise((resolve, reject) => {
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "ready") return resolve(true);
        if (m.type === "error") return reject(new Error(m.message));
        if (m.type === "response") {
          const p = pending.get(m.id);
          if (p) { pending.delete(m.id); p.resolve({ status: m.status, body: m.body }); }
        }
      };
      worker.onerror = (e) => reject(new Error(e.message || "worker error"));
    });
    worker.postMessage({
      type: "init", baseUrl: BASE, files: manifest.files,
      legacyReport, legacyIndex, customSpecs: loadSavedCustom(),
    });
    await ready;
    installFetchInterceptor();
    return true;
  })().catch(() => false);
  return bootPromise;
}
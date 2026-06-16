"""User-defined metrics (FR-5.1 extension): a safe formula over per-episode fields.

A custom metric is a per-episode arithmetic/boolean expression plus an aggregation across a
combo's episodes. Example specs:
  - expr "plan_margin_ratio", agg "mean"      → mean e_plan margin
  - expr "e_track_steady_rms_mm > 0.5", agg "rate" → fraction of episodes over a tracking budget
  - expr "1 - success", agg "mean"            → failure rate

Expressions are evaluated with a hardened ast walker — NO Python eval(): only numeric/boolean
arithmetic over a whitelist of field names is allowed, so a user-submitted formula can never
import, call arbitrary functions, or touch attributes. This keeps "users register metrics
themselves" safe even though the formula runs on the backend.

R-8 still applies: every metric (built-in or custom) must bind >=1 improvement action.
"""
from __future__ import annotations

import ast
import json
import math
import statistics
from pathlib import Path

# Whitelisted per-episode fields (subset of webdata.index_record). bool success → 1.0/0.0.
# Each maps the record key to a coercion; None values make a record skip the metric.
METRIC_FIELDS = {
    "success": "1.0 when the episode succeeded else 0.0",
    "plan_margin_ratio": "e_plan lateral offset / tolerance (>1 = doomed)",
    "e_track_steady_rms_mm": "steady-state tracking error RMS [mm] (hardware)",
    "peak_uncertainty": "peak model uncertainty (None when the model emits none)",
    "lux": "lighting factor of the scene",
    "duration_s": "episode duration [s]",
}
AGGS = ("mean", "median", "max", "min", "sum", "rate")
OWNERS = ("model", "hardware", "both", "environment")
LEVELS = ("L0", "L1", "L2", "L3")

# Functions allowed inside a formula (called as abs(x), min(a,b), max(a,b)).
_SAFE_FUNCS = {"abs": abs, "min": min, "max": max}
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_UNARYOPS = (ast.UAdd, ast.USub, ast.Not)


class MetricSpecError(ValueError):
    """Raised when a custom-metric spec is malformed or its formula is unsafe."""


def _record_value(rec: dict, name: str):
    v = rec.get(name)
    if name == "success":
        return 1.0 if v else 0.0
    return None if v is None else float(v)


def _check_node(node: ast.AST) -> None:
    """Recursively assert that the AST uses only the safe arithmetic/boolean subset."""
    if isinstance(node, ast.Expression):
        return _check_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or isinstance(node.value, (int, float)):
            return
        raise MetricSpecError(f"only numeric constants are allowed, got {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in METRIC_FIELDS or node.id in _SAFE_FUNCS:
            return
        raise MetricSpecError(
            f"unknown name '{node.id}'. Allowed fields: {sorted(METRIC_FIELDS)}; "
            f"functions: {sorted(_SAFE_FUNCS)}")
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BINOPS):
        _check_node(node.left); _check_node(node.right); return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARYOPS):
        return _check_node(node.operand)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        for v in node.values:
            _check_node(v)
        return
    if isinstance(node, ast.Compare):
        if not all(isinstance(op, _CMPOPS) for op in node.ops):
            raise MetricSpecError("unsupported comparison operator")
        _check_node(node.left)
        for c in node.comparators:
            _check_node(c)
        return
    if isinstance(node, ast.IfExp):  # ternary: a if cond else b
        _check_node(node.test); _check_node(node.body); _check_node(node.orelse); return
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS
                and not node.keywords):
            raise MetricSpecError("only abs()/min()/max() calls are allowed")
        for a in node.args:
            _check_node(a)
        return
    raise MetricSpecError(f"disallowed expression element: {type(node).__name__}")


def compile_expr(expr: str) -> ast.Expression:
    """Parse + safety-check a formula. Returns the AST; raises MetricSpecError if unsafe."""
    if not expr or not expr.strip():
        raise MetricSpecError("expression is empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise MetricSpecError(f"syntax error: {e.msg}")
    _check_node(tree)
    return tree


def _names_in(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id in METRIC_FIELDS}


def _eval(node: ast.AST, env: dict):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id] if node.id in env else _SAFE_FUNCS[node.id]
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env)
        return +v if isinstance(node.op, ast.UAdd) else -v if isinstance(node.op, ast.USub) else (not v)
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        op = node.op
        if isinstance(op, ast.Add): return a + b
        if isinstance(op, ast.Sub): return a - b
        if isinstance(op, ast.Mult): return a * b
        if isinstance(op, ast.Div): return a / b
        if isinstance(op, ast.FloorDiv): return a // b
        if isinstance(op, ast.Mod): return a % b
        if isinstance(op, ast.Pow): return a ** b
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            ok = (left < right if isinstance(op, ast.Lt) else
                  left <= right if isinstance(op, ast.LtE) else
                  left > right if isinstance(op, ast.Gt) else
                  left >= right if isinstance(op, ast.GtE) else
                  left == right if isinstance(op, ast.Eq) else left != right)
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _eval(node.body, env) if _eval(node.test, env) else _eval(node.orelse, env)
    if isinstance(node, ast.Call):
        return _SAFE_FUNCS[node.func.id](*[_eval(a, env) for a in node.args])
    raise MetricSpecError(f"disallowed expression element: {type(node).__name__}")  # pragma: no cover


def evaluate(spec: dict, records: list[dict]) -> float | None:
    """Evaluate one custom metric over a list of index records. None if no usable records."""
    tree = spec["_ast"] if "_ast" in spec else compile_expr(spec["expr"])
    needed = _names_in(tree)
    vals = []
    for rec in records:
        env = {n: _record_value(rec, n) for n in needed}
        if any(v is None for v in env.values()):  # a referenced field is missing on this record
            continue
        vals.append(float(_eval(tree, env)))
    if not vals:
        return None
    agg = spec["agg"]
    if agg == "mean": return statistics.mean(vals)
    if agg == "median": return statistics.median(vals)
    if agg == "max": return max(vals)
    if agg == "min": return min(vals)
    if agg == "sum": return math.fsum(vals)
    if agg == "rate": return sum(1 for v in vals if v != 0) / len(vals)
    raise MetricSpecError(f"unknown aggregation: {agg}")  # pragma: no cover


def validate_spec(spec: dict) -> dict:
    """Validate + normalize a user-submitted spec. Returns the clean spec; raises MetricSpecError."""
    mid = str(spec.get("metric_id", "")).strip()
    if not mid:
        raise MetricSpecError("metric_id is required")
    if not all(c.isalnum() or c in "._-" for c in mid):
        raise MetricSpecError("metric_id may contain only letters, digits, and . _ -")
    level = str(spec.get("level", "L2"))
    if level not in LEVELS:
        raise MetricSpecError(f"level must be one of {LEVELS}")
    owner = str(spec.get("owner", "model"))
    if owner not in OWNERS:
        raise MetricSpecError(f"owner must be one of {OWNERS}")
    definition = str(spec.get("definition", "")).strip()
    if not definition:
        raise MetricSpecError("definition is required")
    agg = str(spec.get("agg", "mean"))
    if agg not in AGGS:
        raise MetricSpecError(f"agg must be one of {AGGS}")
    actions = spec.get("improvement_actions") or []
    if isinstance(actions, str):
        actions = [a.strip() for a in actions.splitlines() if a.strip()]
    actions = [str(a).strip() for a in actions if str(a).strip()]
    if not actions:  # R-8
        raise MetricSpecError("at least one improvement action is required (R-8)")
    compile_expr(str(spec.get("expr", "")))  # raises if unsafe/invalid
    return {"metric_id": mid, "level": level, "owner": owner, "definition": definition,
            "expr": str(spec["expr"]).strip(), "agg": agg, "improvement_actions": actions,
            "custom": True}


def compute_for_combos(specs: list[dict], records: list[dict]) -> dict[str, dict[str, float]]:
    """Group records by '<model_id> @ <hw_config_id>' and evaluate each spec per combo."""
    if not specs:
        return {}
    combos: dict[str, list[dict]] = {}
    for r in records:
        combos.setdefault(f"{r['model_id']} @ {r['hw_config_id']}", []).append(r)
    compiled = [dict(s, _ast=compile_expr(s["expr"])) for s in specs]
    out: dict[str, dict[str, float]] = {}
    for combo, recs in combos.items():
        out[combo] = {s["metric_id"]: evaluate(s, recs) for s in compiled}
    return out


class CustomMetricStore:
    """JSON-file persistence for user-registered metrics (one file, list of specs)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._specs: list[dict] = []
        if self.path.exists():
            try:
                self._specs = json.loads(self.path.read_text())
            except (OSError, ValueError):
                self._specs = []

    def list(self) -> list[dict]:
        return list(self._specs)

    def add(self, spec: dict) -> dict:
        clean = validate_spec(spec)
        if any(s["metric_id"] == clean["metric_id"] for s in self._specs):
            raise MetricSpecError(f"metric_id already exists: {clean['metric_id']}")
        self._specs.append(clean)
        self._save()
        return clean

    def delete(self, metric_id: str) -> bool:
        before = len(self._specs)
        self._specs = [s for s in self._specs if s["metric_id"] != metric_id]
        if len(self._specs) == before:
            return False
        self._save()
        return True

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._specs, ensure_ascii=False, indent=1))
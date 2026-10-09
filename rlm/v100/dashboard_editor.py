"""Validated fixed-path guest layout edits and host publication evidence."""

import base64
import json
import re
import time
from pathlib import Path

from rlm.v100.dashboard_layout import BASE_TEMPLATE, guest_layout, validate_template


def status(root: Path, include_html: bool = False) -> dict:
    try:
        content = guest_layout(root)
    except (ValueError, OSError, RuntimeError) as error:
        return {"valid": False, "published": False, "error": str(error)[:300]}
    try:
        identity = validate_template(content)
    except ValueError as error:
        result = {
            "valid": False,
            "published": False,
            "error": str(error)[:300],
            "repair_tool": "repair_dashboard",
        }
        if include_html:
            result["html"] = content
            result["scope"] = "Untrusted source text for editing only; not executable or published"
        return result
    path = root / "research/dashboard/layout-status.json"
    receipt = json.loads(path.read_text()) if path.exists() and path.stat().st_size < 8192 else {}
    published = (
        receipt.get("state") == "validated layout active"
        and receipt.get("active_sha256") == identity
    )
    result = {
        "valid": True,
        "published": published,
        "guest_sha256": identity,
        "host": receipt,
        "scope": "Published only when the host reports this exact validated layout",
    }
    if include_html:
        result["html"] = content
    return result


def write(root: Path, content: str) -> dict:
    from rlm.v100.desktop import run

    identity = validate_template(content)
    encoded = base64.b64encode(content.encode()).decode()
    script = (
        "python3 - <<'DASHBOARD_EDIT'\nimport base64,os\nfrom pathlib import Path\n"
        "p=Path('/workspace/dashboard/index.html')\np.parent.mkdir(parents=True,exist_ok=True)\n"
        f"backup=p.with_name('index.before-{time.time_ns()}.html')\n"
        "if p.exists(): backup.write_bytes(p.read_bytes())\n"
        f"raw=base64.b64decode({encoded!r})\n"
        "temporary=p.with_name('index.next.html')\n"
        "with temporary.open('wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())\n"
        "temporary.replace(p)\nprint('VALIDATED_GUEST_LAYOUT_WRITTEN')\nDASHBOARD_EDIT\n"
    )
    if len(script) > 16000:
        raise ValueError("Dashboard edit exceeds the guest command budget; use smaller HTML/CSS")
    result = run(root, script, seconds=15, output_limit=2000)
    if result["exit_code"]:
        raise RuntimeError(result.get("stderr") or "Guest layout write failed")
    return {
        "written": True,
        "guest_sha256": identity,
        "published": False,
        "scope": "Validated and saved in guest; dashboard_status confirms later publication",
    }


def repair(root: Path) -> dict:
    """Repair invalid passive HTML and preserve the original in a guest backup."""
    content = guest_layout(root)
    try:
        validate_template(content)
        return {"repaired": False, "reason": "Existing layout valid"}
    except ValueError as error:
        reason = str(error)[:300]
    cleaned = re.sub(r"<script\b[^>]*>.*?</script\s*>", "", content, flags=re.I | re.S)
    try:
        validate_template(cleaned)
        mode = "scripts removed; passive layout preserved"
    except ValueError:
        cleaned = BASE_TEMPLATE
        mode = "default valid layout restored; original backed up"
    return {**write(root, cleaned), "repaired": True, "reason": reason, "mode": mode}

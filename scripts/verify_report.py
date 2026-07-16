from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path


def main() -> int:
    html_path = Path(sys.argv[1])
    tab = sys.argv[2] if len(sys.argv) > 2 else "overview"
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
    scroll_y = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    selector = sys.argv[5] if len(sys.argv) > 5 else ""
    target = json.dumps(selector)
    drive = (
        f"document.querySelector('[data-tab=\"{tab}\"]').click();"
        f"if ({target}) {{const t=document.querySelector({target});"
        "t.style.position='fixed';t.style.inset='0';t.style.width='100vw';t.style.height='100vh';"
        "t.style.zIndex='9999';t.style.background='white';Plotly.Plots.resize(t);}}"
        f"else window.scrollTo(0,{scroll_y});"
    )
    inject = f"""<script>
window.__errs=[];
window.onerror=(m,s,l,c,e)=>window.__errs.push(String(m));
window.addEventListener('unhandledrejection',e=>window.__errs.push('rej:'+e.reason));
window.addEventListener('load',()=>setTimeout(()=>{{{drive}
document.title='ERRJSON:'+encodeURIComponent(JSON.stringify(window.__errs));}},2200));
</script>"""
    probe = Path("/tmp/tracking2-report-probe.html")
    probe.write_text(html_path.read_text().replace("</body>", inject + "</body>"))
    screenshot = Path(f"/tmp/tracking2-{tab}-{width}-{scroll_y}.png")
    chrome = [
        "/home/crw/.local/bin/chromium", "--headless", "--no-sandbox", "--disable-gpu",
        f"--window-size={width},1400", "--virtual-time-budget=9000",
    ]
    subprocess.run(chrome + [f"--screenshot={screenshot}", str(probe)], check=True, capture_output=True)
    dom = subprocess.run(chrome + ["--dump-dom", str(probe)], check=True, capture_output=True, text=True).stdout
    match = re.search(r"<title>ERRJSON:([^<]*)</title>", dom)
    errors = json.loads(urllib.parse.unquote(match.group(1))) if match else ["page did not complete"]
    print(json.dumps({"tab": tab, "width": width, "scroll_y": scroll_y, "errors": errors, "screenshot": str(screenshot)}))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())

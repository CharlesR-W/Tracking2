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
    initial_scroll = "" if selector else f"window.scrollTo(0,{scroll_y});"
    drive = (
        f"document.querySelector('[data-tab=\"{tab}\"]').click();"
        f"if ({target}) {{const t=document.querySelector({target});"
        "t.style.position='fixed';t.style.inset='0';t.style.width='100vw';t.style.height='100vh';"
        "t.style.zIndex='9999';t.style.background='white';Plotly.Plots.resize(t);}"
        f"else window.scrollTo(0,{scroll_y});"
    )
    inject = f"""<script>
{initial_scroll}
window.__errs=[];
window.onerror=(m,s,l,c,e)=>window.__errs.push(String(m));
window.addEventListener('unhandledrejection',e=>window.__errs.push('rej:'+e.reason));
window.__finalizeVerify=()=>{{document.documentElement.dataset.verifyMetrics=encodeURIComponent(JSON.stringify({{
  scrollY:window.scrollY,scrollHeight:document.documentElement.scrollHeight,
  activePanel:document.querySelector('.panel.active')?.id||null,
  activeHeight:document.querySelector('.panel.active')?.getBoundingClientRect().height||null,
  parts:[...document.querySelectorAll('.pd-panel')].map(x=>({{id:x.id,top:x.getBoundingClientRect().top,height:x.getBoundingClientRect().height}}))
}}));document.title='ERRJSON:'+encodeURIComponent(JSON.stringify(window.__errs));}};
window.addEventListener('load',()=>setTimeout(()=>{{try{{{drive}}}
catch(e){{window.__errs.push('drive:'+String(e));}}
finally{{setTimeout(window.__finalizeVerify,800);}}}},2200));
</script>"""
    probe = Path("/tmp/tracking2-report-probe.html")
    source = html_path.read_text()
    if tab != "overview":
        source = source.replace('data-tab="overview" class="active"', 'data-tab="overview"')
        source = source.replace(f'data-tab="{tab}"', f'data-tab="{tab}" class="active"', 1)
        source = source.replace('<section id="overview" class="panel active">',
                                '<section id="overview" class="panel">')
        source = source.replace(f'<section id="{tab}" class="panel',
                                f'<section id="{tab}" class="panel active', 1)
    if selector.startswith("#") and re.fullmatch(r"#[A-Za-z][\w-]*", selector):
        element_id = selector[1:]
        source = source.replace(
            f'id="{element_id}"',
            f'id="{element_id}" style="position:fixed;inset:0;width:100vw;height:100vh;'
            'z-index:9999;background:white;overflow:auto"',
            1,
        )
    probe.write_text(source.replace("</body>", inject + "</body>"))
    screenshot = Path(f"/tmp/tracking2-{tab}-{width}-{scroll_y}.png")
    chrome = [
        "/home/crw/.local/bin/chromium", "--headless", "--no-sandbox", "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--window-size={width},1400", "--timeout=5000", "--virtual-time-budget=30000",
    ]
    subprocess.run(chrome + [f"--screenshot={screenshot}", str(probe)], check=True, capture_output=True)
    dom = subprocess.run(chrome + ["--dump-dom", str(probe)], check=True, capture_output=True, text=True).stdout
    match = re.search(r"<title>ERRJSON:([^<]*)</title>", dom)
    errors = json.loads(urllib.parse.unquote(match.group(1))) if match else ["page did not complete"]
    metrics_match = re.search(r'data-verify-metrics="([^"]*)"', dom)
    metrics = json.loads(urllib.parse.unquote(metrics_match.group(1))) if metrics_match else {}
    print(json.dumps({"tab": tab, "width": width, "scroll_y": scroll_y, "errors": errors,
                      "metrics": metrics, "screenshot": str(screenshot)}))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())

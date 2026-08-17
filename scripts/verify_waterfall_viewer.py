#!/usr/bin/env python3
"""Headlessly verify and screenshot the self-contained waterfall viewer."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import urllib.parse
from pathlib import Path


def instrument(source: Path, destination: Path) -> None:
    injection = r"""
<script>
window.__waterfallErrors=[];
window.onerror=(m,s,l,c,e)=>window.__waterfallErrors.push(String(m));
window.addEventListener('unhandledrejection',e=>window.__waterfallErrors.push('rejection:'+e.reason));
window.addEventListener('load',()=>setTimeout(()=>{
  const plots=[...document.querySelectorAll('.js-plotly-plot')];
  if(plots.length!==3) window.__waterfallErrors.push('plot-count:'+plots.length);
  plots.forEach((plot,index)=>{
    if(plot.getBoundingClientRect().width<280) window.__waterfallErrors.push('plot-width-'+index+':'+plot.getBoundingClientRect().width);
  });
  const state={errors:window.__waterfallErrors,plots:plots.length,widths:plots.map(p=>Math.round(p.getBoundingClientRect().width))};
  document.title='WATERFALLJSON:'+encodeURIComponent(JSON.stringify(state));
},2400));
</script>
"""
    destination.write_text(source.read_text().replace("</body>", injection + "\n</body>"))


def run_probe(chromium: str, probe: Path, screenshot: Path, size: str) -> dict:
    command = [
        chromium,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        f"--window-size={size}",
        "--virtual-time-budget=10000",
        f"--screenshot={screenshot}",
        str(probe),
    ]
    subprocess.run(command, check=True, capture_output=True)
    dom = subprocess.run(
        command[:-2] + ["--dump-dom", str(probe)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"<title>WATERFALLJSON:([^<]*)</title>", dom)
    if not match:
        raise RuntimeError("Viewer did not finish its JavaScript verification probe.")
    return json.loads(urllib.parse.unquote(match.group(1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/waterfall-viewer"))
    args = parser.parse_args()
    chromium = shutil.which("chromium") or "/home/crw/.local/bin/chromium"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="waterfall-probe-") as temp_dir:
        probe = Path(temp_dir) / "probe.html"
        instrument(args.html.resolve(), probe)
        reports = {
            "desktop": run_probe(
                chromium,
                probe,
                args.output_dir / "waterfalls-desktop.png",
                "1280,1600",
            ),
            "mobile": run_probe(
                chromium,
                probe,
                args.output_dir / "waterfalls-mobile.png",
                "430,1600",
            ),
        }
    print(json.dumps(reports, indent=2))
    errors = [error for report in reports.values() for error in report["errors"]]
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

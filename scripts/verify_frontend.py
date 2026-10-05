import json
import re
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright


sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parents[1]
(root / "diagnostics").mkdir(exist_ok=True)
workflow = json.loads((root / "workflows/qwen21_watermark.json").read_text(encoding="utf-8"))
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", headless=True, args=["--disable-gpu"])
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto("http://127.0.0.1:8188", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_function("() => !!window.comfyAPI?.app?.app?.canvas", timeout=60000)
    page.wait_for_function("() => {const app=window.comfyAPI?.app?.app; const types=app?.graph?.constructor?.registered_node_types || window.LiteGraph?.registered_node_types || {}; return ['QWMDetect','QWMRefineMask','QWMPrepare','QWMComposite'].every(name => !!types[name]);}", timeout=30000)
    page.get_by_role("button", name=re.compile("运行|Run")).first.wait_for(state="visible", timeout=60000)
    page.wait_for_function("() => window.comfyAPI?.app?.app?.rootGraph?._nodes?.length > 0", timeout=30000)
    result = page.evaluate("""async (workflow) => {
        const {app}=await import('/scripts/app.js');
        await app.loadGraphData(workflow);
        const prompt=await app.graphToPrompt();
        app.canvas.ds.scale=0.48;
        app.canvas.ds.offset=[10,100];
        app.canvas.setDirty(true,true);
        return {nodes:app.rootGraph._nodes.map(n=>({id:n.id,type:n.type})),output:prompt.output};
    }""", workflow)
    (root / "diagnostics/frontend_roundtrip.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    expected = json.loads((root / "workflows/qwen21_watermark.api.json").read_text(encoding="utf-8"))
    differences = []
    for key, node in expected.items():
        got = result["output"].get(key)
        if not got:
            differences.append((key, "missing node"))
            continue
        for name, value in node["inputs"].items():
            if got["inputs"].get(name) != value:
                differences.append((key, name, value, got["inputs"].get(name)))
    print("Loaded nodes:", len(result["nodes"]))
    print("Roundtrip differences:", json.dumps(differences, ensure_ascii=False))
    page.screenshot(path=str(root / "diagnostics/workflow_frontend.png"), full_page=True)
    browser.close()
    if differences:
        raise SystemExit(1)

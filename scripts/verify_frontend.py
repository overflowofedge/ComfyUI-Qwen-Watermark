import argparse
import json
import re
from pathlib import Path
import sys
import uuid

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]


def graph_differences(expected, actual):
    differences = []
    for key, node in expected.items():
        got = actual.get(key)
        if not got:
            differences.append((key, "missing node"))
            continue
        for name, value in node["inputs"].items():
            if got["inputs"].get(name) != value:
                differences.append((key, name, value, got["inputs"].get(name)))
    return differences


def main():
    parser = argparse.ArgumentParser(description="验证前端导入、参数回导和运行按钮；拦截提交，不启动推理")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    diagnostics = ROOT / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", headless=True, args=["--disable-gpu"])
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        submissions, errors = [], []

        def intercept_prompt(route):
            if route.request.method != "POST":
                route.continue_()
                return
            submissions.append(route.request.post_data_json)
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "prompt_id": str(uuid.uuid4()), "number": 0, "node_errors": {},
            }))

        page.route("**/prompt", intercept_prompt)
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.server, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_function("() => !!window.comfyAPI?.app?.app?.canvas", timeout=60000)
        page.wait_for_function("() => {const app=window.comfyAPI?.app?.app; const types=app?.graph?.constructor?.registered_node_types || window.LiteGraph?.registered_node_types || {}; return ['QWMDetect','QWMResidualDetect','QWMRefineMask','QWMPrepare','QWMComposite','QWMFinal'].every(name => !!types[name]);}", timeout=30000)
        button = page.get_by_role("button", name=re.compile("运行|Run")).first
        button.wait_for(state="visible", timeout=60000)
        for name in ("qwen21_watermark", "qwen_watermark_detection"):
            workflow = json.loads((ROOT / "workflows" / f"{name}.json").read_text(encoding="utf-8"))
            result = page.evaluate("""async (workflow) => {
                const {app}=await import('/scripts/app.js');
                await app.loadGraphData(workflow);
                const prompt=await app.graphToPrompt();
                // A saved workflow must also remain safe when imported again.
                const saved=JSON.parse(JSON.stringify(app.rootGraph.serialize()));
                await app.loadGraphData(saved);
                const reloadedPrompt=await app.graphToPrompt();
                app.canvas.ds.scale=0.48;
                app.canvas.ds.offset=[10,100];
                app.canvas.setDirty(true,true);
                const nodes=app.rootGraph._nodes;
                return {nodes:nodes.map(n=>({id:n.id,type:n.type})),output:prompt.output,reloaded_output:reloadedPrompt.output,
                    null_text_widgets:nodes.flatMap(n=>(n.widgets||[])
                        .filter(w=>['text','customtext'].includes(w.type) && typeof w.value!=='string')
                        .map(w=>({id:n.id,name:w.name,value:w.value})))};
            }""", workflow)
            expected = json.loads((ROOT / "workflows" / f"{name}.api.json").read_text(encoding="utf-8"))
            result["roundtrip_differences"] = graph_differences(expected, result["output"])
            result["reload_differences"] = graph_differences(expected, result["reloaded_output"])
            before, errors_before = len(submissions), len(errors)
            button.click()
            for _ in range(40):
                page.wait_for_timeout(250)
                if len(submissions) > before or len(errors) > errors_before:
                    break
            result["queue_submissions_intercepted"] = len(submissions) - before
            result["queue_errors"] = errors[errors_before:]
            result["queue_differences"] = graph_differences(expected, submissions[-1]["prompt"]) if len(submissions) > before else []
            result["workflow"] = name
            results.append(result)
            (diagnostics / f"{name}_frontend_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if name == "qwen21_watermark":
                (diagnostics / "frontend_roundtrip.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                page.screenshot(path=str(diagnostics / "workflow_frontend.png"), full_page=True)
            print(name, "nodes:", len(result["nodes"]), "differences:", len(result["roundtrip_differences"]),
                  "queue submissions (intercepted):", result["queue_submissions_intercepted"], "errors:", result["queue_errors"])
        browser.close()
    if any(result["roundtrip_differences"] or result["reload_differences"] or result["null_text_widgets"] or result["queue_differences"] or
           result["queue_submissions_intercepted"] != 1 or result["queue_errors"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

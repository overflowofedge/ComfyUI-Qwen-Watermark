import argparse
import json
from pathlib import Path
import re
import sys
import time

import requests
from playwright.sync_api import sync_playwright


sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="从 ComfyUI 前端导入工作流并诊断运行按钮")
    parser.add_argument(
        "workflow",
        nargs="?",
        type=Path,
        default=Path.home() / "Desktop" / "qwen21_watermark_optimized.json",
    )
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    args = parser.parse_args()

    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    diagnostics = ROOT / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    events = {"console": [], "page_errors": [], "prompt_responses": []}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            headless=True,
            args=["--disable-gpu"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("console", lambda message: events["console"].append(
            {"type": message.type, "text": message.text}
        ))
        page.on("pageerror", lambda error: events["page_errors"].append(str(error)))

        def record_response(response):
            if response.url.rstrip("/").endswith("/prompt"):
                try:
                    body = response.text()
                except Exception as exc:
                    body = f"<unable to read response: {exc}>"
                events["prompt_responses"].append(
                    {"status": response.status, "url": response.url, "body": body}
                )

        page.on("response", record_response)
        page.goto(args.server, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_function("() => !!window.comfyAPI?.app?.app?.canvas", timeout=60_000)
        page.wait_for_function(
            "() => {const app=window.comfyAPI?.app?.app; "
            "const types=app?.graph?.constructor?.registered_node_types || "
            "window.LiteGraph?.registered_node_types || {}; "
            "return ['QWMDetect','QWMResidualDetect','QWMRefineMask','QWMPrepare','QWMComposite']"
            ".every(name => !!types[name]);}",
            timeout=30_000,
        )
        button = page.get_by_role("button", name=re.compile("运行|Run")).first
        button.wait_for(state="visible", timeout=60_000)
        graph = page.evaluate(
            """async (data) => {
                const {app}=await import('/scripts/app.js');
                await app.loadGraphData(data);
                const prompt=await app.graphToPrompt();
                return {
                    nodes: app.rootGraph._nodes.map(n => ({id:n.id,type:n.type,title:n.title})),
                    output: prompt.output,
                };
            }""",
            workflow,
        )
        events["graph"] = {"nodes": graph["nodes"], "output_nodes": len(graph["output"])}
        before = requests.get(f"{args.server}/queue", timeout=10).json()
        events["queue_before"] = before
        button.click()

        deadline = time.time() + 15
        while time.time() < deadline and not events["prompt_responses"]:
            page.wait_for_timeout(250)
        events["queue_after"] = requests.get(f"{args.server}/queue", timeout=10).json()
        page.screenshot(path=str(diagnostics / "frontend_queue_diagnostic.png"), full_page=True)
        browser.close()

    output = diagnostics / "frontend_queue_diagnostic.json"
    output.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(events, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

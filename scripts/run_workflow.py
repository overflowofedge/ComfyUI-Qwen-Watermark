import argparse
import json
from pathlib import Path
import sys
import time
import uuid

import requests


ROOT = Path(__file__).resolve().parents[1]


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必须是整数。") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0。")
    return parsed


def _nonnegative_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必须是整数。") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("不能小于 0。")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="通过本机 ComfyUI 运行 Qwen 2.1 水印工作流")
    parser.add_argument("image", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--boxes", type=Path, help="手动选区 JSON，指定后跳过自动识别")
    parser.add_argument("--hint", default="")
    parser.add_argument("--steps", type=_positive_int, default=20)
    parser.add_argument("--seed", type=int, default=802146)
    parser.add_argument("--max-side", type=_positive_int, default=768)
    parser.add_argument("--detection-side", type=_positive_int, default=1024)
    parser.add_argument("--mask-method", choices=["auto", "sam", "box"], default="auto",
                        help="auto 按检测类型路由；也可强制所有区域使用 SAM 或矩形框")
    parser.add_argument("--sam-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sam-dilation", type=_nonnegative_int, default=12, help="SAM 轮廓向外补偿的基础像素数")
    parser.add_argument("--fixed-sam-dilation", action="store_true",
                        help="关闭按贴片尺寸自动增加轮廓补偿")
    parser.add_argument("--pre-inpaint", choices=["none", "telea", "ns"], default="telea",
                        help="先用 OpenCV 填充遮罩再交给 Qwen；不规则贴片可尝试 telea/ns")
    parser.add_argument("--inpaint-radius", type=_positive_int, default=5)
    parser.add_argument("--timeout", type=_positive_int, default=3600)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"图片不存在：{args.image}")
    if args.boxes and not args.boxes.is_file():
        parser.error(f"手动选区文件不存在：{args.boxes}")
    if args.output.exists() and not args.output.is_dir():
        parser.error(f"输出路径不是目录：{args.output}")
    session = requests.Session()
    session.trust_env = False
    server = args.server.rstrip("/")
    name = "qwen_watermark_detection" if args.detect_only else "qwen21_watermark"
    graph = json.loads((ROOT / "workflows" / f"{name}.api.json").read_text(encoding="utf-8"))
    try:
        infos_response = session.get(server + "/object_info", timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"无法连接 ComfyUI：{server}。请先启动服务。") from exc
    infos_response.raise_for_status()
    try:
        object_info = infos_response.json()
    except ValueError as exc:
        raise RuntimeError("ComfyUI /object_info 返回的不是有效 JSON。") from exc
    missing = sorted({v["class_type"] for v in graph.values()} - object_info.keys())
    if missing:
        raise RuntimeError("ComfyUI 缺少节点：" + ", ".join(missing) + "。请安装本项目自定义节点并重新加载 ComfyUI。")
    upload_name = "qwm_" + uuid.uuid4().hex[:10] + args.image.suffix.lower()
    try:
        with args.image.open("rb") as image:
            response = session.post(server + "/upload/image", files={"image": (upload_name, image)}, data={"type": "input", "overwrite": "false"}, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"上传图片到 ComfyUI 失败：{server}。") from exc
    response.raise_for_status()
    try:
        upload = response.json()
    except ValueError as exc:
        raise RuntimeError("ComfyUI 上传接口返回的不是有效 JSON。") from exc
    if not upload.get("name"):
        raise RuntimeError("ComfyUI 上传接口没有返回文件名。")
    graph["1"]["inputs"]["image"] = "/".join(x for x in [upload.get("subfolder", ""), upload["name"]] if x)
    graph["3"]["inputs"].update(hint=args.hint, detection_side=args.detection_side)
    if args.boxes:
        graph["3"]["inputs"].update(mode="manual_boxes", boxes_json=args.boxes.read_text(encoding="utf-8"))
    if not args.detect_only:
        graph["4"]["inputs"].update(method=args.mask_method, device=args.sam_device,
                                     dilation=args.sam_dilation, adaptive_dilation=not args.fixed_sam_dilation)
        graph["5"]["inputs"]["max_side"] = args.max_side
        graph["5"]["inputs"].update(pre_inpaint=args.pre_inpaint, inpaint_radius=args.inpaint_radius)
        graph["12"]["inputs"].update(seed=args.seed, steps=args.steps)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        response = session.post(server + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"提交工作流到 ComfyUI 失败：{server}。") from exc
    if not response.ok:
        raise RuntimeError("ComfyUI 工作流校验失败：" + response.text)
    try:
        submission = response.json()
    except ValueError as exc:
        raise RuntimeError("ComfyUI /prompt 返回的不是有效 JSON：" + response.text[:500]) from exc
    if not submission.get("prompt_id"):
        raise RuntimeError("ComfyUI /prompt 没有返回 prompt_id：" + response.text[:500])
    if submission.get("node_errors"):
        raise RuntimeError(json.dumps(submission["node_errors"], ensure_ascii=False))
    prompt_id = submission["prompt_id"]
    print("Submitted", prompt_id, flush=True)
    (output / "submission.json").write_text(json.dumps({"prompt_id": prompt_id, "graph": graph}, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.monotonic()
    previous_state = None
    while time.monotonic() - start < args.timeout:
        try:
            response = session.get(server + "/history/" + prompt_id, timeout=20)
        except requests.RequestException as exc:
            raise RuntimeError(f"读取 ComfyUI 任务历史失败：{prompt_id}。") from exc
        response.raise_for_status()
        try:
            history_payload = response.json()
        except ValueError as exc:
            raise RuntimeError("ComfyUI /history 返回的不是有效 JSON。") from exc
        history = history_payload.get(prompt_id)
        if history is not None:
            break
        try:
            queue_response = session.get(server + "/queue", timeout=20)
        except requests.RequestException as exc:
            raise RuntimeError("读取 ComfyUI 队列状态失败。") from exc
        queue_response.raise_for_status()
        try:
            queue = queue_response.json()
        except ValueError as exc:
            raise RuntimeError("ComfyUI /queue 返回的不是有效 JSON。") from exc
        state = "running" if any(item[1] == prompt_id for item in queue["queue_running"]) else "queued"
        if state != previous_state:
            print(state, flush=True)
            previous_state = state
        time.sleep(2)
    else:
        raise TimeoutError(f"等待超时；ComfyUI 中的任务 {prompt_id} 仍可能继续执行，未中断其他任务。")
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    for kind, payload in history.get("status", {}).get("messages", []):
        if kind in ["execution_error", "execution_interrupted"]:
            raise RuntimeError(f"ComfyUI {kind}: " + json.dumps(payload, ensure_ascii=False))
    if not history.get("status", {}).get("completed"):
        raise RuntimeError("ComfyUI 未成功完成；请检查 history.json。")
    labels = {"15": "restored_first_pass", "16": "comparison_first_pass", "17": "detection",
              "21": "edited_crop", "33": "restored", "34": "comparison", "36": "residual_check"}
    files = {}
    for node_id, label in labels.items():
        for index, image in enumerate(history.get("outputs", {}).get(node_id, {}).get("images", [])):
            try:
                response = session.get(server + "/view", params=image, timeout=30)
            except requests.RequestException as exc:
                raise RuntimeError(f"下载 ComfyUI 输出文件失败：{label}。") from exc
            response.raise_for_status()
            path = output / f"{label}{'' if index == 0 else '_' + str(index)}.png"
            path.write_bytes(response.content)
            files[label] = str(path)
    reports = {}
    for node_id, key in [("3", "detection"), ("4", "refine"), ("5", "prepare"), ("14", "composite"),
                         ("26", "residual_detection"), ("27", "residual_refine"),
                         ("28", "residual_prepare"), ("32", "residual_composite")]:
        texts = history.get("outputs", {}).get(node_id, {}).get("text", [])
        if texts:
            reports[key] = json.loads(texts[0])
    if args.detect_only:
        reports["status"] = "detection_complete"
    elif reports.get("prepare", {}).get("status") == "no_overlay_detected":
        reports["status"] = "no_overlay_detected"
        print(reports["prepare"]["message"], flush=True)
    elif "restored" in files:
        reports["status"] = "completed"
    elif "restored_first_pass" in files and reports.get("residual_detection", {}).get("status") == "clean":
        files["restored"] = files["restored_first_pass"]
        files["comparison"] = files["comparison_first_pass"]
        reports["status"] = "completed_clean_first_pass"
    elif "restored_first_pass" in files and "residual_detection" not in reports:
        files["restored"] = files["restored_first_pass"]
        files["comparison"] = files["comparison_first_pass"]
        reports["status"] = "completed_first_pass_unverified"
    else:
        raise RuntimeError("ComfyUI 执行结束但没有生成修复图；请检查 history.json。")
    reports["elapsed_seconds"] = round(time.monotonic() - start, 2)
    reports["prompt_id"] = prompt_id
    reports["files"] = files
    (output / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

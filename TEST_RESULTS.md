# 本机测试结果

测试日期：2026-10-05。使用本机 ComfyUI 0.38.2、RTX 4060 Laptop 8 GB、64 GB 系统内存，以及已安装的 Qwen Image 2.1 BF16 / Qwen3-VL 8B INT8 ConvRot / Qwen 2.1 VAE。未调用外部生成 API。

## 自动识别 + 编辑 + 回贴

测试图：本机 `dancing.jpg` 缩至 768×512，添加半透明 `DEMO WATERMARK` 文字与圆形 W 图标；保留添加前图像和实际水印 alpha 范围用于核查。`scripts/create_demo.py` 可复现。

- 缓存关闭后的完整任务：`c071da1d-18d1-43c1-966d-697e20cbe1b1`。
- 自动识别两个区域，无人工选区、无水印位置提示。
- 20 步，CFG 1，局部参考图补边后 512×320；耗时 54.34 秒（不含服务启动）。
- 扩边后的自动选区覆盖测试水印像素的 100%。
- 实际水印覆盖区域相对添加前图片的 RGB RMSE：0.35385 → 0.02546，值域 0–1。
- 保存后的 PNG 与输入图在选区外逐像素一致，最大差异为 0；87.59% 图像像素处于选区外。
- 对齐相关性 0.99842，估计平移约 0.073 / 0.215 像素。
- 视觉检查：文字和圆形图案已移除，背景延续自然；人物姿态和外部画面保持。

结果保存在 `outputs/cache_off_demo_1/`。关闭 Qwen 前缀缓存后，又在同一个 CF 进程中连续提交手动框任务 `9314826f-7777-4202-9e36-2b82da9f0fe1`，26.20 秒完成，CF 保持运行。这验证了连续执行和模型切换路径。该结果证明这张受控样图的成功，不代表各种水印的总体成功率，也不代表精确恢复原始纹理。

## 真实半透明水印与手动修正

另测试本机 500×747 的城市图，其中画面中央存在半透明文字和图案水印。

- 自动定位耗时 16.11 秒，发现中央水印，但框选偏小，未完全覆盖文字边缘；见 `outputs/real_city_detection/`。
- 使用 `samples/city_boxes.json` 修正原图选区后，20 步编辑 + 回贴耗时 28.22 秒。
- 实际任务：`fe8b0b2f-99f8-4d39-8bdd-9fefa016152d`。
- 选区外像素最大差异 0，约 93.37% 图像像素保留在选区之外。
- 视觉检查：中央叠加水印已移除；遮挡区内建筑窗格属于模型推测，细节与原始建筑可能不同。

结果：`outputs/real_city_manual/comparison.png`。这张图说明低对比度水印仍可能需要人工调整框选。

## 用户图片的矩形花纹遮挡与空框修复

原先自动定位提示偏重文字和 Logo，用户图片上的矩形花纹贴片被当作衣服图案，返回 `{"boxes":[]}`，导致“03 裁剪并记录坐标”显示红错。现已将矩形贴片、硬边接缝、马赛克及图案遮挡加入默认识别提示；空遮罩通过 ComfyUI `ExecutionBlocker(None)` 跳过编辑，并在裁剪状态节点显示 `no_overlay_detected`。

- 用户原图为 2160×2880；未填写 `hint`，未使用手动框。
- 默认识别返回一个贴片区域；扩边后的原图选区为 `[618,1578,1704,2186]`。
- 完整任务：`0178bdc8-2d09-4725-bd6d-1c472b368a7b`，20 步、CFG 1、参考图 768×512，总耗时 66.62 秒。
- 对齐相关性 0.99887，估计平移 -1.134 / 0.095 像素，无对齐警告。
- 视觉检查：矩形花纹贴片已移除，被遮挡的皮肤、项链与衣服边缘得到重建。遮挡内的细节属于模型推测。
- 对保存后的 PNG 逐像素比对，选区外最大差异为 0，89.39% 图像像素处于选区外；结果保持原图 2160×2880 尺寸。基准为对比图左侧保存的实际 ComfyUI 输入，详见同目录 `metrics.json`。本机 ComfyUI 的 JPEG 解码与 Pillow 解码存在像素差异，因此不混用两种解码作为比对基准。

结果：`outputs/user_image_fixed_full/restored.png`；前后对比：`outputs/user_image_fixed_full/comparison.png`。

空选区回归任务 `38e48106-ab8e-48df-a8f2-35503be8260c` 使用手动空框，2.02 秒内成功结束，未生成编辑局部或修复图。通过真实 ComfyUI 执行验证了相同的空遮罩分支。脚本报告记录 `status=no_overlay_detected` 和裁剪提示，不把未编辑任务标记为修复完成。结果保存在 `outputs/empty_detection_regression/`。

## 验证

三十一项 Python 测试通过：覆盖原图坐标映射、SAM 框外轮廓恢复、异常大分割限制、自动与手绘遮罩的自适应外扩、补边还原、分离选区外像素保护、轻微平移校正、空选区停止、错误输出尺寸拒绝、错误框输入校验、整数扩边、遮罩尺寸校验、残留检测 JSON/Markdown 数组解析、残留不确定状态、自动路由报告、API/检测预览/UI 链接一致性检查，以及分享包文档和公开素材清单。

在本机 ComfyUI 前端实际导入更新后的工作流，并导回 API 图；节点参数和连接与提交版本一致，差异为 0。本机安装的五个 Python 节点文件与项目源码逐文件一致。采样使用 `TextEncodeQwenImage21` 输出的尺寸匹配 latent。检测专用工作流的遮罩预览直接连接定位节点，不依赖完整工作流中的 SAM 节点。

本机原先开启 `QwenImage21Cache=cpu` 时，在另一个 Qwen 任务切换到采样器的释放阶段出现一次原生进程中止，堆栈位于 `reset_prefix_cache/free`。交付工作流将缓存设为 `off`，随后连续两次实际采样正常完成。

## 2026-10-05 补强验证

- `python -m pytest -q`：31 项通过；`python -m compileall -q custom_nodes scripts tests packaging`：通过。
- ComfyUI 前端实际导入 `qwen21_watermark.json` 并回导 API 图：工作流节点、参数和连接校验通过，0 条参数差异。
- 通过 CLI 实际提交检测任务：26.16 秒完成并生成 `detection.png`。
- 通过 CLI 实际提交完整手动框任务（box 细化、1 步采样用于快速回归）：8.11 秒完成；`restored.png`、`comparison.png`、`detection.png`、`edited_crop.png` 均生成，`outside_mask_max_difference=0`。
- 自定义节点已重新安装到本机 ComfyUI，项目源码与安装副本逐文件 SHA-256 一致。

### v1.2 自动流程回归

v1.2 将通用扫描和 Logo 专项扫描合并到一次 Qwen3-VL 推理中，避免连续调用 ConvRot 后端造成的 CUDA 断言；随后按 `overlay_type` 自动选择 `box` 或 `sam`。第一次修复后由 `QWMResidualDetect` 检查残片、URL、透明鬼影、贴纸边缘和矩形接缝，只有置信度达到 0.65 才进入最多一次的第二遍局部修复；解析失败会报告 `indeterminate`，不会误报为干净。

- 普通 Logo 检测：`samples/demo_watermarked.png` 返回两处完整 `text_logo`，均为 `box` 路由；最新任务 `877a2b04-61f3-4204-9dce-99b15369ba12`，检测耗时 76.89 秒。
- 干净图检测：`samples/demo_clean.png` 返回空框，未触发盲修；任务 `c1ad726a-57cb-49a3-9372-216b0375de1f`，检测耗时 82.78 秒。
- 半透明城市水印：原图 500×747，自动识别为 `translucent_text_logo`，最终安全框 `[101,306,349,429]`，自动走 `box`；残留复检为空，状态 `completed_clean_first_pass`，选区外最大差异 `0`，对齐相关性 `0.99835`。结果见 `outputs/v12_city_calibrated_full/`。
- 不规则贴纸：无 `hint` 自动识别为 `irregular_sticker`，最终框 `[388,2192,972,2800]`，自动走 `sam`；`confidence=0.99`，结果见 `outputs/v12_detection_sticker_verified/`。
- 当前实机回归使用隔离端口 `8189` 和独立 SQLite 数据库；原端口 `8188` 的旧进程曾在 CUDA 断言后卡死，属于测试环境状态，不影响节点代码和分享包。

### 不规则贴纸回归

使用 2160×3840 的真实样例，画面下方小羊贴纸遮挡人物腿部、黑色服装和地板。原检测框下沿偏短，旧逻辑将 SAM 恢复出的框外轮廓再次裁回安全框，造成腿部和白边残片。

- SAM 候选现在允许在受控 `guard_box` 内越过检测安全框，并报告框外恢复像素与异常候选裁剪量。
- 默认启用自适应外扩：基础值 12 像素，大贴纸按提示框短边的 8% 增加，最高 64 像素；本例自动取 32 像素。
- 最终任务：`b5935719-43a3-4952-a32f-db4f56c726f6`，自动检测、SAM、Telea 预清理和 Qwen 编辑共耗时 86.56 秒。
- 精细遮罩为 184,195 像素，保留图像 97.7793%；`outside_mask_max_difference=0`。
- 视觉检查：贴纸主体、白边和可辨认的腿部残片均已移除，人物腿部和地板连续；结果见 `outputs/irregular_final/comparison.png`。

### 普通文字和 Logo 回归

使用 `samples/demo_watermarked.png` 按默认命令执行，不传手动框、提示词或 SAM 参数覆盖，确认贴纸优化不会扩大普通水印的修复范围。

- 自动检测到两处区域：两行 `DEMO WATERMARK` 文字和圆形 W Logo。
- 两个区域的 `effective_dilation` 均保持基础值 12，没有触发大贴纸补偿。
- 最终任务：`b8dab3d4-8fd6-432b-aaf7-209ad17c302b`，耗时 54.10 秒。
- 精细遮罩为 18,156 像素，保留图像 95.3827%；`outside_mask_max_difference=0`。
- 视觉检查：文字与圆形 Logo 均已移除，人物及道路结构正常；结果见 `outputs/ordinary_watermark_regression/comparison.png`。

### 手绘遮罩残影回归

使用用户在 ComfyUI 遮罩编辑器中保存的 2160×3840 RGBA 输入，重放相同遮罩、提示词、参数和种子。旧逻辑因手绘模式没有检测框而报告 `box_fallback`，节点上的 `dilation=12` 与自适应外扩均未生效，白边、四肢和投影被留在遮罩外并在回贴时形成半透明残影。

- 新逻辑不对 `manual_mask` 调用 SAM，但会根据手绘区域尺寸应用基础和自适应外扩。
- 本例原始手绘范围为 `[543,2251,1007,2760]`，152,309 像素；按短边 8% 自动取 38 像素外扩，精细遮罩增至 227,291 像素。
- 回归任务：`cb45dd8f-dbbd-4727-b630-a14677df6041`，状态 `success`，节点校验错误为 0。
- 对齐相关性 0.99964，估计平移 0.062 / -0.081 像素；`outside_mask_max_difference=0`。
- 视觉检查：小羊主体、白边、腿部和投影残影均已移除；结果为 ComfyUI `output/QwenWatermark/comparison_00036_.png`。

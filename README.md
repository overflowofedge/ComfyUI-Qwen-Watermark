# ComfyUI-Qwen-Watermark 1.3.0

2026-10-08 残留复检节点修补：每个区域的 Qwen3-VL 生成结束后，通过 ComfyUI 原生接口清理预取队列和 CUDA 图，避免下一块区域复用上一块区域的 KV 缓冲区。出现 `CUDA error: device-side assert triggered` / `aimdo memory compile error` 时，更新本包节点并重启 ComfyUI；无需关闭残留复检。

一套在本机 ComfyUI 运行的去水印工作流：Qwen3-VL 会在一次推理中完成通用扫描和 Logo 专项复核，自动识别文字、半透明 Logo、马赛克、矩形色块和不规则贴纸，再按类型选择矩形框或 SAM，交给 Qwen Image 2.1 做局部修复。第一次修复后还会自动检查残留，并在需要时最多追加一次局部修复；未被遮罩覆盖的像素会被逐像素保护。

> 仅处理你拥有版权或已获得授权的图片。遮挡下完全不存在的真实像素无法被恢复，模型只能根据上下文生成合理内容。

## Git 安装

在 ComfyUI Manager 中打开 “Install via Git URL”，粘贴：

```text
https://github.com/overflowofedge/ComfyUI-Qwen-Watermark.git
```

安装完成后重启 ComfyUI，再导入 `workflows/qwen21_watermark.json`。仓库根目录直接包含 ComfyUI 插件入口和依赖文件，Manager 克隆后无需再复制子目录。


## v1.3 贴回与产品化改动

人物肢体、服装边缘或建筑结构被重画时，可导入 `workflows/qwen21_watermark_structure.json`。该版本使用更宽的 384px 上下文、保留原参考图（`pre_inpaint=none`）、原裁剪 VAE 编码和遮罩加噪，限制采样范围；默认开启残留复检，高置信度残片会触发最多一次二采。在“02 自动定位 / 手动改框”的 `edit_prompt` 中填写要移除的叠加物与要保留的姿态/线条；“实际送入模型的编辑提示词”会显示完整指令。复检只判断叠加物残留，人物结构仍需对照原图检查。

针对贴纸腿部、脚部漏选，SAM 在原始紧框之外增加带内部定位点的轮廓补全；候选必须覆盖已有贴纸、置信度足够且面积增长有界，无关连通区域会被剔除。残留复检分别放大相距较远的选区，并比较原图与修复图，帮助识别失去主体的贴纸残腿；旧版无原图输入的工作流仍可运行。此修复需要更新自定义节点、重启 ComfyUI 并重新导入 JSON。

- 贴回不再用 Telea 重建整块选区后混色，也不把带遮挡的原像素混回生成图。全程保持 float32 RGB；软遮罩按修复范围处理。
- 只在未编辑上下文中估计有界颜色校正，并使用保留样本验证改善；颜色已经匹配时直接保留。边缘只延续选区外颜色差，保留生成纹理。
- 默认按类型选择预清理：文字/半透明水印直接编辑，块状遮挡与贴纸预清理；预清理在缩小后的局部裁剪内执行，选区外上下文不经过 8-bit 重编码。局部坐标写入几何报告，编辑提示词使用检测到的周围材质。
- 残留复检限于初始修复范围，二采遮罩限制在该范围的 24px 安全边内。取消按固定位置平移水印坐标的样例特例。
- `QWMFinal` 始终输出一个最终结果和质量报告：空选区返回原图；二采接缝恶化时保留一采；上下文严重失真时退回原图并明确标记失败。
- CLI 默认只保存最终图、对比和报告；`--debug-output` 保存完整诊断图，`--skip-residual` 省去视觉复检。批处理按图片顺序执行，失败继续，可按输入和工作流校验续跑。

升级必须同时更新自定义节点与 JSON，重启 ComfyUI 后重新导入。实际运行请看“最终结果（自动择优）”和“最终质量报告”，一采候选仅用于检查。完整行为和验收边界见 [产品化与验收指南](docs/产品化与验收指南.md)。

```powershell
python scripts/run_batch.py "C:\图片目录" --output "C:\处理结果" --resume
python scripts/replay_composite.py "已保存的任务目录" --mask "实际遮罩.png"
python scripts/run_workflow.py "C:\图片\原图.png" --structure-protection --edit-prompt "只移除贴纸，保留人物坐姿、腿部交叠关系、服装轮廓和地面透视。" --output "C:\结果"
```

`replay_composite.py` 可复用正常的一采裁剪重新贴回，无需重新采样；SAM/手绘任务必须使用实际遮罩。

## 效果示例

这是项目内置的可公开演示图。左图含两行文字水印和一个圆形 Logo，右图为去除结果，中间的黑白图是用于保护范围的遮罩。

<table>
<tr>
<td><img src="docs/images/demo-before.png" alt="带文字和 Logo 水印的演示图" width="360"></td>
<td><img src="docs/images/demo-after.png" alt="去除水印后的演示图" width="360"></td>
<td><img src="docs/images/demo-mask.png" alt="演示水印遮罩" width="360"></td>
</tr>
<tr><td align="center">输入：文字 + Logo</td><td align="center">输出：局部重建</td><td align="center">白色区域：允许编辑</td></tr>
</table>

下面是实际导入 ComfyUI 后的节点图。完整说明见 [`docs/参数与提示词.md`](docs/参数与提示词.md)。

![ComfyUI 工作流节点图](docs/images/workflow-overview.png)

仓库内的参数回归素材：`samples/demo_watermarked.png`（文字 + Logo）、`samples/demo_solid.png`（纯色色块）、`samples/demo_pattern.png`（矩形花纹）和 `samples/demo_exact_mask.png`（手绘遮罩）。前三张对应的 `*_boxes.json` 可直接填入“02 自动定位 / 手动改框”的 `boxes_json`；手绘图连接 `MASK` 输入。不同素材的参数、预期遮罩和检查点见 [节点参数与调参指南](docs/参数与提示词.md#用仓库测试图片验证参数)。

## 适用范围

| 遮挡类型 | 推荐路径 | 关键设置 |
| --- | --- | --- |
| 普通文字、Logo、小型半透明水印 | `auto_qwen` + `auto` | Logo 专项复扫，自动走 `box`，按类型限制扩边 |
| 马赛克、矩形花纹、纯色色块 | `auto_qwen` + `auto` | 自动走 `box`；误检时可用手动框修正 |
| 不规则贴纸、动物/人物贴片 | `auto_qwen` + `auto` | 自动走 `sam`，大型贴纸自适应外扩 |
| 边缘复杂或贴近必须保留的细节 | `manual_mask` | 画白需要移除的区域，必要时关闭自适应外扩 |

工作流不会把整张图交给编辑模型，而是保留上下文的局部裁剪；`QWMComposite` 只在最终遮罩内合成，并检查选区外最大像素差。详细参数、提示词模板和故障排查见 [参数与提示词指南](docs/参数与提示词.md)。

## 快速开始

### 1. 准备 ComfyUI 和模型

建议使用 ComfyUI 0.38.2 或更新版本，并确认当前版本提供官方节点 `TextEncodeQwenImage21`、`QwenImage21Cache`。模型文件不包含在仓库或压缩包中，请按下表放置：

| 文件 | 目录 |
| --- | --- |
| `qwen_image_2.1_bf16.safetensors` | `models/unet/`（部分版本也叫 `diffusion_models/`） |
| `qwen3vl_8b_int8_convrot.safetensors` | `models/clip/` 或 `models/text_encoders/` |
| `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` |
| `sam_vit_b_01ec64.pth` | `models/sams/` |

### 2. 安装 ComfyUI 插件

本仓库本身就是 ComfyUI 自定义节点插件。使用 ComfyUI Manager 的“Install via Git URL”粘贴本仓库地址即可；也可以直接克隆到 `ComfyUI/custom_nodes`：

```powershell
cd "E:\ComfyUI\ComfyUI\custom_nodes"
git clone https://github.com/overflowofedge/ComfyUI-Qwen-Watermark.git
python -m pip install -r .\ComfyUI-Qwen-Watermark\requirements.txt
```

本项目新增六个自定义节点，不是 ComfyUI 内置节点。搜索时使用画布标题；括号内是源码类名：`02 自动定位 / 手动改框`（`QWMDetect`）、`03 不规则轮廓细化（SAM）`（`QWMRefineMask`）、`04 裁剪并记录坐标`（`QWMPrepare`）、`06 对齐 / 保色贴回`（`QWMComposite`）、`07 第一次修复后检查残留`（`QWMResidualDetect`）、`12 质量检查 / 选择最终结果`（`QWMFinal`）。`QWMRefineMask`、`QWMPrepare` 和 `QWMComposite` 在二采区域会各出现一次，按画布编号区分。只导入 JSON 会出现红色缺失节点；如果使用本地工作区脚本安装，也可以运行：

```powershell
python scripts/install_nodes.py --comfy-root "E:\ComfyUI-aki-v3\ComfyUI-aki-v3\ComfyUI"
```

或者下载 Releases 中的 `Qwen-Watermark-Workflow-*.zip`，解压后运行包内 `install.py`。安装后重启 ComfyUI，并在浏览器按 `Ctrl+F5`。

### 3. 导入并运行

1. 导入 `workflows/qwen21_watermark.json`。
2. 在“01 上传原图”节点重新选择图片。
3. 默认保持 `auto_qwen + auto`，工作流会按遮挡类型自动选择 `box` 或 `sam`；仅在低置信度、误检或漏检时改用手动框/遮罩。
4. 如果要手动修复，在“01 上传原图”的遮罩编辑器中画白需要移除的区域，把“02 自动定位 / 手动改框”的 `mode` 改成 `manual_mask`；该工作流已预连 `MASK` 输入，并会自动外扩边缘。
5. 需要改变修复内容时，在“02 自动定位 / 手动改框”的 `edit_prompt` 输入框填写补充要求。编辑节点的 `prompt` 是自动连线，不能直接在 TextEncode 节点内编辑。
6. 运行后查看“定位预览”“细化遮罩预览”“残留复检”和左右对比。

只想确认自动框选时，导入 `workflows/qwen_watermark_detection.json`。命令行批处理示例：

```powershell
python scripts/run_workflow.py "C:\图片\原图.png" --output outputs\my_image
python scripts/run_workflow.py samples\demo_watermarked.png --detect-only --output outputs\detection
python scripts/run_workflow.py "C:\图片\原图.png" --edit-prompt "保留人物和背景结构，只去除右下角叠加物。"
```

## 结果与验证

- 本机环境：ComfyUI 0.38.2、前端 1.53.6、RTX 4060 Laptop 8 GB。
- 普通文字 + Logo 回归：两处 `effective_dilation=12`，选区外最大差异 `0`。
- 不规则贴纸回归：SAM 允许轮廓在安全边界内越过检测框，自动外扩示例为 `32 px`，选区外最大差异 `0`。
- 手绘遮罩回归：大型区域自适应外扩 `38 px`，选区外最大差异 `0`。
- Python 回归测试：`68 passed`（v1.3，含 Git 根目录插件结构、生成缓存释放、贴纸轮廓补全、分区参照复检、文本空值兼容与遮罩采样回归）。

如果点击运行出现 `Cannot read properties of null (reading 'replace')`，请重新导入已修复的工作流 JSON。旧导出器将已连线的文本控件占位值保存为 `null`，会触发 `comfyui-custom-scripts` 的文本预设序列化错误；修复后的值为 `""`，参数顺序与连线保持正确。仅此修复无需重装模型或重启后端。

完整实测数据见 [`TEST_RESULTS.md`](TEST_RESULTS.md)。

## 项目结构

```text
__init__.py, nodes.py, processing.py  ComfyUI 插件入口与六个自定义节点
workflows/                              可导入的 UI/API 工作流
scripts/                                安装、运行、构建和验证脚本
packaging/                              分享包说明、安装器和许可证
docs/参数与提示词.md                    参数表、提示词模板、排错指南
samples/                                不含模型的公开演示素材
tests/                                  处理逻辑和工作流结构测试
```

## 许可证

仓库代码按 [`LICENSE`](LICENSE) 发布。Qwen、SAM 及其他模型权重遵循各自许可证；本项目不重新分发模型权重。

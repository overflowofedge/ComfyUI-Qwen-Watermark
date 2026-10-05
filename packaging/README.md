# Qwen Image 2.1 去水印工作流 1.1.0

这是可分享的本机 ComfyUI 工作流包。它支持普通文字与 Logo、马赛克、矩形花纹或纯色色块、不规则贴纸，以及手绘遮罩。完整参数表和提示词模板见 [`docs/参数与提示词.md`](docs/参数与提示词.md)。

## 包含内容

- `workflows/qwen21_watermark.json`：可导入 ComfyUI 的工作流。
- `custom_nodes/ComfyUI-Qwen-Watermark/`：工作流必需的四个自定义节点。
- `docs/参数与提示词.md`：按场景的参数表、正负向提示词和排错顺序。
- `install.py`：安装节点、依赖并检查模型。
- `docs/images/`：公开演示图；不包含用户原图、输出目录或诊断日志。
- `LICENSE`：本包代码许可。模型权重遵循各自许可，不包含在本包内。

## 环境要求

- 当前版 ComfyUI，必须包含官方节点 `TextEncodeQwenImage21` 和 `QwenImage21Cache`。
- 已在 ComfyUI 0.38.2、前端 1.53.6、RTX 4060 Laptop 8GB 上实测。
- 8GB 显存可以运行，16GB 以上更宽裕；大图仍会以局部 768px 路线处理。

## 安装

关闭 ComfyUI，在本目录运行：

```powershell
python install.py "E:\ComfyUI\ComfyUI"
```

Aki 或其他便携版只要参数指向包含 `main.py` 的 ComfyUI 根目录即可。安装器会优先使用便携版自带的 Python 安装依赖。完成后重启 ComfyUI，并在浏览器按 `Ctrl+F5`。

也可以手动将 `custom_nodes/ComfyUI-Qwen-Watermark` 复制到 ComfyUI 的 `custom_nodes` 目录，再执行该目录中的 `requirements.txt`。

## 模型文件

权重不包含在压缩包内。文件名和位置必须能被 ComfyUI 识别：

| 文件 | 目录 | 本机文件大小 |
| --- | --- | ---: |
| `qwen_image_2.1_bf16.safetensors` | `models/unet/` | 约 13.25GB |
| `qwen3vl_8b_int8_convrot.safetensors` | `models/clip/` 或当前版本的 `models/text_encoders/` | 约 8.71GB |
| `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` | 约 644MB |
| `sam_vit_b_01ec64.pth` | `models/sams/` | 约 358MB |

SAM ViT-B 官方下载地址：<https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>。Qwen 模型请从所用 ComfyUI 发行版或模型发布页获取，并遵守对应许可。

## 使用

1. 导入 `workflows/qwen21_watermark.json`。
2. 在 `01 上传原图` 节点重新选择本机图片。
3. 普通水印可用 `auto_qwen`；矩形色块优先 `manual_boxes + box`；不规则贴纸优先 `manual_boxes + sam`。
4. 使用手绘遮罩时，在 LoadImage 的遮罩编辑器绘制白色区域，将 `MASK` 输出连接到 `QWMDetect.manual_mask`，并把模式设为 `manual_mask`。
5. 运行后查看 `定位预览`、`细化遮罩预览` 和左右对比图。

`manual_mask`、SAM 路径会应用 `dilation`；大型贴纸建议开启 `adaptive_dilation`。`box` 路径通过检测节点的 `expand` 扩大范围，`dilation` 对它不生效。

## 常见问题

- 导入后节点为红色：未安装本包中的自定义节点，或安装后没有重启 ComfyUI。
- 点击运行没有结果：在 LoadImage 节点重新选择图片；若状态为 `no_overlay_detected`，填写 `hint` 或使用手动框/遮罩。
- 手绘遮罩仍有残影：确认使用 1.1.0 节点并开启 `adaptive_dilation`；增加 `dilation`，不要用更大的 `feather` 代替遮罩外扩。
- 模型下拉框为空：模型目录或文件名不匹配。

## 说明

遮挡区域的内容属于生成式重建，并不等同于恢复原始像素。仅处理你拥有版权或已获授权的素材。

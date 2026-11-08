# PresenceGuard v2 · 开发文档

> 一个只用一台旧安卓手机就能跑的「人影守望」应用：相机对着门/走廊/院子，
> 画面里出现人就出声提醒。全程本地推理，不联网。
>
> 本文档描述 v2（YOLO 重写版）。v1 的实现见 `presence-guard-v1` 仓库的 `docs/DEVELOPMENT.md`。
> 两份文档可以对照看：v2 很多设计是"v1 踩过坑之后的结论"，文中会标出来。

---

## 目录

1. [v2 与 v1 的关系](#1-v2-与-v1-的关系)
2. [快速开始](#2-快速开始)
3. [目录结构](#3-目录结构)
4. [整体架构](#4-整体架构)
5. [相机与预览层](#5-相机与预览层)
6. [模型与推理层](#6-模型与推理层)
7. [三条检测路线](#7-三条检测路线)
8. [融合判定层](#8-融合判定层)
9. [多边形选区](#9-多边形选区)
10. [前台服务与息屏](#10-前台服务与息屏)
11. [提示音](#11-提示音)
12. [参数速查表](#12-参数速查表)
13. [实测性能](#13-实测性能)
14. [构建与调试](#14-构建与调试)
15. [已知限制](#15-已知限制)
16. [双指缩放：数字裁切，不是换镜头](#16-双指缩放数字裁切不是换镜头)
17. [选区外零计算](#17-选区外零计算)
18. [瓦片帧差跳过](#18-瓦片帧差跳过)
19. [2 秒全图巡检与前后双摄](#19-2-秒全图巡检与前后双摄)
20. [界面布局：为什么把指标行塞进底栏容器](#20-界面布局为什么把指标行塞进底栏容器)
21. [版本演进](#21-版本演进)

---

## 1. v2 与 v1 的关系

| | v1 | v2 |
|---|---|---|
| 检测引擎 | ML Kit Pose + TFLite EfficientDet-Lite0（双引擎互补印证） | 单路 YOLOv10n（TFLite，自己写 Interpreter 层） |
| 推理框架 | `tensorflow-lite-task-vision` + `mlkit-pose-detection` | 只有 `tensorflow-lite`，解码全部自己写 |
| 输入尺寸 | 320×320（两个引擎各一次） | 640×640 |
| 关键点 | 33 个姿态关键点 + 骨架绘制 | 无（YOLO 不做关键点） |
| 误报压制 | 空间：两个引擎在同一位置互相印证 | 时间：同一位置连续命中 N 帧 |
| APK | 56 MB | 13 MB |
| 单帧推理 | ~80ms（检测）+ ~70ms（姿态） | ~190ms（一路全包） |

**为什么换**：v1 的天花板是 EfficientDet-Lite0 @320 的能力边界 —— 25~45px 的远处目标
特征在缩放后不足 10px，怎么调参都出不来（v1 文档第 5 节实测）。要突破只能上 640 输入的
YOLO 类模型，而 YOLO 的 TFLite 导出没有 metadata，`task-vision` 不认，所以推理层只能自己写。

**v2 放弃了什么**：没有关键点，就没有骨架可视化，也没法用「双肩是否水平」这类几何判据。
换来的是更简单的判定链路和更好的小目标召回。

---

## 2. 快速开始

### 2.1 环境

```
JDK      21（Android Studio 自带 JBR 即可）
Gradle   8.11.1
AGP      8.5.2
Kotlin   2.0.21
compileSdk 34 / minSdk 26 / targetSdk 34
ABI      仅 arm64-v8a
```

### 2.2 构建安装

```bash
export JAVA_HOME=~/.jdks/jbr-21.0.11
gradle :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.presenceguard.yolo/.MainActivity
```

首次运行需要授予相机权限。没授权限时前台服务不会启动，界面只显示提示。

### 2.3 离屏验收探针（debug 构建才有）

改算法前后要客观对比，不能靠肉眼盯屏幕。探针用固定样图跑完整流程，结果打到 logcat：

```bash
# 整帧 + 全瓦片扫描
adb shell am start -n com.presenceguard.yolo/.ProbeActivity \
    --es image probe_far_lake.jpg --ei tiles 1

# 指定瓦片网格
adb shell am start -n com.presenceguard.yolo/.ProbeActivity \
    --es image probe_far_plaza.jpg --ei tiles 1 --ei cols 6 --ei rows 4

# 指定后端 / 线程数
adb shell am start -n com.presenceguard.yolo/.ProbeActivity \
    --es image probe_far_lake.jpg --es accel nnapi --ei threads 4

# 只做输入归一化诊断（打原始输出张量）
adb shell am start -n com.presenceguard.yolo/.ProbeActivity \
    --es image probe_far_lake.jpg --ez diag true

adb logcat -d -s Probe:* YoloDetector:* tflite:*
```

**探针有一条极其重要的实现约束**，见 [14.4](#144-探针前台陷阱非常重要)。

---

## 3. 目录结构

```
app/src/main/java/com/presenceguard/yolo/
├── MainActivity.kt              界面、权限、控制栏、叠加层接线
├── core/
│   ├── DetectionConfig.kt       检测参数 + 三档灵敏度的阈值组合
│   ├── PersonCandidate.kt       单个候选框（框、分数、来源、证据）
│   ├── Roi.kt                   多边形选区数据模型（归一化坐标）
│   └── FrameResult.kt           一帧的汇总结果
├── detect/
│   ├── YoloDetector.kt          TFLite Interpreter 封装：预处理 / 推理 / 解码
│   ├── PresenceFusionEngine.kt  四层判定：形状 → 面积 → 轨迹 → 投票迟滞
│   ├── FrameAnalyzer.kt         三条路线的调度与坐标映射
│   └── TileGrid.kt              瓦片网格切分
├── service/
│   └── DetectionService.kt      前台服务：相机绑定、用例管理、变焦、镜头、选区
├── ui/
│   └── OverlayView.kt           叠加层：画框、画多边形、触摸取点
├── audio/
│   └── AlertPlayer.kt           提示音合成与播放
├── data/
│   └── Settings.kt              SharedPreferences 封装
└── ...

app/src/debug/java/com/presenceguard/yolo/
└── ProbeActivity.kt             离屏验收探针（不进正式包）

app/src/main/assets/
├── yolov10n_int8.tflite         模型（3,125,847 字节）
└── probe_far_*.jpg              两张刻意刁难的验收样图
```

代码量约 3,700 行 Kotlin（不含布局与资源）。

---

## 4. 整体架构

```
                    ┌──────────────────────────────────────────┐
                    │  DetectionService（前台服务，常驻）        │
                    │                                          │
  CameraX           │  bindCamera() ──┬─ Preview?   仅前台时绑定 │
  ─────────         │                 └─ ImageAnalysis 永远绑定 │
  Provider ────────►│                                          │
                    │  applyZoom() / selectorFor(lensIndex)     │
                    └───────────────┬──────────────────────────┘
                                    │ ImageProxy（1080×1440）
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │  FrameAnalyzer（分析线程，同步执行）        │
                    │                                          │
                    │  rotate/convert → 选路线 → 推理 → 解码     │
                    │       │                                  │
                    │       ├─ 选区模式：裁多边形外接矩形 → 放大送检│
                    │       ├─ 瓦片模式：滚 N 片                 │
                    │       └─ 整帧模式：一次 640×640            │
                    └───────────────┬──────────────────────────┘
                                    │ List<PersonCandidate>
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │  PresenceFusionEngine                     │
                    │  形状门限 → 面积门限 → 小目标轨迹 → 投票迟滞 │
                    └───────────────┬──────────────────────────┘
                                    │ Outcome(present, count, ...)
                     ┌──────────────┼───────────────┐
                     ▼              ▼               ▼
              OverlayView     AlertPlayer     MainActivity
              （画框/选区）    （提示音）      （状态卡文字）
```

关键点：**推理是同步的，没有线程池，没有 CountDownLatch**。v1 之所以需要并行，
是 ML Kit 的回调式 API 逼的；v2 只有一路同步推理，直接在分析线程跑完更简单，
也顺手消掉了「`BitmapMlImage.close()` 会回收底层 Bitmap」那一整类崩溃。

---

## 5. 相机与预览层

### 5.1 分析流分辨率

```kotlin
const val ANALYSIS_WIDTH = 1920
const val ANALYSIS_HEIGHT = 1080
// 实测设备上竖屏拿到的是 1080×1440
```

为什么不是更低：选区放大和瓦片放大的本质都是「裁一小块 → 缩放到 640」。
采集本身糊了，放大出来的只是模糊块，模型一样看不见。720p 是 865 上性价比最高的折中。

`STRATEGY_KEEP_ONLY_LATEST`：处理不过来就丢旧帧，绝不排队。延迟比帧率重要。

### 5.2 坐标映射

模型吃的是 letterbox 后的 640×640，界面画的是全屏 `PreviewView`。中间有两层变换：

1. **letterbox 逆变换**：模型输出 → 原图像素坐标
   ```kotlin
   val mul = if (normalized) inputSize.toFloat() else 1f
   val left = ((x1 * mul) - padX) / scale + offsetX
   ```
2. **PreviewView 适配**：`FILL_CENTER` 意味着视图按"取大"比例缩放并居中裁切
   ```kotlin
   val scale = max(width / srcW.toFloat(), height / srcH.toFloat())
   ```

前置摄像头额外要镜像。`OverlayView` 用 `mirrored` 标记处理，**只在画的时候翻，
不在坐标上翻**，这样点击取点不需要做二次逆变换。

### 5.3 预览生命周期与功耗（v1 的核心结论，v2 继承）

**预览是耗电大户：相机输出流 + 显示合成约占 13~17% 单核。** 熄屏后没人看画面，
这部分应该完全归零。

**只调用 `Preview.setSurfaceProvider(null)` 是不会归零的** —— 用例还绑在相机上，
那条输出流照样在跑。必须把它从 `bindToLifecycle` 的用例列表里去掉：

```kotlin
val useCases = buildList {
    if (previewVisible) add(preview)
    add(analysis)
}
camera = provider.bindToLifecycle(this, selectorFor(lensFacing), *useCases.toTypedArray())
```

触发时机（这是 MIUI 上踩过的坑）：

```kotlin
// MainActivity
override fun onResume()  { super.onResume();  if (bound) attachPreview() }
override fun onPause()   { super.onPause();   service?.setPreviewVisible(false) }

// 双保险：熄屏广播
private val screenReceiver = object : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_SCREEN_OFF -> { screenOn = false; service?.setPreviewVisible(false) }
            Intent.ACTION_SCREEN_ON  -> { screenOn = true;  attachPreview() }
        }
    }
}
```

**为什么是 `onPause` 而不是 `onStop`**：MIUI（Android 13）熄屏时 Activity 只走到
`onPause`，`onStop` 经常不触发（`dumpsys activity` 里能看到 `mLastPausedActivity`，
但没有 stop）。只在 `onStop` 里摘预览，熄屏后 CPU 会一直停在 130% 下不来。

实测效果见 [13.3](#133-cpu-占用)。

### 5.4 变焦与多镜头

**变焦**走 `CameraX.setZoomRatio()`，这是**传感器级区域裁切**，对预览流和分析流同时生效，
不是数码放大（数码放大只会得到模糊块）。

```kotlin
fun setZoomRatio(ratio: Float) {
    zoomRatio = ratio.coerceIn(range.first, range.second)
    applyZoom()
}
```

按钮循环 1x → 2x → 3x → 5x → 8x。倍率持久化。

**多镜头**：有些机型暴露多颗后置相机（广角/超广/长焦）。枚举方式：

```kotlin
private fun collectBackCameras(provider: ProcessCameraProvider): List<CameraInfo> =
    provider.availableCameraInfos.filter {
        CameraSelector.DEFAULT_BACK_CAMERA.filter(listOf(it)).isNotEmpty()
    }

private fun selectorFor(index: Int): CameraSelector =
    if (backInfos.size <= 1) CameraSelector.DEFAULT_BACK_CAMERA
    else CameraSelector.Builder()
        .addCameraFilter { list -> list.filter { it == backInfos[index] } }
        .build()
```

实测设备（小米 M2007J1SC）只暴露 1 颗后置相机，所以「焦段」按钮自动隐藏 ——
这是正常行为，不是功能缺失。

> **变焦 vs 选区的区别**：变焦改变相机怎么采样（真的看得更远，但视野变窄、帧率不变）；
> 选区不改变采样，只是把这块区域裁出来喂给模型（等效长焦，但受采集分辨率限制）。
> 追远处的人用变焦，盯固定区域用选区，两个可以叠加。

---

## 6. 模型与推理层

### 6.1 为什么自己写 Interpreter 层

`tensorflow-lite-task-vision` 是给带 metadata 的模型（EfficientDet / MobileNet）准备的。
YOLO 的 TFLite 导出没有配套 metadata，Task 库认不出来；而且 v1 实测 Task 库的 JNI
**不接受量化模型**（`UnsatisfiedLinkError`），还有一个更恶心的特性：
Task 的原生库是**进程级共享**的，关掉最后一个 detector 会卸载它，
同进程再创建新的必然失败。

自己写这层只有几百行，换来三件事：

- 能把量化模型的 `scale` / `zeroPoint` 用对；
- 能按输出形状自适应（v10 的端到端头和 v8 的原始头解码方式完全不同）；
- 想去掉一层依赖就去掉，APK 从 56MB 降到 13MB。

### 6.2 模型选型（这一段是实测记录，不要重蹈覆辙）

候选来自 HF 镜像 `anan19990108/yolov10n-tflite`，三个文件全试过：

| 文件 | 大小 | 结果 |
|---|---|---|
| `yolov10n_int8.tflite` | 3,125,847 | ✅ **采用**。混合量化（int8 权重 + float 激活），湖景图 0.748 / 0.690 / 0.347 |
| `yolov10n_integer_quant.tflite` | 3,092,559 | ❌ 输出退化：300 行里 4 个坐标列全是同样两个常数，类 id 随行号递增 |
| `yolov10n_full_integer_quant.tflite` | 3,092,615 | ❌ 输出张量标定失败：`scale=0.3104 / zero=-128`，动态范围 [-39.7, 39.7]，输出只有 3 个台阶 |

后两个是同一个坏图的不同量化版本。根因是 YOLOv10 的端到端头里有 top-k / argsort，
TFLite 转换时这一步没接上（类 id 递增 = 某个 `arange` 被常量折叠了）。
**这不是配置问题，是模型文件本身坏了**，换参数救不回来。

识别坏模型的方法（`--ez diag true`）：把输出张量摊平打出来看前 36 个数。
正常的 YOLO 输出应该是 300 行各不相同的框 + 递减的分数；
如果所有行的坐标一模一样，或者分数只有几个离散台阶，就是坏的。

模型校验和：

```
sha256  6319e564bac432d8fc95d82fee06c4cb37879ce4830eb5bb1c292a65e644f105
```

### 6.3 预处理：letterbox

等比缩放 + 灰边（114,114,114）填充到 `640×640`：

```kotlin
val scale = min(640f / srcW, 640f / srcH)
val padX = (640 - drawW) / 2
val padY = (640 - drawH) / 2
```

不用直接拉伸：Ultralytics 训练/验证都用 letterbox，拉伸会让宽高比失真，
检测框系统性偏胖或偏瘦。

写入用 `Canvas.drawBitmap` 硬件加速路径一次画完，再 `getPixels` 取回来转 float/int8。
比逐个像素采样快一个数量级。

### 6.4 输出布局自适应

同一个「YOLO」名字下有几种完全不同的输出头，靠形状判断：

| 形状 | 含义 | 处理 |
|---|---|---|
| `[1, N, 6]` | YOLOv10 端到端头，每行 `(x1,y1,x2,y2,score,cls)` | 已无重复框，直接用 |
| `[1, 4+C, N]` | YOLOv5/v8 原始头，通道优先 | 自己解 `(cx,cy,w,h)` + 做 NMS |
| `[1, N, 4+C]` | 同上但行优先 | 同上 |

坐标是归一化 0~1 还是 0~640 像素，靠量级判断（人不可能只有 1.5 像素高）：

```kotlin
val normalized = max(max(x1, y1), max(x2, y2)) <= NORMALIZED_LIMIT  // 1.5f
```

判断结果记在 `normalizedCoords` 里，诊断行会打 `coords=norm|px`。
本机实测是 `coords=norm`。

### 6.5 量化与反量化

**输入量化**（输入张量是 INT8/UINT8 时）：

```kotlin
q = round(value / scale) + zeroPoint
```

这里踩过一个很隐蔽的坑：早先的代码对两种整数类型一律 `coerceIn(-128, 127)`。
对 INT8 是对的；对 **UINT8**（zeroPoint=0，值域 0~255）就是灾难 ——
所有亮度超过 50% 的像素被压成同一个值，送进网络的是一张糊掉的图，
输出全是常数。而且**它不报错，只是安静地低分**。所以现在按 dtype 分开钳位。

**输出反量化**（输出张量是 INT8 时）：

```kotlin
q = if (type == INT8) byte.toInt() else (byte.toInt() and 0xFF)
value = (q - zeroPoint) * scale
```

`INT8` 有符号、`UINT8` 要按无符号展开，弄反会让零点偏移 256 个量化级。

### 6.6 后端选择与回退链

```kotlin
enum class Accel { CPU, CPU_BASIC, NNAPI }
```

| 后端 | 实测结果 |
|---|---|
| `CPU`（XNNPACK，4 线程） | ✅ **默认**。191ms |
| `CPU_BASIC`（关 XNNPACK） | ❌ 836ms，更慢。XNNPACK 确实是 CPU 上最优路径 |
| `NNAPI` | ❌ 编译失败。`Replacing 19 out of 352 node(s)` 后 `ANEURALNETWORKS_BAD_DATA` |
| `GPU`（tensorflow-lite-gpu 2.14） | ❌ `Replacing 13 out of 352 node(s) ... Created 0 GPU delegate kernels`，构造直接抛异常 |

GPU 和 NNAPI 的失败**发生在 `Interpreter` 构造时**，不是运行时降级 ——
一次失败就等于整个检测器不可用。所以必须写成回退链：

```kotlin
for (backend in backends) {
    val attempt = runCatching { /* 配好 options，构造 Interpreter */ }
    attempt.onSuccess { return it }.onFailure { Log.w(TAG, "后端 $backend 不可用，退到下一个") }
}
```

GPU 委托对 int8 张量在 2.14 上基本没有支持（哪怕显式
`setQuantizedModelsAllowed(true)`），YOLOv10 端到端头里的 `CAST→INT64`、`ADD v4`
也不支持。既然一个内核都上不了，就没有理由为它背 13MB 原生库，**依赖已经移除**。

---

## 7. 三条检测路线

`FrameAnalyzer` 里三条路线，按场景自动选，**互斥**：

### 7.1 整帧（默认）

每帧对整帧跑一次 `640×640`。成本固定 191ms，覆盖近处大目标。
远处小目标缩到 640 后只剩十几像素，特征消失 —— 这条路救不了小目标。

### 7.2 瓦片（按需）

把画面切成网格，每帧只滚几片。放大倍数 = `640 / 瓦片边长`。

```
TileGrid(cols = 3, rows = 2, overlap = 0.25f)   // 默认 6 片
```

`overlap` 是相邻瓦片的重叠比例，避免人正好跨在接缝上被切成两半。

每帧滚几片由灵敏度决定：

```kotlin
private fun tilesForSensitivity(s: Sensitivity) = when (s) {
    Sensitivity.HIGH -> 1     // 灵敏档开一片，扫得慢但召回高
    else -> 0                 // 其余关闭，整帧就够
}
```

而且**一旦确认有人就停瓦片**（`lastPresent` 时 budget 归零），不再为已经找到的人花钱。

> 关键认知：**提高采集分辨率不能替代瓦片**。分辨率提高后，缩放到 640 时增益被抵消，
> 送进模型的人还是那么大。唯一有效的手段是缩小视野。

### 7.3 选区（用户画了多边形之后）

见第 9 章。**选区存在时，另外两条路线完全跳过** —— 这是省算力的核心手段。

---

## 8. 融合判定层

`PresenceFusionEngine`，四层。设计目标是「宁可晚 1 秒报，不要天天误报」。

### 8.1 第一层：形状门限

```kotlin
val ratio = box.height() / box.width()
if (ratio !in 0.15f..12f) return null    // 横得像横杆、竖得像立柱，都不像人
```

只拦"离谱"的形状。人躺着、坐着、只露一条腿都在区间内，不会误伤。

> v1 的教训：**不要在这里加"人必须是竖直的"这类朝向相关约束**。
> v1 的"双肩必须水平""髋必须在肩下方"在俯拍、侧躺、蹲着时全废。

### 8.2 第二层：面积门限（双向）

```kotlin
val areaRatio = box.area / (srcW * srcH)
if (areaRatio < cfg.minAreaRatio) return null   // 太小：远处噪点、墙角
if (areaRatio > cfg.maxAreaRatio) return null   // 太大：贴镜头、纯色面被当成人
```

`minAreaRatio` 三档分别是 0.0012 / 0.0030 / 0.0100。

### 8.3 第三层：小目标轨迹持久化

这是 v2 补小目标召回的核心手段 —— **用时间证据换空间置信度**。

```kotlin
if (areaRatio < cfg.smallTargetAreaRatio) {
    val track = tracks.match(box)            // IoU > TRACK_IOU(0.20)
    if (track == null) { tracks.add(box); return null }   // 第一帧先记账
    track.hits++
    if (track.hits < cfg.smallTargetMinHits) return null  // 还没攒够
    if (score < cfg.smallTargetMinScore) return null
    score = max(score, cfg.smallTargetScoreFloor)          // 采信后给个分数下限
}
```

为什么这样对：远处的人单帧分数确实低（0.3 很正常），但**它会一直待在那儿**；
而噪点、树影、晾的衣服是随机的，不会在同一位置连续命中。
门槛从"高分"换成"高分 **或** 持续存在"，召回上去了，误报没上去。

轨迹 TTL 1200ms，超时清理。

### 8.4 第四层：滑动窗口投票 + 迟滞

```kotlin
windowSize   = 8       // 约 5fps 时 ≈ 1.6 秒
votesToEnter = 4       // 窗口内 4 帧命中才确认有人
votesToExit  = 12      // 连续 12 帧无目标才解除
enterScore   = 0.46f   // 进入阈值
exitScore    = 0.32f   // 退出阈值，比进入低 → 迟滞，边界不抖动
```

进入和退出用不同阈值（迟滞），是为了防止人在阈值附近来回时报警器疯狂开关。

### 8.5 三档灵敏度的实际差异

| | 灵敏 HIGH | 均衡 MEDIUM | 稳重 LOW |
|---|---|---|---|
| `enterScore` | 0.34 | 0.46 | 0.62 |
| `exitScore` | 0.24 | 0.32 | 0.45 |
| `votesToEnter` | 3 | 4 | 5 |
| `votesToExit` | 10 | 12 | 14 |
| `minAreaRatio` | 0.0012 | 0.0030 | 0.0100 |
| `smallTargetMinHits` | 2 | 3 | 5 |
| 瓦片 | 1 片 | 关 | 关 |

三档只是同一套算法的不同阈值组合，不存在"另一套逻辑"，现场调参时行为可预期。

---

## 9. 多边形选区

### 9.1 为什么需要

两个理由，都很实际：

1. **提召回**：把选区裁出来放大送检 ≈ 给画面加了一台长焦。一个 40px 的人在
   整帧里缩到 640 后剩 17px；裁掉选区后可能变成 90px，识别率差一个数量级。
2. **省算力**：只扫用户关心的那一块。实测选区模式 **7.5~8fps**，常规模式只有 4fps。

### 9.2 数据模型

归一化坐标（0~1），与分辨率解耦：

```kotlin
data class Roi(val points: List<PointF>, val label: String = "") {
    fun contains(x: Float, y: Float): Boolean   // 射线法 point-in-polygon
    fun areaRatio(): Float                      // 鞋带公式
    fun bounds(): RectF                         // 外接矩形
    fun encode(): String = points.joinToString(";") { "${it.x},${it.y}" }
    companion object {
        const val MIN_POINTS = 3
        const val MIN_VERTEX_GAP = 0.012f       // 顶点太近忽略（防手抖）
        const val MIN_AREA_RATIO = 0.0002f      // 太小没有意义
        fun decode(raw: String): Roi?
    }
}
```

持久化格式：`|` 分隔多边形，`;` 分隔顶点，`x,y` 为归一化坐标。
例：`0.36,0.27;0.65,0.28;0.64,0.54;0.37,0.53`

### 9.3 绘制交互

`OverlayView` 里处理触摸：

- 单击落一个顶点（距上一个顶点 < `MIN_VERTEX_GAP` 则忽略，防手抖重复点）
- 点回起点（距首点 < `CLOSE_GAP = 0.035f`）即闭合
- 闭合后立即生效并持久化
- `撤销` 删最后一个顶点 / 撤销草稿；`清空` 删所有选区

视觉：已确认选区青色 `#FF22D3EE` + 编号圆标；草稿黄色虚线 `#FFFFD60A` + 首点圈。

### 9.4 多个小选区（可以，已实现）

**可以。** 选区的存储是 `List<Roi>`，数量不限，每个独立判定、独立编号。
实际使用建议 **2~4 个**，原因不在算法而在预算：

- 每帧只扫 `roiPerFrame` 个选区（默认 2，省电模式 1），轮转游标推进；
- 选区越多，单个选区被扫到的间隔越长，确认有人需要的绝对时间越久
  （`votesToEnter` 是按"扫到几次"算的，不是按帧）；
- 全画面铺满小选区等于放弃性能优势，不如直接用整帧 + 瓦片。

轮转实现：

```kotlin
val take = min(roiPerFrame.coerceAtLeast(1), activeRois.size)
val start = roiCursor % activeRois.size
val selected = (0 until take).map { activeRois[(start + it) % activeRois.size] }
roiCursor = (start + take) % activeRois.size
```

### 9.5 放大识别流程

```
1. 取多边形外接矩形，外扩 PAD（0.06，最少 12px），保证人不会被裁掉半截
2. 裁切（最小边不低于 40px）
3. Bitmap.createBitmap → 直接送 detectRegion(crop, offsetX, offsetY)
4. 推理后把框的坐标平移回原图（offsetX/offsetY）
5. 过滤：框中心点必须落在多边形内，或与选区交集 ≥ 45%
```

第 5 步是必要的：外接矩形里必然包含多边形之外的区域，不过滤会把选区外的人也算进来。

### 9.6 选区模式的参数差异

```kotlin
val fusionCfg = cfg.copy(
    minAreaRatio = 0.0002f,      // 放大后目标占比天然变小，门限要放开
    smallTargetAreaRatio = 0f    // 选区里全是"小目标"，关掉小目标分支
)
```

---

## 10. 前台服务与息屏

检测必须能在熄屏后继续跑，所以逻辑放在 `DetectionService`（`foregroundServiceType="camera"`）。

- `WAKE_LOCK`：熄屏后保持 CPU 不休眠，推理和提示音才不会被挂起
- 通知常驻，带「暂停/继续」
- `onStartCommand` 返回 `START_STICKY`

**MIUI 上必须做的事**（否则熄屏几分钟后被杀）：

1. 设置里允许「自启动」
2. 电池优化设为「无限制」
3. 锁定后台（任务卡片下拉）

这些不是代码能解决的，README 里写了引导。

---

## 11. 提示音

`AlertPlayer` 自己合成 PCM 再播放，不依赖任何音频资源文件：

- 三档音色（叮咚 / 警报 / 提示），预设可切换
- 音量、时长可调
- 报警有 `cooldownMs = 3500` 冷却，不会连响
- 「试听」按钮直接播一遍

用 `AudioTrack` 而不是 `MediaPlayer`：合成一次 PCM 缓存起来反复播，
延迟低、不用处理文件 IO，也不用往 APK 里塞 wav。

---

## 12. 参数速查表

### 12.1 模型与推理

| 参数 | 值 | 位置 |
|---|---|---|
| 模型文件 | `yolov10n_int8.tflite` | `app/src/main/assets/` |
| 输入 | `640×640` FP32 | 模型自带 |
| 输出 | `[1, 300, 6]` END2END | 模型自带 |
| `MODEL_ASSET` | `yolov10n_int8.tflite` | `YoloDetector` |
| `NUM_THREADS` | 4 | `YoloDetector` |
| `DEFAULT_SCORE_THRESHOLD` | 0.25 | `YoloDetector` |
| `DETECTOR_PRE_FILTER` | 0.20 | `DetectionService` |
| `NORMALIZED_LIMIT` | 1.5 | `YoloDetector` |
| `MAX_CANDIDATES` | 16 | `YoloDetector` |
| `NMS_IOU` | 0.55 | `YoloDetector` |

### 12.2 相机

| 参数 | 值 |
|---|---|
| `ANALYSIS_WIDTH/HEIGHT` | 1920×1080（实测 1080×1440） |
| 背压策略 | `KEEP_ONLY_LATEST` |
| 变焦档位 | 1x → 2x → 3x → 5x → 8x |
| `POWER_SAVE_FRAME_INTERVAL` | 200ms（省电模式主动丢帧到 5fps） |

### 12.3 瓦片与选区

| 参数 | 值 |
|---|---|
| `TileGrid.DEFAULT` | `cols=3, rows=2, overlap=0.25` |
| 每帧瓦片数 | HIGH=1，其余 0 |
| `roiPerFrame` | 2（省电模式 1） |
| 选区外扩 | `REGION_PAD=0.06`，最小 12px |
| 裁切最小边 | 40px |
| 交集接受阈 | `ROI_OVERLAP_ACCEPT=0.45` |
| 顶点最小间距 | 0.012 |
| 闭合判定 | 距首点 < 0.035 |
| 最小面积占比 | 0.0002 |

### 12.4 融合

见 [8.5](#85-三档灵敏度的实际差异)。

---

## 13. 实测性能

设备：小米 M2007J1SC（骁龙 865 / Adreno 650），Android 13。
所有数字都是离屏探针和实机日志量出来的，不是估算。详细过程见 `PERFORMANCE.md`。

### 13.1 单帧推理

| 后端 | 中位 | 备注 |
|---|---|---|
| **CPU（XNNPACK，4 线程）** | **191ms** | 采用。最小 189 / 最大 193 |
| CPU（关 XNNPACK） | 836ms | 更慢 |
| NNAPI | 编译失败 | `ANEURALNETWORKS_BAD_DATA` |
| GPU（OpenCL） | 不可用 | 0 个 GPU 内核 |

### 13.2 端到端帧

| 模式 | 转换 | 推理 | 帧间隔 | 帧率 |
|---|---|---|---|---|
| 整帧（默认） | 25~28ms | 190~230ms | ~240ms | **≈4.2fps** |
| 选区 1 个 | — | 裁剪+推理 | — | **≈7.5fps** |
| 整帧 + 瓦片 1 片 | — | 2× 190ms | — | ≈2.5fps |
| 省电模式 | — | — | ≥200ms | 5fps 上限 |

### 13.3 CPU 占用

| 状态 | 进程 CPU |
|---|---|
| 检测 + 预览 | 260~330% |
| 仅预览 | 13~17% |
| 息屏 + 摘预览 | 检测仍跑，预览部分归零 |

> 上面「260~330%」是前台状态下的真实占用（多核并行生效）。
> 早期测量得到 100% 是因为探针掉进了后台 cpuset，见 [14.4](#144-探针前台陷阱非常重要)。

### 13.4 识别质量（离屏验收）

`probe_far_lake.jpg`（湖边长焦远景，1500×1125，目标 40~80px）：

| 路线 | 最高分 | 说明 |
|---|---|---|
| 整帧 | 0.748 / 0.690 / 0.347 | 3 个候选，位置准确 |
| 瓦片 3×2 | **0.815** / 0.756 / 0.573 | 明显提升 |

`probe_far_plaza.jpg`（黄昏车站广场，目标 20~40px，光线很暗）：

| 路线 | 最高分 | 说明 |
|---|---|---|
| 整帧 | 0 个候选 | 目标太小 |
| 瓦片 3×2 | 0.282 | 1 个 |
| 瓦片 6×4 | 0.288 / 0.261 | 2 个 |

裁切放大后人工核对：两个候选**都是骑摩托车的人，定位准确**。分数低是因为
「骑车的行人在 25px 尺度上」本身就是模棱两可的样本，不是误报。

按「均衡」档 `enterScore=0.46`，这两个候选不会触发报警 —— 这是正确的保守行为。
这张图已经接近物理极限（黄昏 + 25px + 逆光），不要拿它当调参目标；
真实场景里人占屏 100px 以上时稳定在 0.7~0.85。

### 13.5 包体积

| 构成 | 大小 |
|---|---|
| `classes.dex`（debug，未混淆） | 10.5 MB |
| `libtensorflowlite_jni.so`（arm64） | 3.8 MB |
| 模型 | 3.0 MB |
| 两张验收样图 | 0.4 MB |
| 其他资源 | ~1.3 MB |
| **release APK** | **13 MB** |

v1 是 56 MB（ML Kit Pose 的模型和原生库占大头）。

---

## 14. 构建与调试

### 14.1 日志

```bash
adb logcat -s YoloDetector:* DetectionService:* FrameAnalyzer:* MainActivity:*
```

**分析器诊断行**（每次推理一行）：

```
cv=26ms det=214ms inf=214ms src=1080x1440 roi=0(z0) tiles=0 coords=norm raw=[] -> n=0 fs=0.00 ss=0.00 v=0 none
```

| 字段 | 含义 |
|---|---|
| `cv` | ImageProxy → Bitmap 的转换耗时 |
| `det` / `inf` | 本帧推理耗时（`det` 含解码） |
| `src` | 分析流分辨率 |
| `roi=N(zM)` | 有 N 个选区，本帧扫了 M 个 |
| `tiles` | 本帧扫了几片瓦片 |
| `coords` | `norm` = 模型输出归一化坐标；`px` = 像素坐标 |
| `raw` | 模型原始候选（分数+框），**未经过滤** |
| `n` | 融合后通过的候选数 |
| `fs` | 平滑后的分数 |
| `ss` | 状态机分数 |
| `v` | 投票计数 |
| `none/PRESENT` | 最终判定 |

**调参时看 `raw` 和 `n` 的差**：`raw` 有、`n` 没有 → 被门限挡住了，去看是面积还是形状；
`raw` 本身就空 → 模型没看见，这时候调阈值没用，要换路线（瓦片/选区）或调变焦。

### 14.2 常用 adb

```bash
# 截图（-p 参数在新版上会报 usage 错误，用 exec-out）
adb exec-out screencap -p > shot.png

# 模拟点击（画选区用）
adb shell input tap X Y

# 看进程 CPU
adb shell top -b -n 1 | grep presenceguard

# 看权限
adb shell dumpsys package com.presenceguard.yolo | grep -A20 "runtime permissions"

# 清数据重来
adb shell pm clear com.presenceguard.yolo
```

### 14.3 Git Bash 路径转换坑

`adb shell cat /sdcard/x` 在 Git Bash 里会被 MSYS 改写成 Windows 路径。
加 `MSYS_NO_PATHCONV=1` 并用 `//sdcard/x`：

```bash
MSYS_NO_PATHCONV=1 adb shell cat //sdcard/ui.xml
```

### 14.4 探针前台陷阱（非常重要）

**这是本项目最大的一个测量陷阱，v2 在这里浪费了很多时间。**

探针最初的设计是「在 `onCreate` 里起个线程跑完就 `finish()`」。结果量出来的
单帧耗时是 **705ms**，而且线程数从 1 调到 8 完全不变（1036 / 1016 / 1047 / 1032 / 1032ms）。

原因：`finish()` 之后进程立刻从前台掉到后台，Android 把它的 cpuset 限制成 `0-3`
**四个小核**：

```bash
adb shell grep Cpus_allowed_list /proc/$(pidof com.presenceguard.yolo)/status
# 后台： Cpus_allowed_list: 0-3
# 前台： Cpus_allowed_list: 0-7
```

大核一个都不给，自然怎么调线程数都没用，进程 CPU 稳定在 100%（正好一个核）。

修正后同样的模型、同样的参数：**191ms**，快 3.7 倍。

修正需要三件事，缺一不可：

1. 探针**不能**在 `onCreate` 里 `finish()`
2. 必须有真实的内容视图（否则系统判定没完成绘制，直接把前台让给桌面）
3. Activity 主题不能是 `@android:style/Theme.NoDisplay`
   （那个主题要求必须在 `onCreate` 里 `finish`，否则抛
   `Activity did not call finish() prior to onResume() completing`）

而且收尾的 `finish()` 必须延后，否则会在 `onResume` 还没走完时执行，抛同样的异常：

```kotlin
runOnUiThread { Handler(mainLooper).postDelayed({ finish() }, 800) }
```

探针还会把 `cpuset=` 打进日志，方便随时确认这次测量的可信度。

### 14.5 测量前必查

```bash
adb shell cat /sys/devices/system/cpu/online          # 8 核都在？
adb shell cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq   # 大核降频了？
adb shell dumpsys battery | grep -i "AC powered\|level"               # 没在省电模式？
adb shell settings get global low_power                                # 0
```

---

## 15. 已知限制

1. **单帧上限 4.2fps**。YOLOv10n @640 在 865 上就是这个水平。要更快只能换更小的
   输入尺寸的模型（需要重新导出），或者用选区模式把预算集中在关心的区域。

2. **没有 GPU/NPU 加速**。这个混合量化模型在 TFLite 2.14 的 GPU 委托上 0 内核，
   NNAPI 编译失败。想要硬件加速必须换一个真正全量化且图干净的模型。

3. **黄昏/逆光下的小目标召回有限**。见 13.4，广场图 25px 目标只有 0.26~0.29。
   这不是参数问题，是模型能力边界。

4. **多镜头依赖机型暴露**。多数机型只给 1 颗后置相机，此时焦段按钮隐藏。

5. **前后双摄并发依赖相机 HAL**。本机（M2007J1SC / 865）不支持，自动降级为
   单镜头巡检，界面显示「巡检 单」。见第 19.3 节。

6. **双指缩放不增加像素**。数字裁切到 6x 时送进模型的是一小块放大图，
   和真的长焦不是一回事。要真实增益只能靠「焦段」按钮走光学路径。

7. **MIUI 后台限制需要用户手动放行**。见第 10 章。

8. **选区越多，单个选区的判定越慢**。见 9.4。

---

## 16. 双指缩放：数字裁切，不是换镜头

### 16.1 两条完全不同的"拉近"路径

这是整个应用里最容易被误解的一处，所以代码里刻意做成两条互不相干的路：

| | 焦段按钮（`btnZoom`） | 双指缩放（`OverlayView`） |
|---|---|---|
| 实现 | `CameraControl.setZoomRatio()` | `YuvCropper.crop()` 从帧里挖中间一块 |
| 作用层 | 传感器 / ISP | 送进模型之前的软件裁切 |
| 预览画面 | **跟着放大** | 不动（只画一个虚线框） |
| 像素质量 | 真实像素（部分机型还会切长焦） | 像素不增加，放大到极限就是放大模糊 |
| 检测收益 | 真增益 | 只提高信噪比（背景少了、目标占比大了） |
| 上限 | 机型决定（本机 10x） | 6x（`MAX_CROP_SCALE`） |

之所以两个都要：物理焦段受机型限制（很多机器的 `availableCameraInfos` 只报一颗后置），
数字裁切永远可用；而物理焦段在支持的机型上是实打实的画质提升。功能上重叠，收益上不重叠。

### 16.2 为什么裁切要放在 YUV 层

最直觉的写法是 `imageProxy.toBitmap()` 之后 `Bitmap.createBitmap(src, left, top, w, h)`。
两个问题：

- `toBitmap()` 会把**整帧** 1080×1440 解成 ARGB_8888，约 6MB。裁不裁都一样付这个钱。
- 再 `createBitmap` 一次，又拷贝一遍。

`YuvCropper` 直接按目标矩形去读 `Plane` 的 buffer，逐像素做 BT.601 limited-range
YUV→ARGB，**矩形外的像素一个都不碰**：

```kotlin
object YuvCropper {
    fun crop(proxy: ImageProxy, rect: Rect): Bitmap?
}
```

实测（1080×1440 源帧，见 `cv=` 字段）：

| 取景范围 | 转换耗时 |
|---|---|
| 整帧 | 37 ~ 41 ms |
| 中心 1/3 × 1/3（crop 3.0x） | **17 ms** |
| 一个选区（约 15% 面积） | **7 ms** |

### 16.3 旋转映射是这里唯一的坑

`ImageProxy` 的 `width/height` 是**传感器坐标系**，而模型和界面要的是**正立坐标系**。
两者差一个 `imageInfo.rotationDegrees`。`YuvCropper.crop` 收到的 `Rect` 是正立坐标，
必须自己映射回传感器坐标：

| rotation | 正立 (ox, oy) → 传感器 (x, y) |
|---|---|
| 0 | `(ox, oy)` |
| 90 | `(oy, H-1-ox)` |
| 180 | `(W-1-ox, H-1-oy)` |
| 270 | `(W-1-oy, ox)` |

写错一个符号不会崩、不会报错，只会让检测框整体偏移或镜像 —— 而且在小目标上看起来
"好像还能用"，特别难查。

### 16.4 手势的两个细节

```kotlin
private val scaleDetector = ScaleGestureDetector(context, listener).apply {
    isQuickScaleEnabled = false   // 双击快速缩放太容易误触
}
```

- **倍乘而非累加**：`next = base * detector.scaleFactor`，`base` 在 `onScaleBegin` 取一次。
  累加会让连续捏合越捏越迟钝。
- **松手吸附**：`onScaleEnd` 时 `cropScale < 1.12` 直接归 1。捏合永远回不到精确的 1.00，
  差 0.05 就等于"还在裁"，用户会以为自己复位了却一直少看一圈画面。

---

## 17. 选区外零计算

需求是"非选区内不做计算、不做识别、不产生无效性能开销"。这句话在实现上有个陷阱：
**只要还走 `imageProxy.toBitmap()`，选区内外的开销就是一样的**，因为整帧转换已经发生了。

所以选区路线（`FrameAnalyzer.scanRois`）从根上就没有整帧位图：

```kotlin
for (roi in budgetedRois) {
    val region = roi.boundsIn(frameWidth, frameHeight)   // 选区的外接矩形
    val pieces = if (roi.areaRatio() > LARGE_ROI_AREA_RATIO) splitQuadrants(region) else listOf(region)
    val piece = pieces[roiSubCursor++ % pieces.size]
    val crop = YuvCropper.crop(imageProxy, piece) ?: fallbackCrop(...)
    for (c in engine.detectRegion(crop, piece.left, piece.top)) {
        remap(c, roi, frameWidth, frameHeight)?.let { out += it }
    }
}
```

三个要点：

1. **只转换外接矩形**。多边形内部的孔洞和外接矩形外的区域都不解码。
2. **大选区切 2×2 轮转**（面积占比 > 6%）。一个占半屏的选区直接缩到 640 送检，
   远处的人会被压成十几个像素；切成四块轮流送，每块的有效分辨率翻一倍。
   `QUADRANT_PAD = 0.12` 是外扩量，防止目标正好跨在接缝上被切成两半。
3. **`remap()` 做回填**。模型吃的是裁出来的图，输出坐标是相对裁切块的；
   要按 `piece` 的偏移和选区的放大倍率映射回整帧归一化坐标，否则融合层算面积占比会全错。

实测：转换从 41ms 降到 7ms;推理不变（模型输入恒为 640）。

> 所以"选区省性能"这句话要说得准确一点：省的是**转换和无效区域的计算**，
> 推理成本是固定的（选区 1 个 = 1 次推理，和整帧路线一模一样）。
> 选区的真正价值是把同样的推理预算**集中**到关心的区域 —— 等效焦距翻几倍，
> 而不是让单帧变快。要多选区才谈得上省：`roiPerFrame` 限制每帧只扫 N 个，靠游标轮转覆盖。

---

## 18. 瓦片帧差跳过

瓦片路线（灵敏档才开）每扫一片就是一次完整推理，约 320~330ms。监控场景里画面
绝大多数时间是**静止**的 —— 一个空走廊，两秒之间像素几乎不变，再推一遍纯属浪费。

```kotlin
private fun shouldScanTile(source: Bitmap, tile: Rect, index: Int): Boolean {
    val current = fingerprint(source, tile)          // 16×16 灰度采样 = 256 字节
    val previous = tilePrints[index] ?: return true.also {
        tilePrints[index] = current; tileSkips[index] = 0
    }
    val skips = tileSkips[index] ?: 0
    // 连着跳过太多次就强制重扫一遍：防止"慢慢走进画面"的人被积分掉
    if (skips >= FORCE_RESCAN_AFTER) {
        tilePrints[index] = current; tileSkips[index] = 0; return true
    }
    if (meanAbsDiff(previous, current) < MIN_TILE_MAD) {
        tileSkips[index] = skips + 1
        return false
    }
    tilePrints[index] = current; tileSkips[index] = 0
    return true
}
```

参数：`FP_SIZE = 16`（16×16 灰度）、`MIN_TILE_MAD = 6`（0~255 尺度上的平均绝对差）、
`FORCE_RESCAN_AFTER = 8`（连续跳过 8 次强制重扫）。

**为什么要 `FORCE_RESCAN_AFTER`**：如果一个人以极慢的速度走进画面，每帧的差异
可能都低于 6，理论上会被无限跳过。强制重扫把最坏情况压到"最多漏 8 帧"，
在 2~4fps 下大约是 2~4 秒 —— 配合融合层的滑动窗口投票，不会造成真正的漏报。

实测（灵敏档，瓦片数 1）：

| 帧 | det 总耗时 | 说明 |
|---|---|---|
| 首帧 / 画面变化 | 647 ms | 整帧 336 + 瓦片 311 |
| 画面静止 | 336 ms | 瓦片被跳过，只剩整帧那一次推理 |

日志字段：`tiles=1 skip=1` 表示这一帧有 1 片瓦片、其中 1 片因为帧差过小被跳过。

---

## 19. 2 秒全图巡检与前后双摄

### 19.1 为什么需要它

选区和数字裁切都是"望远镜"：看得清，但看不见旁边。有人绕到取景框外、或者站在
选区之外，主判定是完全瞎的。所以要有一条**旁路**，定期退回到最广的视野扫一眼。

### 19.2 三条约束

1. **不切镜头**。切物理镜头要 `unbind/rebind`，实测一次约 100ms 且预览会黑一下；
   每 2 秒闪一次没法用。用"当前镜头 + 无视选区 + 无视数字裁切"的整帧就够了。
2. **不投票**。巡检结果进独立的 `inspectionFusion`，绝不写回主状态机。
   把巡检帧当成一帧"没看见人"喂给主融合，会把滑动窗口的票数反复拉低，
   选区内的人反而更容易判丢。
3. **不 publish**。巡检帧不触发 `onFrame`，界面上的主判定状态不受影响。
   结论单独走 `Listener.onInspection`。

### 19.3 前后双摄并发：期望与现实的差距

需求是"前置和后置同时取流、同时运算，避免人走到屏幕背面被漏掉"。CameraX 在
**1.4.0** 才提供 `ConcurrentCamera` 这套 API（1.3.x 完全没有，这也是本项目把
`cameraxVersion` 从 1.3.4 升到 1.4.1 的唯一原因）：

```kotlin
val concurrent = provider.bindToLifecycle(
    listOf(
        SingleCameraConfig(primarySelector, primaryGroup, this),
        SingleCameraConfig(oppositeSelector, secondaryGroup, this)
    )
)
```

但底层要求相机 HAL 声明支持并发流组合。实测本机（小米 M2007J1SC / 骁龙 865）：

```
W/DetectionService: concurrent bind unsupported, degrade to single camera:
    Concurrent camera is not supported on the device.
```

所以流程是「**试 → 失败就降级**」，而不是"先探测再决定"——探测一次同样要开相机，
成本比试一次还高。降级后的行为：

- 主摄巡检照常（2 秒一次整帧粗判，当前镜头最广视野）；
- 副摄那条输出流不创建；
- 界面按钮从「巡检 双」变成「巡检 单」，明确告诉用户现在是单镜头。

副摄一旦绑上，它用的是自己的 `FrameAnalyzer` 实例 + 独立融合引擎，并且打开
`inspectionOnly` 开关：只有到点那一帧做整帧粗判，其余帧直接丢弃（连位图都不转）。

```kotlin
@Volatile var inspectionOnly: Boolean = false
```

### 19.4 为什么共用同一个解释器

副摄和主摄共用同一个 `YoloDetector`。不是图省事，是必须：

- TFLite 的 `Interpreter` **不是线程安全的**，两个线程同时 `run()` 会输出错乱甚至崩；
- 开两份解释器会让内存翻倍（模型 + 激活值），865 上不划算。

两者都挂在同一个单线程 `cameraExecutor` 上，推理天然串行。"同时获取"在取流层是
真并发，"同时运算"在推理层是背靠背排队。对 2 秒一次的巡检，这个差别可以忽略。

### 19.5 巡检命中就响

```kotlin
private fun onInspectionResult(present: Boolean, score: Float, source: String) {
    notifyListeners { it.onInspection(present, score, source) }
    if (!present || paused) return
    if (!settings.audioEnabled) return
    val now = SystemClock.elapsedRealtime()
    if (now - lastAlertAt < settings.alertIntervalMs) return
    lastAlertAt = now
    alertPlayer.play()
}
```

如果只在"主判定已经报警"时才响，这个功能等于什么也没补上。它的全部意义就是
在**主判定看不见的地方**发现人并报警。和主判定的报警共用 `lastAlertAt`，
不会出现两条路各响一次的重叠。

### 19.6 省电模式的折中

`dualEyeEnabled() = dualInspect && !powerSave`。多开一条相机输出流是这个功能里
最贵的一笔，省电时摘掉；但**主摄巡检保留** —— 2 秒一次的整帧粗判只占
约 300ms/2000ms ≈ 15% 的推理预算，作为"不漏人"的底线值得留。

---

## 20. 界面布局：为什么把指标行塞进底栏容器

原来的写法是指标 `TextView` 用 `layout_gravity="bottom|start"` + `marginBottom="140dp"`
手动避让控制栏。控制栏加到第三行（焦段/选区/清空）之后，三行按钮撑到约 154dp，
直接把帧率文字压在按钮底下 —— 手算出来的避让量，按钮一多就失效。

改成结构化布局，让布局系统自己去算：

```xml
<LinearLayout android:id="@+id/bottomDock"
    android:layout_gravity="bottom"
    android:orientation="vertical"
    android:paddingHorizontal="12dp"
    android:paddingBottom="16dp">

    <TextView android:id="@+id/metricsText" ... />   <!-- 自然排在按钮上方 -->
    <LinearLayout android:id="@+id/controlBar" ... />
</LinearLayout>
```

指标行 `layout_width="match_parent"` + `maxLines="4"`，诊断串再长也在屏内换行，
不会撑出屏幕。

**另一个连带的坑**：`OverlayView` 要实现双指缩放，非编辑模式下必须吃掉全部触摸
（`onTouchEvent` 一律返回 true），否则捏合手势会漏给下层。这导致"暗屏时点一下恢复亮度"
原来挂在 `binding.root.setOnTouchListener` 上的逻辑**永远收不到事件**。修法是把拦截
提前到分发之前：

```kotlin
override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
    if (dimmed) { setDimmed(false); return true }
    return super.dispatchTouchEvent(ev)
}
```

---

## 21. 版本演进

| 版本 | 主题 | 关键结论 |
|---|---|---|
| v1 | 双引擎互补 | 双引擎印证能压误报，但 320 输入撑不起小目标；Task 库不接受量化模型 |
| v2 | 单路 YOLO | 640 输入 + 自写 Interpreter 层；小目标召回靠瓦片/选区放大 + 时间证据 |

如果还有 v3，方向大概会是：

- 换一个**图干净的全量化模型**（或者自己从 ONNX 转），把 GPU/NPU 打通；
- 或者把 `yolov10n` 裁成**只有 person 一个类**再量化，"检测到人"这件事本身
  比"检测到 80 类"便宜得多 —— 但这需要训练环境。

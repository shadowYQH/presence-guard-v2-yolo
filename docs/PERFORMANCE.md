# 性能实测与踩坑记录

设备：小米 M2007J1SC（骁龙 865 / Adreno 650），Android 13，分析流 1080×1440。
所有数字都是离屏探针和实机日志量出来的。本文记录**过程**，包括走过的弯路 ——
因为弯路本身比结论更值钱。

---

## 0. 结论速览

| 项目 | 结果 |
|---|---|
| 单帧推理（CPU / XNNPACK / 4 线程） | **191ms** |
| 端到端帧（整帧模式） | ~240ms ≈ **4.2fps** |
| 端到端帧（选区模式） | ~133ms ≈ **7.5fps** |
| APK（release） | **13MB** |
| GPU 委托 | 不可用（0 个内核） |
| NNAPI | 不可用（编译失败） |

---

## 1. 三次测量事故

### 1.1 事故一：用了坏模型，还以为是参数问题

从 HF 镜像 `anan19990108/yolov10n-tflite` 下了 `yolov10n_integer_quant.tflite`，
跑出来**一个候选都没有**。第一反应是解码逻辑写错了 —— 于是加了输出张量转储。

诊断（`--ez diag true`）打出来：

```
outputs=1 [0]=[1, 300, 6]FLOAT32(s=0.0,z=0)
  flat[0..35]:
   r0: 0.931 0.621 0.931 0.621 0.621 0.000
   r1: 0.931 0.621 0.931 0.621 0.621 0.000
   r2: 0.931 0.621 0.931 0.621 0.310 0.000
   r3: 0.931 0.621 0.931 0.621 0.310 0.000
   r4: 0.931 0.621 0.931 0.621 0.000 1.000
   r5: 0.931 0.621 0.931 0.621 0.000 2.000
```

三个特征一眼就能判死刑：

1. **300 行的 4 个坐标列全是同样两个常数**（0.931 / 0.621），真实检测器不可能这样；
2. **类 id 随行号递增** 1, 2, 3, 4, 5 —— 某个 `arange` 被常量折叠了；
3. 形状确实是 `[1,300,6]`（把输出表当 `[1,6,300]` 喂回去，TFLite 直接抛
   `IllegalArgumentException`，说明形状没读错）。

换 `full_integer_quant` 版本更离谱：输出张量 `scale=0.3104 / zero=-128`，
意味着标定出来的动态范围是 `[-39.7, 39.7]`，而分数应该在 0~1 —— 标定彻底失败。
反量化后输出只有 `0 / 0.310 / 0.621 / 0.931` 四个值。

**根因**：YOLOv10 的端到端头里有 top-k / argsort，TFLite 转换时这一步没接上。
两个文件是同一个坏图的不同量化版本，换参数救不回来。

**结论**：只有 `yolov10n_int8.tflite`（混合量化）是好的。
湖景图 0.748 / 0.690 / 0.347，坐标正确，`coords=norm`。

**教训**：模型不动的时候，先验证模型没坏，再怀疑自己的代码。
判断方法很简单 —— 看输出张量的**结构性特征**：正常的 YOLO 输出是 300 行各不相同的框、
递减的分数；如果所有行长得一样，模型就是坏的。

---

### 1.2 事故二：量化钳位用错了 dtype，图像被"烧"掉

`full_integer_quant` 模型的输入是 INT8（`scale=1/255`，`zero=-128`），
但代码里量化写入一律 `coerceIn(-128, 127)`。

如果输入是 **UINT8**（`zeroPoint=0`，值域 0~255），`coerceIn(-128,127)` 会把
所有亮度 >50% 的像素压成同一个值 —— 送进网络的是一张烧掉的图。

更阴险的是：**它不报错，只是安静地给出低分**。而且当时 `inputDenominator`
只对 FP32 路径生效，量化路径写死了 `/255`，所以调参数完全没反应、
两种归一化方式输出一模一样 —— 这个"没反应"本身就是线索。

**修正**：

```kotlin
private fun quantize(value: Float, invScale: Float): Byte {
    val raw = value * invScale + inputZero
    return if (inputType == DataType.UINT8) {
        raw.roundToInt().coerceIn(0, 255).toByte()
    } else {
        raw.roundToInt().coerceIn(-128, 127).toByte()
    }
}
```

反量化同理：INT8 有符号直接用，UINT8 要先 `and 0xFF`。弄反会让零点偏移 256 个量化级。

---

### 1.3 事故三：探针掉进后台 cpuset —— 705ms 完全是假的

这个是最大的坑，浪费的时间最多。

现象：探针量出单帧 **705ms**，而且线程数 1/2/4/6/8 全部一样：

```
threads=1  整帧 1036ms
threads=2  整帧 1016ms
threads=4  整帧 1047ms
threads=6  整帧 1032ms
threads=8  整帧 1032ms
```

耗时对线程数完全不敏感，说明根本没并行。于是去量进程 CPU：

```
  PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM
 9370 u0_a286   20   0   6.2G   183M    78M S   103   1.5   com.presenceguard.yolo
```

**稳定 100~107%，正好一个核**。而设备侧一切正常：

```
在线核：0-7
cpu4 频率：2246400 kHz（大核满频）
AC powered: true, level: 100
low_power: 0
```

真正的线索在这里：

```bash
$ adb shell grep Cpus_allowed_list /proc/$(pidof com.presenceguard.yolo)/status
Cpus_allowed_list:	0-3
```

**进程被限制在 0-3 四个 A55 小核上，大核一个都没给。**

原因是探针在 `onCreate` 里起了线程就 `finish()` —— Activity 一消失，
进程立刻从 `top-app` cpuset 掉到 `background` cpuset，只留小核。

修正需要三件事，缺一不可：

1. **不能**在 `onCreate` 里 `finish()`，要等探针跑完再收
2. 必须有**真实的内容视图**（`setContentView`），否则系统判定这个 Activity 没完成绘制，
   直接把前台让给桌面（日志里是 `Activity top resumed state loss timeout`）
3. 主题**不能**是 `@android:style/Theme.NoDisplay` —— 那个主题要求必须在 `onCreate`
   里 `finish`，否则抛 `Activity did not call finish() prior to onResume() completing`

而且收尾的 `finish()` 还得延后，不然会在 `onResume` 走完之前执行，抛同一个异常：

```kotlin
runOnUiThread { Handler(mainLooper).postDelayed({ finish() }, 800) }
```

修正后的日志第一行现在会打 cpuset，随时可以确认这次测量可不可信：

```
I Probe: == probe_far_lake.jpg 1500x1125 accel=CPU(XNNPACK) threads=4 ==
I Probe: cpuset=0-7
I Probe: 整帧: 3 个候选, 358ms, coords=true, class=0
I Probe: 基准: 中位 191ms, 最小 189ms, 最大 193ms
```

**705ms → 191ms，快 3.7 倍。同样的模型，同样的代码，只是测量方式对了。**

**教训**：在任何安卓性能测量里，先确认 `Cpus_allowed_list`。
"耗时对线程数不敏感"是进程被限核的典型症状。

---

## 2. 加速器：为什么两条路都走不通

### 2.1 GPU 委托（tensorflow-lite-gpu 2.14）

```kotlin
GpuDelegate(GpuDelegateFactory.Options().apply {
    setPrecisionLossAllowed(true)
    setQuantizedModelsAllowed(true)
})
```

日志：

```
I tflite: Created TensorFlow Lite delegate for GPU.
I tflite: Replacing 13 out of 352 node(s) with delegate (TfLiteGpuDelegateV2) node, yielding 3 partitions
I tflite: Created 0 GPU delegate kernels.
E YoloDetector: 模型加载失败
    java.lang.IllegalArgumentException: Internal error: Failed to apply delegate:
    ADD: Max version supported: 2. Requested version 4.
    CAST: Not supported Cast case. Input type: BOOL and output type: INT64
    CAST: Not supported Cast case. Input type: FLOAT32 and output type: INT64
    CAST: Not supported Cast case. Input type: INT64 and output type: FLOAT32
    CONCATENATION: OP is supported, but tensor type/shape isn't compatible
```

三个问题：

- 2.14 的 GPU 委托对 **int8 张量**基本没有支持（`setQuantizedModelsAllowed(true)` 也救不了）
- YOLOv10 端到端头里的 `CAST → INT64`（top-k 的索引）不支持
- `ADD` 版本 4 不支持

结果 `Created 0 GPU delegate kernels` —— 一个内核都上不了，构造直接抛异常。

**因此 GPU 依赖已从项目中移除。** 为一个跑不起来的后端背 13MB 原生库没有意义。
代码里保留了回退链结构，将来换一个图干净的全量化模型可以直接接回来。

### 2.2 NNAPI

```
W tflite: NNAPI SL driver did not implement SL_ANeuralNetworksDiagnostic_registerCallbacks!  (×N)
I tflite: Replacing 19 out of 352 node(s) with delegate (TfLiteNnapiDelegate) node, yielding 7 partitions
W YoloDetector: 后端 NNAPI 不可用，退到下一个:
    Internal error: Failed to apply delegate: NN API returned error
    ANEURALNETWORKS_BAD_DATA at line 4735 while completing NNAPI compilation.
```

只委派了 19/352 节点，然后编译阶段直接 `BAD_DATA`。
同样是混合量化模型的问题 —— QTI 驱动不接受这种张量组合。

### 2.3 关掉 XNNPACK 用原生内核

想法是：XNNPACK 把图切成 131 个分区，失去了跨层融合；原生内核也许能整图优化。

```
accel=CPU(原生内核, 无 XNNPACK)
整帧: 4 个候选, 1256ms
基准: 中位 836ms
```

**836ms，比 XNNPACK 的 705ms（当时还没修正 cpuset）更慢。**
说明 XNNPACK 确实是 CPU 上最优路径，"分区多所以慢"这个假设不成立。

---

## 3. 最终耗时构成

修正 cpuset 后，整帧模式的端到端拆解：

| 阶段 | 耗时 |
|---|---|
| ImageProxy → Bitmap（旋转 + 转换） | 25~28ms |
| letterbox 预处理（Canvas + getPixels） | ~10ms |
| 推理 | 190ms |
| 解码 + 融合 | < 5ms |
| **合计** | **~240ms** |

推理占 80%，所以优化重点只能是模型本身，不是周边代码。

### 3.1 线程数（修正 cpuset 后）

`setNumThreads(4)` 是默认值，也是 4 个大核的合理选择。
（修正 cpuset 之前的"线程数无效"结论作废 —— 那是被限核造成的。）

### 3.2 各条路线的实测开销（本机 M2007J1SC，1080×1440 源帧）

下表的 `cv` / `det` / `inf` 都直接来自 `FrameAnalyzer` 的诊断日志，
可以在真机上用 `adb logcat -s FrameAnalyzer` 复现。

| 路线 | 转换 `cv` | 整段 `det` | 推理 `inf` | 说明 |
|---|---|---|---|---|
| 整帧（基线） | 37~41 ms | 348~410 ms | 310~360 ms | 每帧一次 640 推理 |
| 数字裁切 3.0x | **17 ms** | 306 ms | 285 ms | 只解中心 1/9，转换省一半 |
| 选区 ×1（约 15% 面积） | **7 ms** | 315 ms | 308 ms | 选区外一个像素都不解码 |
| 瓦片扫到（灵敏档） | 19 ms | **647 ms** | 317 ms | 多一次完整推理 ≈ 320 ms |
| 瓦片帧差跳过 | 19 ms | 336 ms | — | 画面静止，瓦片不送模型 |
| 巡检帧（每 2 s） | ~41 ms | — | 285~317 ms | 独立融合，不 publish |

三个结论：

1. **转换成本可以被压到接近零**（41 → 7 ms），但**推理成本是刚性的**。
   模型输入恒为 640，喂多大的图都跑一次。所以"选区省性能"省的是转换和无效区域，
   不是推理。真正省推理的只有两条路：帧差跳过、降帧率。
2. **瓦片很贵**（+320 ms/片）。灵敏档每帧滚一片，帧率直接腰斩。
   只有确认场景里目标的像素尺度很小时才值得开。
3. **帧差跳过是这个项目里性价比最高的优化**：静态画面下省掉整片瓦片的推理，
   `det` 从 647 ms 回到 336 ms，几乎没有实现成本（16×16 灰度指纹 + 平均绝对差）。

### 3.3 探针 190 ms vs 实机 300 ms：这个差距是真实的

同一个模型、同一台机器、同样 4 线程：

| 环境 | 推理耗时 |
|---|---|
| `ProbeActivity` 探针（纯推理循环） | 190 ms |
| 正式 APK（前台服务 + 预览 + UI + 巡检） | 285~317 ms |

差 50~65%。原因不是模型也不是参数，是**同进程里的其他活**：相机预览的
GPU 合成、`OverlayView` 的 Canvas 重绘、服务通知刷新，都会和推理线程抢 CPU。
`Cpus_allowed_list` 是对的（0-7，`/top-app`），进程 CPU 约 106% —— 也就是说
XNNPACK 的 4 个线程并没有真的并行跑满 4 核，大部分算子还是单线程串行。

**教训**：探针测出来的数字只能用来做**版本间相对比较**（换模型、换后端、换线程数），
不能当作线上性能。要评估真实体验，必须在完整应用里量。

---

## 4. 识别质量对照

### 4.1 湖景图 `probe_far_lake.jpg`（1500×1125，目标 40~80px）

| 路线 | 候选数 | 分数 |
|---|---|---|
| 整帧 | 3 | 0.748 / 0.690 / 0.347 |
| 瓦片 3×2 | 3 | **0.815** / 0.756 / 0.573 |

瓦片把最高分从 0.748 拉到 0.815，符合"缩小视野 = 放大目标"的预期。

### 4.2 广场图 `probe_far_plaza.jpg`（黄昏，目标 20~40px）

| 路线 | 候选数 | 分数 |
|---|---|---|
| 整帧 | 0 | — |
| 瓦片 3×2 | 1 | 0.282 |
| 瓦片 6×4 | 2 | 0.288 / 0.261 |

把候选框裁出来放大 6 倍人工核对，两个都是**骑摩托车的人**，定位准确。
分数低不是误报，是"骑车的行人在 25px 尺度上"这个样本本身模棱两可。

按「均衡」档 `enterScore=0.46`，这两个候选不会触发报警 —— 正确的保守行为。

**这张图接近物理极限**（黄昏 + 逆光 + 25px）。v1 用 EfficientDet 在这张图上到过
0.41~0.43，v2 是 0.26~0.29，但两者都远低于可靠阈值，都不构成"能识别"。
真实场景里人占屏 100px 以上时，v2 稳定在 0.7~0.85。

不要拿这张图当调参目标 —— 把它当"模型能力边界"的标尺。

---

## 5. 功耗

| 状态 | 进程 CPU |
|---|---|
| 检测 + 预览（前台） | 260~330% |
| 仅预览 | 13~17% |
| 熄屏 + 摘预览 | 检测仍在跑，预览部分归零 |

**预览占了 13~17% 单核**，熄屏后必须摘掉 —— 而且只 `setSurfaceProvider(null)` 不够，
要把 Preview 从 `bindToLifecycle` 的用例列表里去掉。详见 `DEVELOPMENT.md` 5.3。

这件事在 v1 上量过完整的三段对比：检测+预览 106~130% → 仅预览 13~17% → 熄屏+暂停 **0.0%**。
v2 沿用同样的机制。

---

## 6. 包体积

```
release APK                              13 MB
├── classes.dex（未混淆）                10.5 MB
├── lib/arm64-v8a/libtensorflowlite_jni.so  3.8 MB
├── assets/yolov10n_int8.tflite           3.0 MB
├── assets/probe_far_*.jpg                0.4 MB
└── resources.arsc + 其他                ~1.3 MB
```

对比 v1 的 56MB：省下来的全是 ML Kit Pose 的模型和原生库。

注意：**增量构建会残留旧的原生库**。加了又删 GPU 依赖之后 APK 从 14MB 涨到 24MB，
`gradle clean` 之后才回到 14MB。报体积前一定先 clean。

---

## 7. 测量清单（照做就不会再被骗）

```bash
# 1. 前台 cpuset（探针自己会打，但也可以手动确认）
adb shell grep Cpus_allowed_list /proc/$(pidof com.presenceguard.yolo)/status
#    期望 0-7；如果是 0-3 说明进程在后台，数字全部作废

# 2. 设备状态
adb shell cat /sys/devices/system/cpu/online
adb shell cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq
adb shell settings get global low_power          # 0
adb shell dumpsys battery | grep -i "AC powered"

# 3. 模型没坏（看输出张量的结构性特征）
adb shell am start -n com.presenceguard.yolo/.ProbeActivity \
    --es image probe_far_lake.jpg --ez diag true

# 4. 体积一定要 clean 之后再量
gradle clean :app:assembleRelease
```

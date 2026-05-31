#import "@local/bubble-sysu:0.1.0": *

#show: report.with(
  title: "实验六：图文多模态问答",
  subtitle: "大数据原理与技术实验报告",
  student: (name: "元朗曦", id: "23336294"),
  school: "计算机学院",
  major: "计算机科学与技术",
  class: "计八",
)

= 实验目的

本实验围绕图文多模态问答（Visual Question Answering, VQA）任务，调用 Hugging Face 上的 ViLT 预训练模型，实现“输入图片 + 自然语言问题，输出文本答案”的完整流程。实验目标包括三点：第一，理解图像与文本联合输入的多模态推理形式；第二，使用 ViLT 处理器和问答模型完成单张图片推理；第三，设计多个测试问题，记录模型答案与置信度，并将测试案例截图整理到报告中。

= 任务内容

根据作业要求，本实验需要完成以下内容：

- 使用 ViLT 预训练模型，模型来源为 Hugging Face 模型库。
- 尝试调用模型，输入图片和问题，例如“图中有什么动物？”。
- 问题可以自行定义，程序应输出对应答案。
- 提交一份至少 2 页的简单报告，报告中需要包含测试案例截图。

本实验代码位于 `lab/06`，核心模型为 `dandelin/vilt-b32-finetuned-vqa`。该模型以 ViLT 为骨干，并在 VQA 数据集上微调，可直接用于开放词表规模受限的视觉问答任务。

= 项目结构

```text
lab/06
├── data
│   ├── cases.json
│   ├── README.md
│   └── samples
│       └── cat_dog.jpg
├── report
│   ├── figures
│   │   ├── case_1.png
│   │   ├── case_2.png
│   │   └── case_3.png
│   ├── report.typ
│   └── result.csv
├── requirements.txt
├── run.sh
└── src
    ├── run_cases.py
    └── vqa.py
```

其中 `src/vqa.py` 提供单张图片问答命令行接口，`src/run_cases.py` 读取 `data/cases.json` 并批量运行测试案例。默认测试图片 `data/samples/cat_dog.jpg` 来自 Wikimedia Commons，来源与许可记录在 `data/README.md`。运行后，答案会写入 `report/result.csv`，每个测试案例会生成一张可放入报告的 PNG 截图卡片。

= 方法设计

ViLT 的核心思想是将图像块和文本 token 统一送入 Transformer 编码器。与一些先提取目标区域特征、再进行跨模态融合的模型不同，ViLT 直接使用较轻量的图像嵌入和文本嵌入进行联合建模，减少了外部目标检测器带来的计算成本。本实验使用 `ViltProcessor` 完成图像缩放、归一化和文本分词，再用 `ViltForQuestionAnswering` 输出候选答案类别的 logits。

推理过程如下：

1. 读取输入图片，并转换为 RGB 格式。
2. 将图片和问题交给 `ViltProcessor`，得到模型需要的张量输入。
3. 将输入送入 `ViltForQuestionAnswering`。
4. 对 logits 做 softmax，选择概率最高的答案标签作为输出。
5. 记录答案、置信度、图片路径、问题和模型名称。

这种做法适合快速验证 VQA 模型调用流程。需要注意的是，VQA 微调模型的答案通常来自训练集中的高频答案集合，因此它更擅长回答颜色、数量、物体类别、场景等短答案问题；对于复杂推理或长文本解释，输出可能不稳定。

= 运行方式

建议在仓库根目录创建并启用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

进入实验目录后安装依赖：

```bash
cd lab/06
python -m pip install -r requirements.txt
```

运行完整批量测试：

```bash
bash run.sh
```

也可以对任意图片和问题运行单次推理：

```bash
python src/vqa.py --image data/samples/cat_dog.jpg --question "What animal is in front?"
```

如果模型已经提前下载到 Hugging Face 缓存，且希望离线运行，可以加入：

```bash
python src/vqa.py --local-files-only --image data/samples/cat_dog.jpg --question "How many dogs are in the image?"
```

= 测试案例

默认测试图片来自 Wikimedia Commons，文件名为 `041115 Brindle Boxer and house cat.jpg`，作者 Rufus Sarsaparilla 已将其释放到 public domain。图片中一只猫在前方，一只狗在后方。测试问题覆盖对象识别、空间关系和数量三类常见 VQA 类型。

#let result = csv("result.csv")

#figure(
  text(size: 8pt)[
    #table(
      columns: result.at(0).len(),
      table.header(..result.at(0).map(it => strong(it))),
      ..result.slice(1).flatten(),
    )
  ],
  caption: [ViLT 图文问答测试结果],
)

#figure(
  image("figures/case_1.png", width: 92%),
  caption: [测试案例 1：询问前方动物],
)

#figure(
  image("figures/case_2.png", width: 92%),
  caption: [测试案例 2：询问猫后方动物],
)

#figure(
  image("figures/case_3.png", width: 92%),
  caption: [测试案例 3：询问图片中的狗数量],
)

= 结果分析

从测试结果可以观察到，ViLT 能够同时利用图像内容和文本问题进行预测。当问题询问前后关系时，模型需要定位猫和狗在画面中的相对位置；当问题询问数量时，模型需要对图中主要动物进行计数。真实照片中的光照、遮挡和姿态更复杂，因此比手绘测试图更能体现视觉问答模型在实际图像上的表现。

模型输出的置信度可作为结果可信程度的辅助参考，但不能简单等同于人工判断的正确率。若问题表达比较清晰、答案属于常见短词，模型通常更容易输出稳定答案；若问题过于开放，或者图片风格与真实照片差异较大，模型可能给出错误答案。实际使用时，可以替换为真实照片，并在 `data/cases.json` 中加入更多自定义问题进行测试。

= 结论

本实验完成了 ViLT 预训练模型的调用流程，支持图片和自然语言问题作为输入，并输出答案与置信度。代码提供单次推理和批量测试两种入口，批量测试会自动生成结果表和测试案例图，便于撰写报告和复现实验。

通过本实验可以看到，VQA 任务不是单纯的图像分类或文本问答，而是需要模型建立图像区域、文本对象和问题语义之间的对应关系。ViLT 将视觉和语言输入统一到 Transformer 中，为图文多模态任务提供了简洁的实现方式。后续可以进一步尝试真实照片、中文问题翻译、更多 Hugging Face VQA 模型对比，或将该流程扩展成一个简单的交互式网页应用。

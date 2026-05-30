#import "@local/bubble-sysu:0.1.0": *

#show: report.with(
  title: "实验五：股票价格预测",
  subtitle: "大数据原理与技术实验报告",
  student: (name: "元朗曦", id: "23336294"),
  school: "计算机学院",
  major: "计算机科学与技术",
  class: "计八",
)

= 实验目的

本实验围绕股票收盘价时间序列预测任务，分别使用传统统计模型 ARIMA 和深度学习模型 LSTM 预测未来 7 个交易日的股票价格，并使用 MAE 与 RMSE 指标比较两类模型的预测误差。实验目标包括三点：第一，完成股票历史数据 CSV 的读取、清洗和训练测试划分；第二，使用 ARIMA 建立时间序列基线模型；第三，使用 LSTM 建立序列神经网络模型，并从误差和运行时间角度分析二者差异。

股票价格预测是典型的时间序列回归问题。与普通监督学习样本不同，价格序列具有明显的时间顺序，训练阶段不能随机打乱日期，也不能把未来数据泄露给过去。因此，本实验将最后 7 个交易日作为测试集，其余历史收盘价作为训练集，两个模型都只根据训练集向后预测 7 步。

= 数据集与预处理

实验代码默认使用 Alpha Vantage 提供的日频股票历史数据，默认股票代码为 `AAPL`。Alpha Vantage 的官方文档说明，`TIME_SERIES_DAILY` 接口支持 `datatype=csv` 返回 CSV，并使用 `symbol` 和 `apikey` 指定股票代码与 API key。运行 `python src/main.py` 时，程序会在配置 `LAB05_ALPHA_VANTAGE_APIKEY` 后把 CSV 下载到 `data/raw/aapl_alpha_vantage.csv`。若网络不可用、未提供 API key，或返回文件不是合法股票 CSV，代码会生成 `data/raw/demo_stock.csv`，用于验证完整实验流程。正式提交时可以将手动下载的股票历史 CSV 放入 `data/raw`，再通过 `LAB05_CSV=data/raw/文件名.csv python src/main.py` 指定输入文件。

CSV 至少需要包含 `Date` 和 `Close` 两列。程序读取数据后，会将日期列转换为时间类型，将收盘价列转换为浮点数，并按日期升序排序。随后将最后 7 条记录作为测试集，其余记录作为训练集。ARIMA 直接使用原始收盘价序列；LSTM 则先用训练集均值和标准差做标准化，再用滑动窗口构造监督学习样本，避免价格量纲过大影响神经网络训练。

= 方法设计

== ARIMA 模型

ARIMA 模型由自回归项 AR、差分项 I 和移动平均项 MA 组成，适合描述单变量时间序列的线性相关结构。本实验使用 `ARIMA(5, 1, 0)` 作为基线配置，其中一阶差分用于削弱股票价格序列中的趋势性，5 阶自回归项用于利用最近若干天的价格变化信息。模型在训练集上拟合后，一次性向后预测未来 7 个交易日的收盘价。

ARIMA 的优点是结构清晰、训练成本较低，并且在样本规模不大时也能稳定运行。它的局限在于主要表达线性时间相关关系，对强非线性波动、突发事件和复杂市场因素的建模能力有限。

== LSTM 模型

LSTM 是循环神经网络的一种改进结构，通过输入门、遗忘门和输出门控制历史信息的保留与更新，能够在一定程度上缓解普通 RNN 的长期依赖问题。本实验使用一个轻量 LSTM 网络：输入为最近 20 个交易日的标准化收盘价，LSTM 层提取序列特征，最后通过线性层输出下一日价格。

为了完成未来 7 日预测，LSTM 采用递归预测方式。模型先预测第 1 天，把预测值追加到历史窗口中，再继续预测第 2 天，直到得到 7 个预测值。这种方式符合真实预测场景，因为第 2 天之后的真实价格在预测时并不可见。不过，递归预测也会带来误差累积问题，如果前几步预测偏差较大，后续窗口会继续使用这些预测值，可能放大误差。

= 运行方式

实验代码位于 `lab/05` 目录。建议先在仓库根目录创建并启用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

进入实验目录后安装依赖：

```bash
cd lab/05
python -m pip install -r requirements.txt
```

然后运行完整实验：

```bash
python src/main.py
```

可以通过环境变量调整股票代码、输入 CSV、预测窗口和训练轮数：

```bash
LAB05_ALPHA_VANTAGE_APIKEY=your_key python src/main.py
LAB05_SYMBOL=MSFT LAB05_ALPHA_VANTAGE_APIKEY=your_key LAB05_LSTM_EPOCHS=80 python src/main.py
LAB05_CSV=data/raw/my_stock.csv python src/main.py
```

实验结束后，MAE、RMSE 和运行时间会写入 `report/result.csv`，预测曲线与 LSTM 损失曲线会写入 `report/figures/`。若要生成 PDF 报告，可以进入 `report` 目录执行：

```bash
typst compile report.typ
```

= 实验结果

#let result = csv("result.csv")

#figure(
  text(size: 8pt)[
    #table(
      columns: result.at(0).len(),
      table.header(..result.at(0).map(it => strong(it))),
      ..result.slice(1).flatten(),
    )
  ],
  caption: [ARIMA 与 LSTM 的 7 日预测误差对比],
)

#figure(
  image("figures/forecast.svg", width: 92%),
  caption: [真实收盘价与两种模型预测结果],
)

#figure(
  image("figures/loss.svg", width: 82%),
  caption: [LSTM 训练过程中的均方误差损失],
)

从结果表中可以直接比较两种模型的 MAE 和 RMSE。MAE 表示平均绝对误差，单位与股票价格一致，容易理解模型平均偏离真实价格多少；RMSE 对较大的误差更加敏感，因此当某几天预测偏差明显时，RMSE 会比 MAE 更快上升。若 ARIMA 的 MAE 和 RMSE 更低，说明该股票最近一段时间的价格变化可以较好地由线性时间相关结构解释；若 LSTM 更低，则说明滑动窗口中的非线性模式对预测有所帮助。

预测曲线可以辅助判断误差来源。ARIMA 曲线通常较平滑，适合作为趋势基线，但在短期价格突然上升或下降时可能反应不足。LSTM 曲线受训练轮数、窗口长度和标准化方式影响更明显，如果训练数据较少或训练轮数不足，可能出现预测滞后或过度平滑。损失曲线用于观察神经网络训练是否收敛，若损失持续下降且后期趋于稳定，说明当前轻量模型已经基本学习到训练窗口中的局部模式。

= 结论

本实验完成了股票历史 CSV 数据读取、ARIMA 预测、LSTM 预测和 MAE/RMSE 指标比较。代码将最后 7 个交易日作为测试集，符合“预测未来 7 天价格”的任务设定；同时输出统一的 CSV 指标表和 SVG 图表，便于报告展示和重复运行。

从方法特点看，ARIMA 训练速度快、可解释性较强，适合用作时间序列预测基线；LSTM 能表达更复杂的非线性序列关系，但需要更多训练数据和参数调节。股票价格受市场情绪、宏观事件和公司消息影响明显，仅使用单变量历史收盘价无法覆盖全部因素，因此本实验结果更适合用于比较两种建模思路，而不应被理解为真实投资建议。后续可以加入成交量、开盘价、技术指标或多只相关股票数据，进一步扩展为多变量预测任务。

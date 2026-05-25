#import "@local/bubble-sysu:0.1.0": *

#show: report.with(
  title: "实验二：构建知识图谱",
  subtitle: "大数据原理与技术实验报告",
  student: (name: "元朗曦", id: "23336294"),
  school: "计算机学院",
  major: "计算机科学与技术",
  class: "计八",
)

= 实验目的

本实验面向人工智能领域构建一个小型知识图谱。实验选择人工智能研究人物、代表论文、研究机构和核心概念作为领域对象，使用自然语言处理工具从文本语料中抽取实体和关系，再将抽取结果组织为节点表、关系表和三元组表，并提供 Neo4j 导入脚本与查询语句用于图数据库构建和可视化分析。

实验目标包括三点：第一，说明自选数据集的来源和结构；第二，使用 NLP 方法完成实体识别与关系抽取；第三，用 Neo4j 的图模型表达领域对象之间的联系，并通过关键节点邻域展示图谱结构。

= 实验内容

== 数据集来源

本实验的数据文件为 `data/raw/ai_corpus.json`。语料由若干条短文本组成，每条文本包含 `id`、`title`、`source` 和 `text` 四个字段。文本内容人工整理自公开资料页面和论文元数据，来源包括 Geoffrey Hinton、Yann LeCun、Yoshua Bengio、Google Brain、OpenAI 等公开介绍页面，以及 `Attention Is All You Need`、`BERT`、`AlexNet` 等论文页面。

选择该数据集的原因是人工智能领域的实体和关系较清晰：人物通常与机构、论文、研究概念相连；论文通常与作者、年份、技术概念相连；概念之间也存在明显的相关关系。这样的结构适合用知识图谱表达，也方便在 Neo4j 中进行邻域查询和路径分析。

== 实体与关系抽取方法

本实验脚本 `src/extract.py` 使用 spaCy 的规则流水线思路完成抽取。为了保证作业在本地可复现，脚本优先使用 `spacy.blank("en")` 和 `EntityRuler` 注册领域词表；如果本机尚未安装 spaCy，则使用同一份领域词表进行规则匹配作为降级路径。这样既保留了 NLP 工具的实体识别流程，也避免因为模型下载失败导致实验无法运行。

实体类型设计如下：

- `Person`：研究人员，例如 Geoffrey Hinton、Yann LeCun。
- `Paper`：代表论文，例如 `Attention Is All You Need`、`BERT`。
- `Institution`：研究机构或公司，例如 University of Toronto、OpenAI。
- `Concept`：领域概念，例如 Transformer、deep learning。
- `Year`：论文发表年份，例如 2012、2017、2018。

关系类型设计如下：

- `AUTHORED`：人物撰写论文。
- `AFFILIATED_WITH`：人物或团队隶属于机构。
- `PROPOSED`：人物、论文或机构提出相关概念。
- `RELATED_TO`：概念、机构与研究方向之间存在关联。
- `PUBLISHED_IN`：论文发表于某一年。

抽取结果分别保存为 `data/processed/nodes.csv`、`data/processed/relationships.csv` 和 `data/processed/triples.csv`。其中节点表记录实体编号、名称、类型和来源文档；关系表记录起点、终点、关系类型、证据句和来源文档；三元组表则用 `(head, relation, tail)` 的形式保存图谱事实。

== 图谱统计与本地可视化

#let result = csv("result.csv")

#figure(
  table(
    columns: 2,
    table.header(..result.at(0).flatten().map(it => strong(it))),
    ..result.slice(1).flatten(),
  ),
  caption: [知识图谱规模统计],
)

#figure(
  image("figures/entity_distribution.svg", width: 80%),
  caption: [实体类型分布],
)

#figure(
  image("figures/graph_overview.svg", width: 95%),
  caption: [人工智能领域知识图谱局部结构],
)

从统计结果可以看出，图谱覆盖了人物、论文、机构、概念和年份五类实体。人物节点数量最多，说明语料主要围绕代表性研究人员和论文作者展开；概念节点连接了不同论文和机构，是图谱中较重要的语义桥梁。例如 Transformer 同时连接论文 `Attention Is All You Need`、BERT 和自然语言处理方向，能够体现技术概念在不同研究对象之间的关联作用。

== Neo4j 构建与查询

Neo4j 是图数据库，适合保存由节点和关系构成的知识图谱。本实验提供 `docker-compose.yml`，并在 `run.sh` 中使用 Docker Compose 启动本地 Neo4j 服务，用户名和密码为 `neo4j/sysu-big-data`。Neo4j 的数据、日志和插件目录由 Docker named volume 管理，因此项目目录下不需要创建运行时 `var` 文件夹。运行 `./run.sh` 后，可以执行 `src/import_neo4j.py` 将节点和关系写入 Neo4j。

导入后，可在 Neo4j Browser 中执行 `src/neo4j/queries.cypher` 中的查询。例如查询 Geoffrey Hinton 的一跳或两跳邻域：

```cypher
MATCH path = (p:Person {name: 'Geoffrey Hinton'})-[*1..2]-(x)
RETURN path
LIMIT 50;
```

该查询会展示 Geoffrey Hinton 与 University of Toronto、AlexNet、deep learning、neural networks 等节点之间的连接。再例如查询 Transformer 相关子图：

```cypher
MATCH path = (x)-[*1..2]-(c:Concept {name: 'Transformer'})
RETURN path
LIMIT 50;
```

该查询能够观察 Transformer 与论文、BERT、自然语言处理和自注意力之间的联系。相比 CSV 表格，Neo4j 的图视图更直观地呈现了实体之间的网络结构，便于发现关键节点和多跳关系。

#figure(
  image("figures/neo4j_query_graph.svg", width: 95%),
  caption: [Neo4j 查询导出的 Geoffrey Hinton 邻域图],
)

= 运行实验

实验代码位于 `lab/02` 目录。首先进入实验目录并安装 Python 依赖：

```bash
cd lab/02
python3 -m pip install -r requirements.txt
```

随后运行主脚本：

```bash
./run.sh
```

该脚本会依次完成实体关系抽取、图谱统计、本地 SVG 可视化生成，并通过 Docker Compose 启动 Neo4j 服务。脚本执行后，`data/processed/` 中会生成 `nodes.csv`、`relationships.csv` 和 `triples.csv`，`report/figures/` 中会生成实体类型分布图和图谱结构图。

等待 Neo4j 完成启动后，执行以下命令将抽取结果写入图数据库：

```bash
python3 src/import_neo4j.py
```

如果需要从 Neo4j 查询结果直接生成报告图片，可以运行：

```bash
python3 src/export_neo4j_graph.py
```

该脚本会连接 Neo4j，执行以 Geoffrey Hinton 为中心的一到两跳邻域查询，并将返回的节点和关系保存为 `report/figures/neo4j_query_graph.svg`。也可以通过环境变量调整中心节点，例如 `NEO4J_FOCUS=Transformer python3 src/export_neo4j_graph.py`。

最后打开 `http://localhost:7474`，使用用户名 `neo4j` 和密码 `sysu-big-data` 登录 Neo4j Browser，并执行 `src/neo4j/queries.cypher` 中的查询语句，即可查看全图统计、人物邻域、Transformer 相关子图和论文作者关系。

= 实验总结

本实验完成了一个小型人工智能领域知识图谱的构建流程。首先根据公开资料整理语料并说明数据来源；其次使用 spaCy 规则流水线完成实体和关系抽取；最后将抽取结果整理为节点表、关系表、三元组表，并提供 Neo4j 导入和查询脚本。实验结果表明，知识图谱能够将人物、论文、机构和概念统一到同一种图结构中，适合表达复杂领域对象之间的关联。

本实验的局限在于语料规模较小，关系抽取主要依赖规则模板，复杂句式和隐含关系尚不能充分识别。后续可以扩大数据来源，引入更完整的论文元数据和引用关系，并结合统计模型或大语言模型提高关系抽取的覆盖率。

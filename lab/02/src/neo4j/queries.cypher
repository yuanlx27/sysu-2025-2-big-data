// 1. 查看全图规模
MATCH (n)
RETURN labels(n) AS labels, count(n) AS count
ORDER BY count DESC;

MATCH ()-[r]->()
RETURN type(r) AS relation, count(r) AS count
ORDER BY count DESC;

// 2. 可视化 Geoffrey Hinton 的一跳邻域
MATCH path = (p:Person {name: 'Geoffrey Hinton'})-[*1..2]-(x)
RETURN path
LIMIT 50;

// 3. 可视化 Transformer 相关论文与概念
MATCH path = (x)-[*1..2]-(c:Concept {name: 'Transformer'})
RETURN path
LIMIT 50;

// 4. 查询论文、作者和发表年份
MATCH (author:Person)-[:AUTHORED]->(paper:Paper)-[:PUBLISHED_IN]->(year:Year)
RETURN paper.name AS paper, collect(author.name) AS authors, year.name AS year
ORDER BY year;

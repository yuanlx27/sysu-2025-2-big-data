LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (e:Entity {id: row.id})
SET e.name = row.name,
    e.label = row.label,
    e.source_doc = row.source_doc
WITH e, row
FOREACH (_ IN CASE WHEN row.label = 'Person' THEN [1] ELSE [] END | SET e:Person)
FOREACH (_ IN CASE WHEN row.label = 'Paper' THEN [1] ELSE [] END | SET e:Paper)
FOREACH (_ IN CASE WHEN row.label = 'Institution' THEN [1] ELSE [] END | SET e:Institution)
FOREACH (_ IN CASE WHEN row.label = 'Concept' THEN [1] ELSE [] END | SET e:Concept)
FOREACH (_ IN CASE WHEN row.label = 'Year' THEN [1] ELSE [] END | SET e:Year);

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (source:Entity {id: row.source})
MATCH (target:Entity {id: row.target})
FOREACH (_ IN CASE WHEN row.type = 'AUTHORED' THEN [1] ELSE [] END |
  MERGE (source)-[r:AUTHORED]->(target)
  SET r.evidence = row.evidence, r.source_doc = row.source_doc
)
FOREACH (_ IN CASE WHEN row.type = 'AFFILIATED_WITH' THEN [1] ELSE [] END |
  MERGE (source)-[r:AFFILIATED_WITH]->(target)
  SET r.evidence = row.evidence, r.source_doc = row.source_doc
)
FOREACH (_ IN CASE WHEN row.type = 'PROPOSED' THEN [1] ELSE [] END |
  MERGE (source)-[r:PROPOSED]->(target)
  SET r.evidence = row.evidence, r.source_doc = row.source_doc
)
FOREACH (_ IN CASE WHEN row.type = 'RELATED_TO' THEN [1] ELSE [] END |
  MERGE (source)-[r:RELATED_TO]->(target)
  SET r.evidence = row.evidence, r.source_doc = row.source_doc
)
FOREACH (_ IN CASE WHEN row.type = 'PUBLISHED_IN' THEN [1] ELSE [] END |
  MERGE (source)-[r:PUBLISHED_IN]->(target)
  SET r.evidence = row.evidence, r.source_doc = row.source_doc
);

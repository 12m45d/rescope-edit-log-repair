-- E10 verification: the relational workload of amendment R22 on TX-G2XL-rome-T100, the verifier's own schema and statements
-- (written from PROTOCOL.md R22 (a)-(d); independent of run/code/r22.sql).  Executed by v3_r22.py by name.
--
-- tuple(rid, ...)      one row per record of D: the requests of the log (edited = 1) and the VIEW pool (half 'report' | 'test')
-- state(name, rid, v)  the value of a tuple (first token id of the greedy continuation of its request prompt) in a state:
--                      'R0' (pre-edit model), 'deployed', 'intended' (R*), and every repaired model
-- inview(half, relation, rid)   membership of the selection view V_r of a half: the half's VIEW tuples of relation r and the
--                               edited tuples of relation r (which belong to the view of either half)

-- name: schema
CREATE TABLE tuple (rid INTEGER PRIMARY KEY, subject TEXT NOT NULL, relation TEXT NOT NULL, prompt TEXT NOT NULL,
                    edited INTEGER NOT NULL, half TEXT, pos INTEGER, new_first INTEGER);
CREATE TABLE state (name TEXT NOT NULL, rid INTEGER NOT NULL, v INTEGER NOT NULL, PRIMARY KEY (name, rid));
CREATE TABLE inview (half TEXT NOT NULL, relation TEXT NOT NULL, rid INTEGER NOT NULL, PRIMARY KEY (half, relation, rid));
CREATE TABLE ownreq (rid INTEGER PRIMARY KEY, edit INTEGER NOT NULL);

-- name: fill_inview
INSERT INTO inview (half, relation, rid)
  SELECT h.half, t.relation, t.rid FROM tuple t JOIN (SELECT 'report' AS half UNION ALL SELECT 'test') h
   WHERE t.edited = 1 OR t.half = h.half;

-- name: intended_state
-- R*: the first token of the new object for edited tuples, the pre-edit value for all other tuples
INSERT INTO state (name, rid, v)
  SELECT 'intended', t.rid, CASE WHEN t.edited = 1 THEN t.new_first ELSE r0.v END
    FROM tuple t JOIN state r0 ON r0.rid = t.rid AND r0.name = 'R0';

-- name: queries_of_half
-- the relations with at least 20 tuples in the view of the half (each gives a selection view and an aggregate)
SELECT relation, COUNT(*) AS n FROM inview WHERE half = :half GROUP BY relation HAVING COUNT(*) >= 20 ORDER BY relation;

-- name: dropped_of_half
SELECT relation, COUNT(*) AS n FROM inview WHERE half = :half GROUP BY relation HAVING COUNT(*) < 20 ORDER BY relation;

-- name: state_sizes
SELECT name, COUNT(*) AS n FROM state GROUP BY name ORDER BY name;

-- name: states_differ
-- number of tuples of a tuple class whose value differs between two states
SELECT COUNT(*) AS n, SUM(a.v != b.v) AS n_differ
  FROM tuple t JOIN state a ON a.rid = t.rid AND a.name = :a JOIN state b ON b.rid = t.rid AND b.name = :b
 WHERE t.edited = :edited AND (:half IS NULL OR t.half = :half);

-- name: reference_preservation
-- unedited tuples of the test half: value under the model equals the pre-edit value
SELECT COUNT(*) AS n, SUM(m.v = r0.v) AS n_ok
  FROM tuple t JOIN state m ON m.rid = t.rid AND m.name = :model JOIN state r0 ON r0.rid = t.rid AND r0.name = 'R0'
 WHERE t.edited = 0 AND t.half = 'test';

-- name: update_retention
-- edited tuples: value under the model is the first token of the new object
SELECT COUNT(*) AS n, SUM(m.v = t.new_first) AS n_ok
  FROM tuple t JOIN state m ON m.rid = t.rid AND m.name = :model
 WHERE t.edited = 1;

-- name: selection_views
-- per selection view of the half: tuples whose value differs from the intended state (the view's result equals its result on R* iff 0)
SELECT q.relation, COUNT(*) AS n, SUM(m.v != i.v) AS n_wrong
  FROM inview q JOIN state m ON m.rid = q.rid AND m.name = :model JOIN state i ON i.rid = q.rid AND i.name = 'intended'
 WHERE q.half = :half
 GROUP BY q.relation HAVING COUNT(*) >= 20 ORDER BY q.relation;

-- name: aggregates
-- per aggregate SELECT value, COUNT(*) ... GROUP BY value of the half: half the L1 distance between the grouped counts under the
-- model and under R* (signed counts: +1 per tuple under the model, -1 per tuple under R*), and the view size
WITH signed AS (
  SELECT q.relation AS relation, m.v AS v, 1 AS w FROM inview q JOIN state m ON m.rid = q.rid AND m.name = :model WHERE q.half = :half
  UNION ALL
  SELECT q.relation AS relation, i.v AS v, -1 AS w FROM inview q JOIN state i ON i.rid = q.rid AND i.name = 'intended' WHERE q.half = :half),
 g AS (SELECT relation, v, SUM(w) AS d FROM signed GROUP BY relation, v),
 n AS (SELECT relation, COUNT(*) AS n FROM inview WHERE half = :half GROUP BY relation)
SELECT g.relation, n.n AS n, 0.5 * SUM(ABS(g.d)) AS half_l1, 0.5 * SUM(ABS(g.d)) / n.n AS tv
  FROM g JOIN n ON n.relation = g.relation WHERE n.n >= 20 GROUP BY g.relation ORDER BY g.relation;

-- name: view_tuples
-- the tuples of the selection view of a relation in the report half (what the operator's diagnosis evaluates)
SELECT q.rid FROM inview q WHERE q.half = 'report' AND q.relation = :relation ORDER BY q.rid;

-- name: diagnosis
-- tuple-level complaints of a complained query on relation r: tuples of its view whose deployed value differs from the pre-edit
-- value and whose prompt is not the own request of a logged edit
SELECT q.rid FROM inview q JOIN state d ON d.rid = q.rid AND d.name = 'deployed' JOIN state r0 ON r0.rid = q.rid AND r0.name = 'R0'
 WHERE q.half = 'report' AND q.relation = :relation AND d.v != r0.v AND q.rid NOT IN (SELECT rid FROM ownreq) ORDER BY q.rid;

-- name: update_retention_range
-- edited tuples at log positions lo <= pos < hi: value under the model is the first token of the new object
SELECT COUNT(*) AS n, SUM(m.v = t.new_first) AS n_ok
  FROM tuple t JOIN state m ON m.rid = t.rid AND m.name = :model
 WHERE t.edited = 1 AND t.pos >= :lo AND t.pos < :hi;

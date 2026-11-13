-- E10 / R22: relational maintenance workload on a text log.  ALL read-outs of the workload are computed by the statements
-- of this file on a sqlite3 database (one per log: out/r22/<log>.sqlite); run_r22.py only fills the tables and executes
-- the statements by name (":name" parameters).
--
-- Relation Facts(subject, relation, value).  A tuple is a record: the requests of the log (edited tuples) and the VIEW pool.
-- The value of a tuple under a served model is the first token (token id) of its greedy continuation of the record's prompt.
-- States are rows of val(model, rid, value): 'preedit' (R0), 'deployed', every repaired model; the INTENDED state R* is
-- facts.intended (first token of the new object for edited tuples, the pre-edit value for all others).

-- name: schema
CREATE TABLE facts (
  rid       INTEGER PRIMARY KEY,   -- record index of the data set
  subject   TEXT NOT NULL,
  relation  TEXT NOT NULL,
  prompt    TEXT NOT NULL,
  edited    INTEGER NOT NULL,      -- 1: a request of the log
  half      TEXT NOT NULL,         -- 'report' | 'test' (VIEW pool) | 'both' (edited tuples belong to the views of either half)
  pos       INTEGER,               -- position in the log (edited tuples)
  own_edit  INTEGER NOT NULL,      -- >= 0: a position of the prompt carries the logged solver key of that edit (own request); -1: none
  pre       INTEGER,               -- value under the pre-edit model (R0)
  intended  INTEGER                -- value in the intended state R*
);
CREATE TABLE val (model TEXT NOT NULL, rid INTEGER NOT NULL, value INTEGER NOT NULL, PRIMARY KEY (model, rid));
CREATE TABLE member (half TEXT NOT NULL, relation TEXT NOT NULL, rid INTEGER NOT NULL, PRIMARY KEY (half, relation, rid));
CREATE TABLE valid_rel (half TEXT NOT NULL, relation TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY (half, relation));
CREATE TABLE blocklist (setname TEXT NOT NULL, rid INTEGER NOT NULL, PRIMARY KEY (setname, rid));

-- name: fill_member
INSERT INTO member (half, relation, rid)
  SELECT half, relation, rid FROM facts WHERE edited = 0
  UNION ALL SELECT 'report', relation, rid FROM facts WHERE edited = 1
  UNION ALL SELECT 'test', relation, rid FROM facts WHERE edited = 1;

-- name: fill_valid_rel
-- the queries of a half: every relation with at least 20 tuples in the view of that half
INSERT INTO valid_rel (half, relation, n)
  SELECT half, relation, COUNT(*) FROM member GROUP BY half, relation HAVING COUNT(*) >= 20;

-- name: dropped_relations
SELECT half, relation, COUNT(*) AS n FROM member GROUP BY half, relation HAVING COUNT(*) < 20 ORDER BY half, relation;

-- name: set_pre
UPDATE facts SET pre = (SELECT value FROM val WHERE model = 'preedit' AND val.rid = facts.rid);

-- name: set_intended_unedited
-- R*: the pre-edit value for every tuple that the log does not edit (edited tuples: the first token of the new object, from the log)
UPDATE facts SET intended = pre WHERE edited = 0;

-- name: materialise_intended
-- R* as a state table, so that it is read by the same statements as every model
INSERT INTO val (model, rid, value) SELECT 'intended', rid, intended FROM facts;

-- name: materialise_blocklist
-- deployed model + exact-match list: a listed tuple is answered by the pre-edit model
INSERT INTO val (model, rid, value)
  SELECT :model, d.rid, CASE WHEN d.rid IN (SELECT rid FROM blocklist WHERE setname = :setname) THEN f.pre ELSE d.value END
  FROM val d JOIN facts f ON f.rid = d.rid WHERE d.model = 'deployed';

-- name: the_selection_view
-- V_r of a half under a model (the query a user asks; listed for completeness, used by the test of the SQL)
SELECT f.subject, v.value FROM member m JOIN facts f ON f.rid = m.rid JOIN val v ON v.rid = m.rid AND v.model = :model
 WHERE m.half = :half AND m.relation = :relation ORDER BY f.rid;

-- name: the_aggregate
-- SELECT value, COUNT(*) FROM Facts WHERE relation = r GROUP BY value, on the view of a half under a model
SELECT v.value, COUNT(*) AS c FROM member m JOIN val v ON v.rid = m.rid AND v.model = :model
 WHERE m.half = :half AND m.relation = :relation GROUP BY v.value ORDER BY v.value;

-- name: reference_preservation
-- share of unedited test tuples whose value equals the pre-edit value
SELECT COUNT(*) AS n, SUM(v.value = f.pre) AS n_ok
  FROM facts f JOIN val v ON v.rid = f.rid AND v.model = :model
 WHERE f.edited = 0 AND f.half = 'test';

-- name: update_retention
-- share of edited tuples (log positions lo <= pos < hi) whose value is the new object
SELECT COUNT(*) AS n, SUM(v.value = f.intended) AS n_ok
  FROM facts f JOIN val v ON v.rid = f.rid AND v.model = :model
 WHERE f.edited = 1 AND f.pos >= :lo AND f.pos < :hi;

-- name: tuple_flags
-- per-tuple outcome of a model on the tuples of the test views (for intervals)
SELECT f.rid, f.edited, f.pos, (v.value = f.pre) AS same_as_pre, (v.value = f.intended) AS as_intended
  FROM facts f JOIN val v ON v.rid = f.rid AND v.model = :model
 WHERE f.half IN ('test', 'both') ORDER BY f.rid;

-- name: view_results
-- per selection view of a half: number of tuples and number of tuples whose value differs from the intended state
-- (the result of the view equals its result on R* exactly iff n_wrong = 0)
SELECT m.relation, COUNT(*) AS n, SUM(v.value != f.intended) AS n_wrong
  FROM member m JOIN facts f ON f.rid = m.rid JOIN val v ON v.rid = m.rid AND v.model = :model
 WHERE m.half = :half AND m.relation IN (SELECT relation FROM valid_rel WHERE half = :half)
 GROUP BY m.relation ORDER BY m.relation;

-- name: aggregate_results
-- per aggregate of a half: total-variation distance between the grouped counts under the model and under R*
-- (0.5 * sum over values of |count_model - count_intended| / number of tuples); the results are equal iff it is 0
WITH mem AS (SELECT relation, rid FROM member WHERE half = :half AND relation IN (SELECT relation FROM valid_rel WHERE half = :half)),
     cm AS (SELECT mem.relation AS relation, v.value AS value, COUNT(*) AS c
              FROM mem JOIN val v ON v.rid = mem.rid AND v.model = :model GROUP BY mem.relation, v.value),
     cs AS (SELECT mem.relation AS relation, f.intended AS value, COUNT(*) AS c
              FROM mem JOIN facts f ON f.rid = mem.rid GROUP BY mem.relation, f.intended),
     ks AS (SELECT relation, value FROM cm UNION SELECT relation, value FROM cs),
     d AS (SELECT ks.relation AS relation, COALESCE(cm.c, 0) AS a, COALESCE(cs.c, 0) AS b
             FROM ks LEFT JOIN cm ON cm.relation = ks.relation AND cm.value = ks.value
                     LEFT JOIN cs ON cs.relation = ks.relation AND cs.value = ks.value)
SELECT d.relation, (SELECT COUNT(*) FROM mem WHERE mem.relation = d.relation) AS n,
       0.5 * SUM(ABS(d.a - d.b)) / (SELECT COUNT(*) FROM mem WHERE mem.relation = d.relation) AS tv,
       SUM(d.a != d.b) AS n_groups_different
  FROM d GROUP BY d.relation ORDER BY d.relation;

-- name: diagnosis
-- operator's diagnosis of a complained query on relation r (report half): the tuples of its view whose value under the
-- deployed model differs from the pre-edit value and whose prompt is not the own request of a logged edit
SELECT m.rid FROM member m JOIN facts f ON f.rid = m.rid JOIN val d ON d.rid = m.rid AND d.model = 'deployed'
 WHERE m.half = 'report' AND m.relation = :relation AND d.value != f.pre AND f.own_edit < 0 ORDER BY m.rid;

-- name: view_size
SELECT COUNT(*) AS n, SUM(f.own_edit >= 0) AS n_own_requests FROM member m JOIN facts f ON f.rid = m.rid
 WHERE m.half = :half AND m.relation = :relation;

-- name: counts
SELECT (SELECT COUNT(*) FROM facts WHERE edited = 1) AS n_edited,
       (SELECT COUNT(*) FROM facts WHERE edited = 0 AND half = 'report') AS n_report,
       (SELECT COUNT(*) FROM facts WHERE edited = 0 AND half = 'test') AS n_test,
       (SELECT COUNT(*) FROM valid_rel WHERE half = 'report') AS n_queries_report,
       (SELECT COUNT(*) FROM valid_rel WHERE half = 'test') AS n_queries_test,
       (SELECT COUNT(*) FROM facts WHERE edited = 0 AND own_edit >= 0) AS n_view_tuples_carrying_a_solver_key,
       (SELECT COUNT(*) FROM facts WHERE edited = 1 AND own_edit < 0) AS n_edited_tuples_not_recognised;

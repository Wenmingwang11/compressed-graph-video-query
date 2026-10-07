"""EQUI-VOCAL query execution, adapted to ordered maximal interval output.

Provenance: uwdb/EQUI-VOCAL, src/utils.py::postgres_execute_cache_sequence,
commit 96f4f4c631603376fb24d6c1c09ffb0b151433d6 (2025-09-05).
The published executor joins object relations in PostgreSQL and checks duration
with LEAD(fid, duration-1) partitioned by the video and ordered object variables.
Its final DISTINCT video IDs are replaced here by all maximal event intervals.
SQL-native angle/distance expressions replace the project's compiled C UDFs.
This is the execution component, not the paper's example-driven query synthesis.

Every query is executed in PostgreSQL. There is no query answer cache and no
call to the common Python exact verifier. Loading objects, creating indexes,
and ANALYZE are setup; SQL compilation/execution, fetching, and conversion are
online query work. The caller should time query() in its entirety.
"""
from __future__ import annotations

import io
import math
import os
import time
import uuid

import pandas as pd
import psycopg2

from supplement.corrected_query_model import QueryResult, QuerySpec


DEFAULT_DSN = 'host=127.0.0.1 port=55432 dbname=postgres user=baseline connect_timeout=5'
SOURCE_COMMIT = '96f4f4c631603376fb24d6c1c09ffb0b151433d6'


class EquivocalPostgresStrategy:
    artifact_kind = 'equivocal-postgresql-object-relation-and-duration-windows'

    def __init__(self, data: pd.DataFrame, frame_width: int, frame_height: int,
                 *, dsn: str | None = None) -> None:
        started = time.perf_counter()
        required = ['frame', 'track_id', 'class_id', 'cx', 'cy']
        missing = set(required).difference(data.columns)
        if missing:
            raise ValueError(f'detection table is missing columns: {sorted(missing)}')
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError('frame dimensions must be positive')
        self.frame_width, self.frame_height = int(frame_width), int(frame_height)
        self._table = 'equivocal_objects_' + uuid.uuid4().hex
        self._connection = None
        # Preserve the established exact_query behavior on colliding IDs:
        # all observed classes remain eligible, geometry is the stable last row.
        # Do not drop identical full rows before choosing that last geometry.
        positions = data[['frame', 'track_id', 'cx', 'cy']].drop_duplicates(
            ['frame', 'track_id'], keep='last')
        membership = data[['frame', 'track_id', 'class_id']].drop_duplicates()
        records = membership.merge(positions, on=['frame', 'track_id'], how='left',
                                   sort=False, validate='many_to_one')[required]
        self.duplicate_input_rows = int(data.duplicated(['frame', 'track_id']).sum())
        if not all(math.isfinite(float(value)) for key in ('cx', 'cy') for value in records[key]):
            raise ValueError('object coordinates must be finite')
        self.input_rows = len(records)
        self.input_frames = int(records['frame'].nunique())
        self._connection = psycopg2.connect(dsn or os.environ.get('EQUIVOCAL_PG_DSN', DEFAULT_DSN))
        self._connection.autocommit = True
        try:
            with self._connection.cursor() as cursor:
                # Match CPU query timing across methods; no parallel database workers.
                cursor.execute('SET max_parallel_workers_per_gather = 0')
                cursor.execute('SET jit = off')
                cursor.execute(f'CREATE TEMP TABLE {self._table} '
                               '(fid bigint NOT NULL, oid bigint NOT NULL, class_id integer NOT NULL, '
                               'cx double precision NOT NULL, cy double precision NOT NULL)')
                buffer = io.StringIO()
                for frame, oid, class_id, cx, cy in records.itertuples(index=False, name=None):
                    buffer.write(f'{int(frame)}\t{int(oid)}\t{int(class_id)}\t{float(cx)!r}\t{float(cy)!r}\n')
                buffer.seek(0)
                cursor.copy_from(buffer, self._table, columns=('fid', 'oid', 'class_id', 'cx', 'cy'))
                cursor.execute(f'CREATE INDEX ON {self._table} (class_id, fid)')
                cursor.execute(f'CREATE UNIQUE INDEX ON {self._table} (fid, oid, class_id)')
                cursor.execute(f'ANALYZE {self._table}')
                cursor.execute('SELECT pg_total_relation_size(%s::regclass)', (self._table,))
                self.artifact_bytes = int(cursor.fetchone()[0])
                cursor.execute('SHOW server_version')
                self.server_version = cursor.fetchone()[0]
        except BaseException:
            self.release()
            raise
        self.setup_seconds = time.perf_counter() - started
        self.artifact = {'table': self._table, 'bytes': self.artifact_bytes,
                         'source_commit': SOURCE_COMMIT, 'server_version': self.server_version,
                         'duplicate_input_rows': self.duplicate_input_rows,
                         'duplicate_policy': 'per frame/track all classes, stable last geometry'}

    def _query_sql(self, query: QuerySpec, return_all: bool) -> tuple[str, list]:
        role_count = len(query.role_types)
        bindings = ', '.join(f'oid{i}' for i in range(role_count))
        projection = ', '.join(f'o{i}.oid AS oid{i}' for i in range(role_count))
        sources = f'{self._table} o0'
        for i in range(1, role_count):
            sources += f' JOIN {self._table} o{i} ON o{i}.fid = o0.fid'
        conditions, parameters = [], []
        for i, class_id in enumerate(query.role_types):
            conditions.append(f'o{i}.class_id = %s')
            parameters.append(int(class_id))
        for i in range(role_count):
            for j in range(i + 1, role_count):
                conditions.append(f'o{i}.oid <> o{j}.oid')
        for (source, target), constraint in query.relations.items():
            dx = f'(o{target}.cx - o{source}.cx)'
            dy = f'(o{target}.cy - o{source}.cy)'
            # PostgreSQL geometric distance uses a numerically stable hypot.
            distance = f'(point(o{source}.cx, o{source}.cy) <-> point(o{target}.cx, o{target}.cy))'
            conditions.append(f'floor(atan2({dy}, {dx}) / (pi() / %s))::integer = ANY(%s)')
            parameters.extend([int(query.theta_parts), sorted(map(int, constraint.theta_bins))])
            conditions.append(f'floor({distance} / %s * %s)::bigint BETWEEN %s AND %s')
            parameters.extend([float(self.frame_width + self.frame_height), int(query.distance_parts),
                               int(constraint.d_min), int(constraint.d_max)])
        duration = int(query.min_consecutive_frames)
        limit = '' if return_all else f'LIMIT {int(query.topk)}'
        # The dataset table is the video scope (the original executor also
        # partitions by vid when a table contains multiple separate segments).
        sql = f'''WITH matched AS (
            SELECT DISTINCT o0.fid, {projection}
            FROM {sources} WHERE {' AND '.join(conditions)}
        ), duration_windows AS (
            SELECT *, lead(fid, {duration - 1}) OVER (
                PARTITION BY {bindings} ORDER BY fid) AS end_fid
            FROM matched
        ), valid_windows AS (
            SELECT * FROM duration_windows WHERE end_fid = fid + {duration - 1}
        ), window_groups AS (
            SELECT *, fid - row_number() OVER (
                PARTITION BY {bindings} ORDER BY fid) AS interval_group
            FROM valid_windows
        ), intervals AS (
            SELECT {bindings}, min(fid) AS start_frame, max(end_fid) AS end_frame
            FROM window_groups GROUP BY {bindings}, interval_group
        )
        SELECT {bindings}, start_frame, end_frame, count(*) OVER () AS qualifying_intervals
        FROM intervals
        ORDER BY (end_frame - start_frame + 1) DESC, {bindings}, start_frame, end_frame
        {limit}'''
        return sql, parameters

    def query(self, query: QuerySpec, *, stats: dict | None = None,
              return_all: bool = False) -> list[QueryResult]:
        if self._connection is None:
            raise RuntimeError('EQUI-VOCAL PostgreSQL strategy has been released')
        started = time.perf_counter()
        sql, parameters = self._query_sql(query, return_all)
        compiled = time.perf_counter()
        with self._connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            rows = cursor.fetchall()
        fetched = time.perf_counter()
        nroles = len(query.role_types)
        results = [QueryResult(tuple(int(v) for v in row[:nroles]),
                               int(row[nroles]), int(row[nroles + 1])) for row in rows]
        if stats is not None:
            stats.update(
                sql_compile_s=compiled - started,
                database_execute_fetch_s=fetched - compiled,
                result_conversion_s=time.perf_counter() - fetched,
                qualifying_intervals=int(rows[0][-1]) if rows else 0,
                postgres_version=self.server_version,
                source_commit=SOURCE_COMMIT,
                input_frames=self.input_frames,
                pruning_audit_available=False,
                execution_scope='PostgreSQL joins, predicates, duration windows, maximal intervals, ordered output',
            )
        return results

    def explain(self, query: QuerySpec) -> dict:
        """Untimed diagnostic plan; never substitutes an execution result."""
        if self._connection is None:
            raise RuntimeError('EQUI-VOCAL PostgreSQL strategy has been released')
        sql, parameters = self._query_sql(query, True)
        with self._connection.cursor() as cursor:
            cursor.execute('EXPLAIN (FORMAT JSON) ' + sql, parameters)
            return cursor.fetchone()[0][0]

    def release(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self.artifact = None


__all__ = ['EquivocalPostgresStrategy', 'SOURCE_COMMIT', 'DEFAULT_DSN']

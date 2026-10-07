"""EVA-MV (adapted): symbolic UDF-view reuse over pre-extracted objects.

Reference: Xu et al., SIGMOD 2022, DOI 10.1145/3514221.3526142, sections 3-4.
This is a documented reimplementation of a restricted reuse core, NOT the
authors' complete EVA system. Its symbolic language is a DNF of class equality
and closed integer frame ranges. Signature canonicalization, intersection /
difference / union, left-join conditional apply, and storing missing outputs
are implemented. General SymPy predicates, Cascades, vision model inference,
accuracy-constrained model selection, and EVA's full cost optimizer are not.

The offline view contains per-object class/center outputs already extracted by
the common benchmark input pipeline. No pair geometry, relational postings,
duration results, or query answers are precomputed. Every online query starts
from the same offline snapshot, performs all spatial conditions, joins, and
temporal grouping anew. Missing object-feature outputs use the supplied tracked
table, not a vision model; this is a post-extraction comparison only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Mapping

import numpy as np
import pandas as pd

from supplement.corrected_query_model import QueryResult, QuerySpec, quantize_relation


_COLUMNS = ['frame', 'track_id', 'class_id', 'cx', 'cy']
_KEYS = ['frame', 'track_id', 'class_id']


def _normalize_intervals(intervals):
    result = []
    for start, end in sorted((int(a), int(b)) for a, b in intervals):
        if end < start:
            continue
        if result and start <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return tuple(result)


@dataclass(frozen=True)
class PredicateDomain:
    """Canonical symbolic DNF: OR(class=c AND start<=frame<=end)."""
    ranges: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] = ()

    @classmethod
    def from_ranges(cls, mapping: Mapping):
        normalized = [(int(c), _normalize_intervals(intervals)) for c, intervals in mapping.items()]
        return cls(tuple(sorted((c, intervals) for c, intervals in normalized if intervals)))

    @property
    def is_empty(self):
        return not self.ranges

    def __and__(self, other):
        right = dict(other.ranges)
        return self.from_ranges({c: [(max(a, x), min(b, y)) for a, b in intervals
                                      for x, y in right.get(c, ()) if max(a, x) <= min(b, y)]
                                 for c, intervals in self.ranges})

    def __or__(self, other):
        values = {c: list(intervals) for c, intervals in self.ranges}
        for c, intervals in other.ranges:
            values.setdefault(c, []).extend(intervals)
        return self.from_ranges(values)

    def __sub__(self, other):
        right = dict(other.ranges)
        values = {}
        for c, intervals in self.ranges:
            remaining = list(intervals)
            for x, y in right.get(c, ()):
                pieces = []
                for a, b in remaining:
                    if y < a or x > b:
                        pieces.append((a, b))
                    else:
                        if a < x:
                            pieces.append((a, x - 1))
                        if y < b:
                            pieces.append((y + 1, b))
                remaining = pieces
            values[c] = remaining
        return self.from_ranges(values)

    def mask(self, data: pd.DataFrame):
        result = np.zeros(len(data), dtype=bool)
        frames, classes = data['frame'].to_numpy(), data['class_id'].to_numpy()
        for c, intervals in self.ranges:
            frame_match = np.zeros(len(data), dtype=bool)
            for start, end in intervals:
                frame_match |= (frames >= start) & (frames <= end)
            result |= (classes == c) & frame_match
        return result


@dataclass(frozen=True)
class UDFSignature:
    name: str
    sources: tuple[str, ...]
    parameters_json: str

    @classmethod
    def create(cls, name, sources, parameters):
        # Sources are a set in EVA's signature. Ordered roles, when needed,
        # belong in parameter values and are deliberately not sorted.
        return cls(str(name), tuple(sorted(set(map(str, sources)))),
                   json.dumps(parameters, sort_keys=True, separators=(',', ':')))


@dataclass(frozen=True)
class MaterializedView:
    domain: PredicateDomain
    outputs: pd.DataFrame


class MaterializedFeatureManager:
    """Restricted EVA UDF manager with guarded evaluation of missing tuples."""
    def __init__(self, source: pd.DataFrame, signature: UDFSignature | None = None):
        self.source = source
        if signature is None:
            fingerprint = hashlib.sha256(pd.util.hash_pandas_object(source, index=False).values.tobytes()).hexdigest()
            signature = UDFSignature.create('PreExtractedObjectFeatures/v1', (fingerprint,), {})
        self.signature = signature
        self.views = {signature: MaterializedView(PredicateDomain(), source.iloc[:0][_COLUMNS].copy())}

    def fork(self):
        new = MaterializedFeatureManager(self.source, self.signature)
        new.views = dict(self.views)
        return new

    def materialize(self, domain: PredicateDomain):
        return self.conditional_apply(domain)

    def conditional_apply(self, request: PredicateDomain, *, stats: dict | None = None):
        previous = self.views[self.signature]
        intersection = previous.domain & request
        difference = request - previous.domain
        union = previous.domain | request
        if difference.is_empty:
            outputs = previous.outputs.loc[request.mask(previous.outputs), _COLUMNS].copy()
            reused, computed = len(outputs), 0
        else:
            requested_inputs = self.source.loc[request.mask(self.source), _COLUMNS]
            if intersection.is_empty:
                # The adapted UDF reads the common pre-extracted class/center
                # output. It does not call a detector or reuse query answers.
                fresh = requested_inputs.copy()
                outputs = fresh
                reused, computed = 0, len(fresh)
            else:
                # Section 4.4: LEFT JOIN -> missing-value conditional apply -> STORE.
                joined = requested_inputs.merge(previous.outputs, on=_KEYS, how='left',
                                                suffixes=('_input', '_mv'), indicator=True)
                missing = joined['_merge'].eq('left_only')
                fresh = joined.loc[missing, _KEYS + ['cx_input', 'cy_input']].rename(
                    columns={'cx_input': 'cx', 'cy_input': 'cy'})
                cached = joined.loc[~missing, _KEYS + ['cx_mv', 'cy_mv']].rename(
                    columns={'cx_mv': 'cx', 'cy_mv': 'cy'})
                outputs = pd.concat([cached, fresh], ignore_index=True)
                reused, computed = len(cached), len(fresh)
            stored = pd.concat([previous.outputs, fresh], ignore_index=True)
            self.views[self.signature] = MaterializedView(union, stored)
        outputs = outputs.sort_values(['frame', 'track_id'], kind='stable').reset_index(drop=True)
        if stats is not None:
            for key, value in [('mv_symbolic_analyses', 1), ('mv_reused_input_rows', reused),
                               ('mv_computed_input_rows', computed)]:
                stats[key] = stats.get(key, 0) + value
            stats.setdefault('mv_coverage_trace', []).append({
                'request': request.ranges, 'intersection': intersection.ranges,
                'difference': difference.ranges, 'union': union.ranges})
        return outputs


class EVASymbolicMVStrategy:
    artifact_kind = 'eva-symbolic-object-feature-materialized-views-adapted'

    def __init__(self, data: pd.DataFrame, frame_width: int, frame_height: int,
                 *, preload_fraction: float = 1.0, frame_batch_size: int = 256):
        started = time.perf_counter()
        missing = set(_COLUMNS).difference(data.columns)
        if missing:
            raise ValueError(f'detection table is missing columns: {sorted(missing)}')
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError('frame dimensions must be positive')
        if not 0 <= preload_fraction <= 1 or frame_batch_size <= 0:
            raise ValueError('preload_fraction must be in [0,1] and frame_batch_size positive')
        self.frame_width, self.frame_height = int(frame_width), int(frame_height)
        self.frame_batch_size = int(frame_batch_size)
        # Match the pre-existing exhaustive oracle on ID collisions: a track
        # is eligible for every observed class, at the last input-row position.
        positions = data[['frame', 'track_id', 'cx', 'cy']].drop_duplicates(
            ['frame', 'track_id'], keep='last')
        membership = data[['frame', 'track_id', 'class_id']].drop_duplicates()
        source = membership.merge(positions, on=['frame', 'track_id'], how='left',
                                   sort=False, validate='many_to_one')
        source = source[_COLUMNS].sort_values(['frame', 'track_id', 'class_id'], kind='stable').reset_index(drop=True)
        duplicate_input_rows = int(data.duplicated(['frame', 'track_id']).sum())
        for c in ['frame', 'track_id', 'class_id']:
            source[c] = source[c].astype('int64')
        for c in ['cx', 'cy']:
            source[c] = source[c].astype('float64')
        if not np.isfinite(source[['cx', 'cy']].to_numpy()).all():
            raise ValueError('object coordinates must be finite')
        self._frames = np.sort(source.frame.unique())
        self._manager = MaterializedFeatureManager(source)
        preload_count = int(math.ceil(len(self._frames) * preload_fraction))
        if preload_count:
            first, last = int(self._frames[0]), int(self._frames[preload_count - 1])
            preload = PredicateDomain.from_ranges({int(c): [(first, last)] for c in source.class_id.unique()})
            self._manager.materialize(preload)
        view = self._manager.views[self._manager.signature]
        self.artifact_bytes = int(view.outputs.memory_usage(index=True, deep=True).sum())
        self.artifact = {
            'display_name': 'EVA-MV (adapted)', 'paper_doi': '10.1145/3514221.3526142',
            'udf_signature': {'name': self._manager.signature.name, 'sources': self._manager.signature.sources},
            'preloaded_predicate': view.domain.ranges, 'materialized_rows': len(view.outputs),
            'materialized_columns': list(_COLUMNS), 'bytes': self.artifact_bytes,
            'preload_fraction': float(preload_fraction), 'frame_batch_size': self.frame_batch_size,
            'source_table_bytes': int(source.memory_usage(index=True, deep=True).sum()),
            'query_dependent_views_retained': False,
            'pair_geometry_or_query_answers_precomputed': False,
            'symbolic_language': 'DNF of class equality and closed integer frame ranges',
            'scope': 'Symbolic UDF reuse core over pre-extracted objects; not full EVA or vision inference',
            'optimizer_scope': 'Cardinality-guided joins and early predicate evaluation; no original Cascades/model selection',
            'duplicate_input_rows': duplicate_input_rows,
            'duplicate_policy': 'per frame/track all classes, stable last geometry',
        }
        self.metadata = self.artifact
        self.setup_seconds = time.perf_counter() - started

    def _spatial_mask(self, table, source, target, constraint, query):
        sx, sy = table[f'x{source}'].to_numpy(), table[f'y{source}'].to_numpy()
        tx, ty = table[f'x{target}'].to_numpy(), table[f'y{target}'].to_numpy()
        dx, dy = tx - sx, ty - sy
        scaled_theta = np.arctan2(dy, dx) / (math.pi / query.theta_parts)
        scaled_distance = np.hypot(dx, dy) / float(self.frame_width + self.frame_height) * query.distance_parts
        theta, distance = np.floor(scaled_theta).astype('int64'), np.floor(scaled_distance).astype('int64')
        # At exact/near bin boundaries use the same scalar libm path as the
        # oracle; vector libm can differ by a final rounding bit at an integer.
        boundary = ((np.abs(scaled_theta - np.rint(scaled_theta)) < 1e-10) |
                    (np.abs(scaled_distance - np.rint(scaled_distance)) < 1e-10))
        for i in np.flatnonzero(boundary):
            theta[i], distance[i] = quantize_relation((sx[i], sy[i]), (tx[i], ty[i]),
                self.frame_width, self.frame_height, query.theta_parts, query.distance_parts)
        return ((distance >= constraint.d_min) & (distance <= constraint.d_max) &
                np.isin(theta, list(constraint.theta_bins)))

    def _batch_matches(self, roles, query, audit):
        if any(t.empty for t in roles.values()):
            return None
        seed = min(roles, key=lambda r: (len(roles[r]), r))
        bound = {seed}
        current = roles[seed]
        pending = set(roles).difference(bound)
        applied = set()
        while pending:
            connected = [r for r in pending if any(
                (r, previous) in query.relations or (previous, r) in query.relations for previous in bound)]
            role = min(connected or list(pending), key=lambda r: (len(roles[r]), r))
            current = current.merge(roles[role], on='frame', how='inner', sort=False)
            if audit is not None:
                audit['join_rows_formed'] = audit.get('join_rows_formed', 0) + len(current)
            different = np.ones(len(current), dtype=bool)
            for previous in bound:
                different &= current[f'id{role}'].to_numpy() != current[f'id{previous}'].to_numpy()
            current = current.loc[different]
            bound.add(role)
            pending.remove(role)
            ready = [(edge, c) for edge, c in query.relations.items()
                     if edge not in applied and edge[0] in bound and edge[1] in bound]
            ready.sort(key=lambda item: (len(item[1].theta_bins) * (item[1].d_max - item[1].d_min + 1), item[0]))
            for edge, constraint in ready:
                if audit is not None:
                    audit['spatial_predicate_evaluations'] = audit.get('spatial_predicate_evaluations', 0) + len(current)
                current = current.loc[self._spatial_mask(current, *edge, constraint, query)]
                applied.add(edge)
            if current.empty:
                return None
        return current[['frame'] + [f'id{i}' for i in range(len(query.role_types))]].drop_duplicates()

    def query(self, query: QuerySpec, *, stats: dict | None = None,
              return_all: bool = False) -> list[QueryResult]:
        if self._manager is None:
            raise RuntimeError('EVA-MV strategy has been released')
        started = time.perf_counter()
        if stats is not None:
            stats.update(mv_reused_input_rows=0, mv_computed_input_rows=0,
                         mv_symbolic_analyses=0, mv_coverage_trace=[],
                         spatial_predicate_evaluations=0, join_rows_formed=0,
                         pruning_audit_available=False)
        if not len(self._frames):
            if stats is not None:
                stats['qualifying_intervals'] = 0
            return []
        manager = self._manager.fork()
        first, last = int(self._frames[0]), int(self._frames[-1])
        classes = {int(c): manager.conditional_apply(
            PredicateDomain.from_ranges({int(c): [(first, last)]}), stats=stats)
            for c in sorted(set(query.role_types))}
        views_finished = time.perf_counter()
        # Build no cross-object artifact in setup; all joins and geometry occur here.
        role_tables = {r: classes[int(c)][['frame', 'track_id', 'cx', 'cy']].rename(
            columns={'track_id': f'id{r}', 'cx': f'x{r}', 'cy': f'y{r}'})
            for r, c in enumerate(query.role_types)}
        role_frames = {r: table.frame.to_numpy() for r, table in role_tables.items()}
        matches = []
        for offset in range(0, len(self._frames), self.frame_batch_size):
            begin = self._frames[offset]
            end = self._frames[min(offset + self.frame_batch_size, len(self._frames)) - 1]
            roles = {r: table.iloc[np.searchsorted(role_frames[r], begin, side='left'):
                                  np.searchsorted(role_frames[r], end, side='right')]
                     for r, table in role_tables.items()}
            found = self._batch_matches(roles, query, stats)
            if found is not None and not found.empty:
                matches.append(found)
        spatial_finished = time.perf_counter()
        results = []
        if matches:
            ids = [f'id{i}' for i in range(len(query.role_types))]
            matching = pd.concat(matches, ignore_index=True).sort_values(ids + ['frame'], kind='stable')
            breaks = matching[ids].ne(matching[ids].shift()).any(axis=1) | matching.frame.diff().ne(1)
            grouping = breaks.cumsum()
            intervals = matching.groupby(grouping, sort=False).agg(
                **{name: (name, 'first') for name in ids},
                start_frame=('frame', 'min'), end_frame=('frame', 'max'))
            intervals = intervals.loc[intervals.end_frame - intervals.start_frame + 1 >= query.min_consecutive_frames]
            results = [QueryResult(tuple(int(v) for v in row[:len(ids)]), int(row[-2]), int(row[-1]))
                       for row in intervals.itertuples(index=False, name=None)]
        temporal_finished = time.perf_counter()
        results.sort(key=lambda r: (-r.duration_frames, r.binding, r.start_frame, r.end_frame))
        returned = results if return_all else results[:query.topk]
        if stats is not None:
            stats.update(mv_lookup_s=views_finished-started,
                         spatial_join_filter_s=spatial_finished-views_finished,
                         temporal_merge_s=temporal_finished-spatial_finished,
                         topk_s=time.perf_counter()-temporal_finished,
                         qualifying_intervals=len(results),
                         source_object_frames=len(self._manager.source),
                         query_dependent_views_retained=False)
        return returned

    def release(self):
        self._manager = None
        self._frames = np.array([], dtype='int64')
        self.artifact = None
        self.metadata = None


__all__ = ['EVASymbolicMVStrategy', 'MaterializedFeatureManager', 'PredicateDomain', 'UDFSignature']

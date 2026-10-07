"""Experimental MLI type-directory variant of the frozen prepared executor.

The additional directory stores only references to existing per-window pair
keys, grouped by endpoint types and direction. It contains no spatial or frame
postings. Query constraints are compiled lazily inside the measured candidate
phase and discarded after that query. Joining, raw verification, fallback,
interval merging and ranking are inherited without modification.
"""
from collections import defaultdict
from contextvars import ContextVar
import pickle
import time

from vsimsearch.prepared_mli_querying import PreparedMLIQueryEngine, _PickleByteCounter


class FastPreparedMLIQueryEngine(PreparedMLIQueryEngine):
    def __init__(self, *args, **kwargs):
        started = time.perf_counter()
        super().__init__(*args, **kwargs)
        directory_started = time.perf_counter()
        self._typed_pair_directories = {}
        self._query_bins = ContextVar('mli_query_bin_compilation', default=None)
        directory_entries = pair_keys = 0
        for pairs, types, _ in self.windows.values():
            if id(pairs) in self._typed_pair_directories:
                continue
            groups = defaultdict(list)
            for pair in pairs:
                source, target = pair
                pair_keys += 1
                for source_type in types.get(source, ()):
                    for target_type in types.get(target, ()):
                        groups[(source_type, target_type, False)].append(pair)
                        directory_entries += 1
                        if self.legacy:
                            groups[(target_type, source_type, True)].append(pair)
                            directory_entries += 1
            self._typed_pair_directories[id(pairs)] = dict(groups)
        measure_started = time.perf_counter()
        counter = _PickleByteCounter()
        pickle.Pickler(counter, protocol=pickle.HIGHEST_PROTOCOL).dump(self._typed_pair_directories)
        directory_size_s = time.perf_counter() - measure_started
        base_payload_bytes = self.metadata['prepared_auxiliary_pickle_bytes']
        self.metadata.update(
            engine='prepared_mli_window_type_directory_with_raw_candidate_verification',
            typed_pair_directory_entries=directory_entries,
            typed_pair_directory_unique_pairs=pair_keys,
            typed_pair_directory_pickle_bytes=counter.size,
            typed_pair_directory_scope='per-window endpoint-type/direction groups of original pair keys only; no bins or frame postings',
            typed_pair_directory_setup_s=time.perf_counter()-directory_started,
            typed_pair_directory_size_measurement_s=directory_size_s,
            base_prepared_auxiliary_pickle_bytes=base_payload_bytes,
            prepared_auxiliary_pickle_bytes=base_payload_bytes+counter.size,
            prepared_auxiliary_pickle_scope=self.metadata['prepared_auxiliary_pickle_scope']+'; plus separately serialized type-pair directory payload',
            size_measurement_s=self.metadata['size_measurement_s']+directory_size_s,
            setup_s=time.perf_counter()-started,
        )

    def _compile_allowed_bins(self, constraint):
        distances = {stored for stored, actual in self.distance_options.items()
                     if any(constraint.d_min <= d <= constraint.d_max for d in actual)}
        compiled = []
        for options in (self.theta_options, self.reverse_theta_options):
            angles = {stored for stored, actual in options.items()
                      if not actual.isdisjoint(constraint.theta_bins)}
            compiled.append(frozenset((angle, distance) for angle in angles for distance in distances))
        return tuple(compiled)

    def _relation_candidates(self, pairs, types, query, roles, constraint):
        # This first-use compilation runs inside the superclass's candidate
        # timer. Context-local lifetime also keeps concurrent queries separate.
        cache = self._query_bins.get()
        if cache is None:
            allowed = self._compile_allowed_bins(constraint)
        else:
            if constraint not in cache:
                cache[constraint] = self._compile_allowed_bins(constraint)
            allowed = cache[constraint]
        groups = self._typed_pair_directories[id(pairs)]
        source_role, target_role = roles
        source_type, target_type = query.role_types[source_role], query.role_types[target_role]
        result = {}
        for reverse in ((False, True) if self.legacy else (False,)):
            accepted_bins = allowed[reverse]
            if not accepted_bins:
                continue
            for pair in groups.get((source_type, target_type, reverse), ()):
                matched = set()
                for td, frames in pairs[pair].items():
                    if td in accepted_bins:
                        matched.update(frames)
                if matched:
                    source, target = pair
                    binding = [None]*len(query.role_types)
                    binding[source_role], binding[target_role] = ((int(target), int(source))
                        if reverse else (int(source), int(target)))
                    result.setdefault(tuple(binding), set()).update(matched)
        return result

    def query(self, query, *, stats=None, return_all=False):
        token = self._query_bins.set({})
        try:
            return super().query(query, stats=stats, return_all=return_all)
        finally:
            self._query_bins.reset(token)

    def release(self):
        self._typed_pair_directories.clear()
        super().release()

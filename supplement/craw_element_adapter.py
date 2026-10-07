"""Official Craw MSU/element-index components with a conservative QST verifier.

This is NOT Craw's full T/I/F similarity retrieval. Source files are loaded
unchanged; segmentation lookup acceleration preserves the native decisions.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from supplement.corrected_baseline_strategies import CorrectedBaselineStrategy


DEFAULT_CRAW_ROOT = Path('D:/pycharm/_baseline_prepare_20260919/Craw_Submission-source/sys')
NATIVE_FILES = ('utils/logger.py', 'config/settings.py', 'data/frame_parser.py',
                'data/msu_splitter.py', 'storage/inverted_index.py')


@lru_cache(maxsize=1)
def load_native_craw(root: str = str(DEFAULT_CRAW_ROOT)):
    """Temporarily resolve upstream absolute imports, then restore the host imports.

    Intended for this single-threaded benchmark; no modules named config/data/utils
    are left installed globally by this loader.
    """
    root_path = Path(root)
    for relative in NATIVE_FILES:
        if not (root_path / relative).is_file():
            raise FileNotFoundError(root_path / relative)
    namespace = '_qst_craw_native'
    aliases = ('utils', 'utils.logger', 'config', 'config.settings')
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        for suffix in ('', '.utils', '.config', '.data', '.storage'):
            name = namespace + suffix
            package = ModuleType(name)
            package.__path__ = [str(root_path / suffix.lstrip('.'))]
            sys.modules[name] = package
        for short in ('utils', 'config'):
            sys.modules[short] = sys.modules[f'{namespace}.{short}']
        loaded = {}
        for relative in NATIVE_FILES:
            suffix = relative[:-3].replace('/', '.')
            name = f'{namespace}.{suffix}'
            spec = importlib.util.spec_from_file_location(name, root_path / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[suffix] = module
            if suffix in aliases:
                sys.modules[suffix] = module
        return SimpleNamespace(MSUSplitter=loaded['data.msu_splitter'].MSUSplitter,
                               MSUConfig=loaded['config.settings'].MSUConfig,
                               InvertedIndex=loaded['storage.inverted_index'].InvertedIndex)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def fast_splitter_class(native):
    class FastLookupSplitter(native.MSUSplitter):
        def split(self, frame_info):
            self._frame_ids = {int(frame): {int(row['id']) for row in rows}
                               for frame, rows in frame_info.items()}
            return super().split(frame_info)

        def _id_set_at_frame(self, frame):
            return self._frame_ids.get(frame, set()).intersection(self.filtered_ids)

        def _build_duration_map(self, start, end):
            result = {}
            # Preserve upstream object insertion order (including histogram ties).
            for object_id, frames in self._obj_frames.items():
                if object_id not in self.filtered_ids:
                    continue
                left, right = bisect_left(frames, start), bisect_right(frames, end)
                if left < right:
                    result[object_id] = (frames[left], frames[right - 1])
            return result

    return FastLookupSplitter


class CrawElementStrategy(CorrectedBaselineStrategy):
    artifact_kind = 'official-craw-msu-element-index-plus-exact-verifier'

    def __init__(self, data, frame_width, frame_height, *, artifact_dir,
                 source_root=DEFAULT_CRAW_ROOT):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if (self.artifact_dir / 'inverted_index.json').exists():
            raise FileExistsError('Refusing to reuse stale Craw postings in artifact_dir')
        self.source_root = Path(source_root)
        super().__init__(data, frame_width, frame_height)

    def _build_artifact(self):
        native = load_native_craw(str(self.source_root))
        frame_info = {}
        frame_classes = {}
        for row in self._data.itertuples(index=False):
            frame = int(row.frame)
            frame_info.setdefault(str(frame), []).append({'id': int(row.track_id),
                                                          'cls': int(row.class_id)})
            frame_classes.setdefault(frame, set()).add(int(row.class_id))
        splitter = fast_splitter_class(native)()
        ranges, _ = splitter.split(frame_info)
        del frame_info, splitter
        ordered_frames = sorted(frame_classes)
        index = native.InvertedIndex(str(self.artifact_dir))
        msu_frames = {}
        covered = set()
        for msu_id, (start, end) in enumerate(ranges):
            frames = tuple(ordered_frames[bisect_left(ordered_frames, start):
                                          bisect_right(ordered_frames, end)])
            msu_frames[msu_id] = frames
            covered.update(frames)
            # Conservative F:class posting: retain every observed class, even if
            # short tracks were excluded from the splitter's guidance statistics.
            classes = set().union(*(frame_classes[frame] for frame in frames))
            index.update(msu_id, [[class_id] for class_id in sorted(classes)], [])
        if covered != set(ordered_frames):
            raise RuntimeError('Native Craw partition failed to cover the supplied sampled frames')
        (self.artifact_dir / 'msu_ranges.json').write_text(json.dumps(ranges), encoding='utf-8')
        if not (self.artifact_dir / 'inverted_index.json').is_file():
            raise OSError('Native Craw failed to persist its inverted index')
        return SimpleNamespace(index=index, msu_frames=msu_frames, ranges=ranges)

    def _candidates(self, query):
        msus = self.artifact.index.query_and('F', sorted(set(query.role_types)))
        # Verify globally so an event is not split at artificial MSU boundaries.
        return None, {frame for msu in msus for frame in self.artifact.msu_frames[msu]}

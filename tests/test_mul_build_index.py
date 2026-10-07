import unittest
from pathlib import Path

from vsimsearch import mul_build_index


class MulBuildIndexPathTests(unittest.TestCase):
    def test_discretization_tag_uses_theta_and_distance_parts(self):
        self.assertEqual(
            mul_build_index.discretization_tag(10, 8),
            "theta10_d8",
        )

    def test_artifact_paths_use_optional_output_tag(self):
        index_path, cp_graphs_path, windowid_index_path, time_path = mul_build_index.artifact_paths(
            "drtest",
            0.7,
            output_tag="theta10_d8",
            base_dir=Path("storage/index"),
        )

        self.assertEqual(index_path, Path("storage/index/drtest_0.7_theta10_d8_index.pkl"))
        self.assertEqual(cp_graphs_path, Path("storage/index/drtest_0.7_theta10_d8_cp_graphs.pkl"))
        self.assertEqual(windowid_index_path, Path("storage/index/drtest_0.7_theta10_d8_windowid_index.pkl"))
        self.assertEqual(time_path, Path("storage/index/build_cp_graph_time/drtest_0.7_theta10_d8_time.txt"))

    def test_slice_window_frame_objects_converts_1_based_key_frames(self):
        frames = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
        key_frames = [1, 4, 7]

        self.assertEqual(
            mul_build_index.slice_window_frame_objects(frames, key_frames, 0),
            ["f1", "f2", "f3"],
        )
        self.assertEqual(
            mul_build_index.slice_window_frame_objects(frames, key_frames, 1),
            ["f4", "f5", "f6"],
        )
        self.assertEqual(
            mul_build_index.slice_window_frame_objects(frames, key_frames, 2),
            ["f7", "f8"],
        )


if __name__ == "__main__":
    unittest.main()

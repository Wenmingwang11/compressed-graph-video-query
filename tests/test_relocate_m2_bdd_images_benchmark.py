import tempfile
import unittest
import zipfile
from pathlib import Path

from supplement.relocate_m2_bdd_images_benchmark import (
    BDD_CATEGORY_TO_CLASS,
    crop_similarity,
    parse_pair_constraint,
    read_bdd_frame_detections,
)


class RelocateM2BddImagesBenchmarkTests(unittest.TestCase):
    def test_parse_pair_constraint(self):
        constraint = parse_pair_constraint("0-1:10,8")
        self.assertEqual((constraint.left_role, constraint.right_role), (0, 1))
        self.assertEqual((constraint.theta_lt, constraint.d_ratio_lt), (10.0, 8.0))

    def test_crop_similarity_identity_is_high(self):
        from PIL import Image

        image = Image.new("RGB", (32, 32), (10, 120, 200))
        score = crop_similarity(image, image)
        self.assertGreater(score, 0.99)

    def test_read_bdd_frame_detections_filters_allowed_classes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            image_root = tmp / "images"
            image_root.mkdir()
            (image_root / "sample.jpg").write_bytes(b"fake")

            label_zip = tmp / "labels.zip"
            payload = {
                "name": "sample",
                "frames": [
                    {
                        "timestamp": 10000,
                        "objects": [
                            {
                                "category": "car",
                                "id": 1,
                                "box2d": {"x1": 1, "y1": 2, "x2": 11, "y2": 22},
                            },
                            {
                                "category": "traffic sign",
                                "id": 2,
                                "box2d": {"x1": 21, "y1": 22, "x2": 31, "y2": 42},
                            },
                            {
                                "category": "bus",
                                "id": 3,
                                "box2d": {"x1": 41, "y1": 42, "x2": 71, "y2": 82},
                            },
                        ],
                    }
                ],
                "attributes": {},
            }
            with zipfile.ZipFile(label_zip, "w") as zf:
                zf.writestr("100k/train/sample.json", __import__("json").dumps(payload))

            frames, detections = read_bdd_frame_detections(
                image_root=image_root,
                label_zip_path=label_zip,
                allowed_classes={
                    BDD_CATEGORY_TO_CLASS["car"],
                    BDD_CATEGORY_TO_CLASS["traffic sign"],
                },
            )

            self.assertEqual(len(frames), 1)
            self.assertEqual(len(detections), 2)
            self.assertEqual(sorted(det.cls for det in detections), [2, 82])


if __name__ == "__main__":
    unittest.main()

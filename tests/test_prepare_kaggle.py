from __future__ import annotations

import csv
import dataclasses
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Respect TMPDIR on the server, but also run on CI and developer machines.
_TEMP = tempfile.TemporaryDirectory(prefix="plates-kaggle-tests-")
TMP_ROOT = Path(_TEMP.name)


def ktemp(prefix: str) -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=TMP_ROOT))

from prepare_kaggle import (
    PreparationError,
    Sample,
    assign_splits,
    build,
    discover,
    parse_cvat,
    group_similar,
    yaml_quote,
)

from PIL import Image


def write_png(path: Path, width: int, height: int, value: int = 200) -> None:
    Image.new("RGB", (width, height), (value, 0, 0)).save(path)


def make_root(xml: str, images: list[str]) -> Path:
    root = ktemp("kaggle-test-")
    (root / "photos" / "photos").mkdir(parents=True)
    for name in images:
        write_png(root / "photos" / "photos" / name, 80, 60)
    (root / "annotations.xml").write_text(xml, encoding="utf-8")
    return root


class ParseCvatTest(unittest.TestCase):
    def test_valid_one_and_multiple_boxes(self) -> None:
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<annotations><image id=\"0\" name=\"101.jpg\" width=\"80\" height=\"60\">"
            "<box label=\"plate\" xtl=\"10\" ytl=\"10\" xbr=\"30\" ybr=\"30\" z_order=\"0\">"
            "<attribute name=\"plate number\">XYZ12</attribute></box>"
            "<box label=\"plate\" xtl=\"40\" ytl=\"40\" xbr=\"70\" ybr=\"55\" z_order=\"1\"/>"
            "</image></annotations>"
        )
        root = make_root(xml, ["101.jpg"])
        xml_path, photos, _ = discover(root)
        samples, _ = parse_cvat(xml_path, photos)
        self.assertEqual(len(samples), 1)
        image = samples[0]
        self.assertEqual(image.boxes, ((10.0, 10.0, 30.0, 30.0), (40.0, 40.0, 70.0, 55.0)))
        lines = image.yolo_lines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            class_id, cx, cy, w, h = line.split()
            self.assertEqual(class_id, "0")
            for value in (float(cx), float(cy), float(w), float(h)):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_box_outside_image_raises(self) -> None:
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<annotations><image id=\"0\" name=\"202.jpg\" width=\"80\" height=\"60\">"
            "<box label=\"plate\" xtl=\"-5\" ytl=\"10\" xbr=\"40\" ybr=\"30\" z_order=\"0\"/>"
            "</image></annotations>"
        )
        root = make_root(xml, ["202.jpg"])
        xml_path, photos, _ = discover(root)
        with self.assertRaises(PreparationError):
            parse_cvat(xml_path, photos)

    def test_box_inverted_or_degenerate_raises(self) -> None:
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<annotations><image id=\"0\" name=\"203.jpg\" width=\"80\" height=\"60\">"
            "<box label=\"plate\" xtl=\"40\" ytl=\"10\" xbr=\"10\" ybr=\"30\" z_order=\"0\"/>"
            "</image></annotations>"
        )
        root = make_root(xml, ["203.jpg"])
        xml_path, photos, _ = discover(root)
        with self.assertRaises(PreparationError):
            parse_cvat(xml_path, photos)

    def test_missing_image_raises(self) -> None:
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<annotations><image id=\"0\" name=\"303.jpg\" width=\"80\" height=\"60\">"
            "<box label=\"plate\" xtl=\"10\" ytl=\"10\" xbr=\"30\" ybr=\"30\" z_order=\"0\"/>"
            "</image></annotations>"
        )
        root = make_root(xml, ["303.jpg"])
        (root / "photos" / "photos" / "303.jpg").unlink(missing_ok=True)
        with self.assertRaises(PreparationError):
            xml_path, photos, _ = discover(root)
            parse_cvat(xml_path, photos)

    def test_image_without_box_is_kept_as_unannotated_not_raised(self) -> None:
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<annotations>"
            '<image id="0" name="404.jpg" width="80" height="60"/>'
            '<image id="1" name="405.jpg" width="80" height="60">'
            "<box label=\"plate\" xtl=\"10\" ytl=\"10\" xbr=\"30\" ybr=\"30\" z_order=\"0\"/>"
            "</image></annotations>"
        )
        root = make_root(xml, ["404.jpg", "405.jpg"])
        xml_path, photos, _ = discover(root)
        samples, unannotated = parse_cvat(xml_path, photos)
        self.assertEqual(unannotated, ["404.jpg"])
        self.assertEqual(len(samples), 2)
        labeled = [s for s in samples if s.boxes]
        self.assertEqual([s.image_id for s in labeled], ["405.jpg"])

    def test_per_box_texts_are_kept_separately(self) -> None:
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<annotations><image id=\"0\" name=\"501.jpg\" width=\"80\" height=\"60\">"
            '<box label="plate" xtl="10" ytl="10" xbr="30" ybr="30" z_order="0">'
            "<attribute name=\"plate number\">AAA</attribute></box>"
            '<box label="plate" xtl="40" ytl="40" xbr="70" ybr="55" z_order="1">'
            "<attribute name=\"plate number\">BBB</attribute>"
            "<attribute name=\"color\">blue</attribute></box>"
            "</image></annotations>"
        )
        root = make_root(xml, ["501.jpg"])
        xml_path, photos, _ = discover(root)
        samples, _ = parse_cvat(xml_path, photos)
        self.assertEqual(samples[0].plate_texts, ("AAA", "BBB"))

    def test_dimension_mismatch_with_real_image_raises(self) -> None:
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<annotations><image id="0" name="601.jpg" width="64" height="60"/></annotations>'
        )
        root = make_root(xml, ["601.jpg"])  # real image is 80x60
        xml_path, photos, _ = discover(root)
        with self.assertRaises(PreparationError):
            parse_cvat(xml_path, photos)

    def test_bad_image_name_and_stem_collision_raise(self) -> None:
        for names, case in (
            (["a.jpg", "b.jpg"], '<image id="0" name="../x.jpg" width="80" height="60"/>'),
            (["a.jpg", "A.jpg"], '<image id="0" name="a.jpg" width="80" height="60"/><image id="1" name="A.jpg" width="80" height="60"/>'),
            (["c.jpg", "d.jpg"], '<image id="0" name="x%2F.jpg" width="80" height="60"/>'),
        ):
            xml = '<?xml version="1.0" encoding="utf-8"?>'
            xml += f"<annotations>{case}</annotations>"
            root = make_root(xml, names)
            xml_path, photos, _ = discover(root)
            with self.assertRaises(PreparationError):
                parse_cvat(xml_path, photos)

    def test_discover_ignores_non_images_and_rejects_empty_dirs(self) -> None:
        root = ktemp("kaggle-disc-")
        xml = '<?xml version="1.0" encoding="utf-8"?>' "<annotations></annotations>"
        # real photos + a non-image file elsewhere in the tree
        (root / "photos" / "photos").mkdir(parents=True)
        write_png(root / "photos" / "photos" / "1.jpg", 80, 60)
        (root / "extra").mkdir(parents=True)
        write_png(root / "extra" / "note.png", 10, 10)
        (root / "annotations.xml").write_text(xml, encoding="utf-8")
        xml_path, photos, version = discover(root)
        self.assertTrue(xml_path.is_file())
        # relative-path keys are registered; non-image junk is kept if it has an image suffix
        self.assertEqual(set(photos), {"photos/photos/1.jpg", "1.jpg", "extra/note.png", "note.png"})
        self.assertEqual(version, "")
        # no xml -> error
        empty = ktemp("kaggle-disc2-")
        with self.assertRaises(PreparationError):
            discover(empty)
        # only non-image files -> error
        junk = ktemp("kaggle-disc3-")
        (junk / "annotations.xml").write_text(xml, encoding="utf-8")
        (junk / "data.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(PreparationError):
            discover(junk)

    def test_discover_rejects_duplicate_basenames(self) -> None:
        root = make_root("<annotations/>", ["1.jpg"])
        (root / "other").mkdir()
        write_png(root / "other" / "1.jpg", 80, 60)
        with self.assertRaises(PreparationError):
            discover(root)


def similar_samples() -> list[Sample]:
    # With seed 3 the bucket per group is: a1->64(val), b1->51(train),
    # group(c1,c2) (Hamming distance 0)->6(train), x5->71(test),
    # so every split is non-empty and the similar pair stays together.
    return [
        sample("a1.png", 80, 60, ((10.0, 10.0, 30.0, 30.0),), ("t1",), 0, "aa"),
        sample("b1.png", 80, 60, ((10.0, 10.0, 30.0, 30.0),), ("t2",), 15, "bb"),
        sample("c1.png", 80, 60, ((10.0, 10.0, 30.0, 30.0),), ("t3",), 51, "cc"),
        sample("c2.png", 80, 60, ((10.0, 10.0, 30.0, 30.0),), ("t4",), 51, "dd"),
        sample("x5.png", 80, 60, ((10.0, 10.0, 30.0, 30.0),), ("t5",), 60, "ee"),
    ]


def sample(image_id: str, width: int, height: int, boxes, texts: tuple[str, ...], dhash: int, digest: str) -> Sample:
    return Sample(image_id, Path(image_id), width, height, boxes, texts, dhash, digest)


class SplitAssignmentTest(unittest.TestCase):
    def _run(self, samples: list[Sample], seed: int) -> dict[str, str]:
        return assign_splits(samples, group_similar(samples), seed)

    def test_groups_stay_together_and_split_buckets_cover_train_val_test(self) -> None:
        samples = similar_samples()
        first = self._run(samples, 3)
        second = self._run(samples, 3)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "val", "test"})
        # Similar images (c1/c2) must share the same split, never leak apart.
        self.assertEqual(first["c1.png"], first["c2.png"])
        for value in first.values():
            self.assertIn(value, ("train", "val", "test"))


def build_dataset(out_dir: Path, samples: list[Sample], assignments: dict[str, str], seed: int, raw: Path) -> None:
    with redirect_stdout(io.StringIO()):
        build(out_dir, samples, assignments, [], seed, raw, "6")


def explicit_assignments(samples: list[Sample]) -> dict[str, str]:
    # One hand-chosen 3-split assignment (assign_splits is tested separately),
    # so build tests do not depend on the hash-based split buckets.
    by_id = {s.image_id: s for s in samples}
    return {
        "a1.png": "val",
        "b1.png": "train",
        "c1.png": "train",
        "c2.png": "train",
        "x5.png": "test",
    }


def make_raw_samples(samples: list[Sample]) -> Path:
    root = ktemp("kaggle-raw-") / "raw"
    root.mkdir(parents=True)
    for s in samples:
        write_png(root / s.image_id, 80, 60)
    images_root = root.parent
    return images_root


class BuildTest(unittest.TestCase):
    def test_build_layout_manifest_and_no_overwrite(self) -> None:
        samples = similar_samples()
        images_root = make_raw_samples(samples)
        raw = images_root / "raw"
        # Real on-disk sources (a1.png etc.) instead of the raw-dir basename paths:
        samples = [dataclasses.replace(s, image_path=raw / s.image_id) for s in samples]
        out_dir = ktemp("kaggle-out-")
        build_dataset(out_dir, samples, explicit_assignments(samples), 3, raw)
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                files = [f for f in (out_dir / kind / split).iterdir() if f.is_file()]
                self.assertGreaterEqual(len(files), 0)
        # Every image appears exactly once across all splits (no leakage).
        all_images = {
            f.name
            for split in ("train", "val", "test")
            for f in (out_dir / "images" / split).iterdir()
            if f.is_file()
        }
        self.assertEqual(all_images, {s.image_id for s in samples})
        # Label files mirror images: one per image, named by stem.
        all_labels = {
            f.name
            for split in ("train", "val", "test")
            for f in (out_dir / "labels" / split).iterdir()
            if f.is_file()
        }
        self.assertEqual(
            all_labels,
            {Path(s.image_id).stem + ".txt" for s in samples},
        )
        self.assertTrue((out_dir / "manifest.json").is_file())
        self.assertTrue((out_dir / "gt.csv").is_file())
        self.assertTrue((out_dir / "SOURCE.md").is_file())
        yaml_text = (out_dir / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("train: images/train", yaml_text)
        self.assertIn("val: images/val", yaml_text)
        self.assertIn("test: images/test", yaml_text)
        self.assertIn("names:\n  0: plate", yaml_text)
        # Path is a quoted YAML scalar (json.dumps, ensure_ascii=False).
        self.assertIn(f"path: {yaml_quote(str(out_dir))}", yaml_text)
        # gt.csv rows: one per box, texts verbatim, proper csv quoting.
        with (out_dir / "gt.csv").open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["image", "width", "height", "xmin", "ymin", "xmax", "ymax", "plate_text"])
        self.assertEqual(len(rows) - 1, 5)
        by_image: dict[str, list[str]] = {}
        for row in rows[1:]:
            by_image.setdefault(row[0], []).append(row[7])
        self.assertEqual(by_image,
            {s.image_id: list(s.plate_texts) for s in samples})
        # Non-empty pre-existing output must be refused, not overwritten.
        pre = ktemp("kaggle-pre-") / "data"
        pre.mkdir(parents=True)
        (pre / "marker.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(PreparationError):
            build_dataset(pre, samples, explicit_assignments(samples), 3, raw)
        self.assertTrue((pre / "marker.txt").read_text(encoding="utf-8") == "keep")
        # Output directory equal to the input directory must be refused.
        with self.assertRaises(PreparationError):
            build_dataset(raw, samples, explicit_assignments(samples), 3, raw)


class DeterminismTest(unittest.TestCase):
    def test_same_seed_gives_identical_manifest(self) -> None:
        samples = similar_samples()
        images_root = make_raw_samples(samples)
        raw = images_root / "raw"
        samples = [dataclasses.replace(s, image_path=raw / s.image_id) for s in samples]
        assignments = explicit_assignments(samples)
        first = ktemp("kaggle-det-a-")
        second = ktemp("kaggle-det-b-")
        build_dataset(first, samples, assignments, 1, raw)
        build_dataset(second, samples, assignments, 1, raw)
        self.assertEqual(
            (first / "manifest.json").read_text(encoding="utf-8"),
            (second / "manifest.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            json.loads((first / "manifest.json").read_text()),
            json.loads((second / "manifest.json").read_text()),
        )
        self.assertEqual(
            (first / "gt.csv").read_text(encoding="utf-8"),
            (second / "gt.csv").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()

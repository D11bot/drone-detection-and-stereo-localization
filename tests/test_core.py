"""Tests for box conversion and time-aligned stereo localization."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_anti_uav import bbox_xywh_to_yolo
from flight01_gps_localization import (
    Config,
    localize,
    optional_image_previews,
    right_observation_at,
    split_name,
)


class BoundingBoxTests(unittest.TestCase):
    def test_box_inside_image(self):
        self.assertEqual(bbox_xywh_to_yolo([20, 10, 40, 30], 200, 100),
                         (0.2, 0.25, 0.2, 0.3))

    def test_box_partly_outside_image(self):
        # Clip the visible box before computing YOLO center and size.
        xc, yc, w, h = bbox_xywh_to_yolo([180, 20, 40, 50], 200, 100)
        self.assertAlmostEqual(xc, .95)
        self.assertAlmostEqual(yc, .45)
        self.assertAlmostEqual(w, .10)
        self.assertAlmostEqual(h, .50)
        self.assertLessEqual(xc + w / 2, 1.0)

    def test_box_fully_outside_or_invalid(self):
        for box in ([250, 0, 30, 10], [-40, 0, 20, 10],
                    [10, 10, -5, 20], [math.nan, 0, 5, 5]):
            with self.subTest(box=box):
                self.assertIsNone(bbox_xywh_to_yolo(box, 200, 100))


class StereoAlignmentTests(unittest.TestCase):
    @staticmethod
    def fixture():
        # Synthetic fixture with a 1 m stereo baseline and a target 10 m ahead.
        K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
        models = {
            'left': {'K': K, 'R_world_to_camera': np.eye(3),
                     't_world_to_camera': np.array([0., 0., 0.]),
                     'export_to_srt_s': 0.0},
            'right': {'K': K, 'R_world_to_camera': np.eye(3),
                      't_world_to_camera': np.array([-1., 0., 0.]),
                      'export_to_srt_s': -1.0},
        }
        telemetry = {
            'grid': np.array([129., 130., 131., 132.]),
            'world': np.array([[0., 0., 10.]] * 4),
        }
        left_hit = {'center_x_px': '50.0', 'center_y_px': '50.0'}
        right_hit = {'center_x_px': '40.0', 'center_y_px': '50.0'}
        pairs = [
            {
                'recording_id': 'synthetic', 'frame_index': 3900,
                'left_elapsed_time_s': 130.0, 'right_elapsed_time_s': 130.0,
                'left_filename': 'left_130.jpg', 'right_filename': 'right_130.jpg',
                'left_detections': [left_hit], 'right_detections': [],
                'pair_status': 'missing_detection',
            },
            {
                'recording_id': 'synthetic', 'frame_index': 3930,
                'left_elapsed_time_s': 131.0, 'right_elapsed_time_s': 131.0,
                'left_filename': 'left_131.jpg', 'right_filename': 'right_131.jpg',
                'left_detections': [], 'right_detections': [right_hit],
                'pair_status': 'missing_detection',
            },
        ]
        return pairs, models, telemetry

    def test_uses_right_observation_at_corrected_time(self):
        pairs, models, telemetry = self.fixture()
        rows, baseline = localize(pairs, models, telemetry, Config())
        self.assertEqual(rows[0]['status'], 'gps_calibrated_exploratory')
        self.assertEqual(rows[0]['right_count'], 0)  # No nominal right detection.
        self.assertEqual(rows[0]['right_source_frame_indices'], '3930')
        self.assertEqual(rows[0]['right_alignment_method'],
                         'nearest_detection_with_residual_time_error')
        self.assertAlmostEqual(rows[0]['estimate_east_m'], 0., places=6)
        self.assertAlmostEqual(rows[0]['estimate_dji_relative_height_m'], 10., places=6)
        self.assertAlmostEqual(rows[0]['label_difference_3d_m'], 0., places=6)
        self.assertAlmostEqual(baseline, 1.)
        self.assertEqual(rows[1]['status'], 'missing_left_detection')

    def test_right_interpolation_stays_in_validation(self):
        cfg = Config()
        track = [
            {'time': 129., 'xy': np.array([20., 50.]), 'frame': 100, 'split': 'validation'},
            {'time': 130., 'xy': np.array([30., 50.]), 'frame': 101, 'split': 'validation'},
            {'time': 130.5, 'xy': np.array([99., 50.]), 'frame': 102, 'split': 'test'},
        ]
        aligned = right_observation_at(129.5, track, cfg, 'validation')
        self.assertEqual(aligned['method'], 'linear_interpolation_of_two_observed_detections')
        self.assertAlmostEqual(aligned['xy'][0], 25.)
        self.assertIsNone(right_observation_at(130.5, track[:2], cfg, 'test'))

    def test_buffer_is_excluded(self):
        cfg = Config()
        self.assertEqual(split_name(119.0, cfg), 'calibration')
        self.assertEqual(split_name(120.0, cfg), 'buffer')
        self.assertEqual(split_name(124.0, cfg), 'buffer')
        self.assertEqual(split_name(125.0, cfg), 'validation')
        self.assertEqual(split_name(166.0, cfg), 'test')

    def test_preview_with_missing_nominal_right_detection(self):
        import cv2
        pairs, models, telemetry = self.fixture()
        rows, _ = localize(pairs, models, telemetry, Config())
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'frames'
            out = Path(temp_dir) / 'previews'
            out.mkdir()
            for camera in ('left', 'right'):
                image_dir = root / camera / 'images'
                image_dir.mkdir(parents=True)
                (root / camera / 'frames.csv').write_text('placeholder\n', encoding='utf-8')
                for pair in pairs:
                    cv2.imwrite(str(image_dir / pair[camera + '_filename']),
                                np.zeros((100, 100, 3), dtype=np.uint8))
            cfg = Config(export_root=str(root))
            result = optional_image_previews(pairs, rows, cfg, {}, out)
            self.assertEqual(result, 'previews_created_from_original_export_images')
            self.assertEqual(len(list(out.glob('*.jpg'))), 1)


if __name__ == '__main__':
    unittest.main()

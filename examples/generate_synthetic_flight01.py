"""Generate synthetic stereo observations for testing camera localization.

The generator projects a 3D trajectory into two simulated cameras and writes
detections, telemetry, and metadata in the localization script's input format.
"""

import argparse
import csv
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import random


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FOCAL_PX = 1320.0
START = datetime(2026, 1, 1, 10, 0, 0)
BASE_LATITUDE = 0.0
BASE_LONGITUDE = 0.0
EARTH_RADIUS = 6371008.8


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def unit(v):
    length = math.sqrt(dot(v, v))
    return tuple(x / length for x in v)


def camera_axes(center, target):
    """Orient a pinhole camera toward the center of the synthetic trajectory."""
    forward = unit(sub(target, center))
    right = unit(cross(forward, (0., 0., 1.)))
    down = cross(forward, right)
    return right, down, forward


def trajectory(t):
    # Combine periodic components to make calibration points non-planar.
    east = 7.5 * math.sin(2 * math.pi * t / 73) + .025 * t
    north = 8.0 * (math.cos(2 * math.pi * t / 61) - 1) + .012 * t
    height = 12 + 3.5 * math.sin(2 * math.pi * t / 47) + .008 * t
    return east, north, height


def project(point, center, axes):
    delta = sub(point, center)
    x, y, z = (dot(axis, delta) for axis in axes)
    if z <= 0:
        raise ValueError('My synthetic point is behind the camera')
    return (FOCAL_PX * x / z + (IMAGE_WIDTH - 1) / 2,
            FOCAL_PX * y / z + (IMAGE_HEIGHT - 1) / 2)


def format_dms(decimal_degrees, positive, negative):
    direction = positive if decimal_degrees >= 0 else negative
    value = abs(decimal_degrees)
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    seconds = (value - degrees - minutes / 60) * 3600
    return f'{degrees} deg {minutes}\' {seconds:.6f}" {direction}'


def write_srt(path, duration_seconds):
    lines = []
    for index in range(duration_seconds * 10 + 1):
        t = index / 10
        east, north, height = trajectory(t)
        latitude = BASE_LATITUDE + math.degrees(north / EARTH_RADIUS)
        longitude = BASE_LONGITUDE + math.degrees(east / EARTH_RADIUS)
        timestamp = START + timedelta(seconds=t)
        lines.extend([
            str(index + 1),
            timestamp.strftime('%H:%M:%S,') + f'{timestamp.microsecond // 1000:03d}'
            + ' --> ' + (timestamp + timedelta(milliseconds=100)).strftime('%H:%M:%S,%f')[:12],
            timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            f'latitude: {latitude:.10f}',
            f'longitude: {longitude:.10f}',
            f'rel_alt: {height:.6f}',
            '',
        ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_metadata(path, left_center, right_center):
    def section(name, center):
        # Create synthetic camera GPS tags for the metadata format.
        latitude = BASE_LATITUDE + math.degrees(center[1] / EARTH_RADIUS)
        longitude = BASE_LONGITUDE + math.degrees(center[0] / EARTH_RADIUS)
        return (f'{name.capitalize()} camera:\n'
                f'File Name: synthetic_{name}.mov\n'
                f'Image Width: {IMAGE_WIDTH}\nImage Height: {IMAGE_HEIGHT}\n'
                'Camera Focal Length 35mm Equivalent: 26\n'
                f'Creation Date: {START.strftime("%Y:%m:%d %H:%M:%S%z")}+0000\n'
                f'GPS Latitude: {format_dms(latitude, "N", "S")}\n'
                f'GPS Longitude: {format_dms(longitude, "E", "W")}\n'
                f'GPS Altitude: {center[2]:.2f} m\n'
                'Location Accuracy Horizontal: 1\n\n')

    path.write_text(section('right', right_center) + section('left', left_center)
                    + 'After syncing:\nSynthetic input only.\n', encoding='utf-8')


def write_detections(path, left_center, right_center, rng):
    target = (0., -4., 12.)
    cameras = {'left': (left_center, camera_axes(left_center, target)),
               'right': (right_center, camera_axes(right_center, target))}
    fields = ['recording_id', 'camera_id', 'filename', 'frame_index',
              'elapsed_time_s', 'width', 'height', 'center_x_px', 'center_y_px']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for second in range(208):
            xyz = trajectory(float(second))
            for camera, (center, axes) in cameras.items():
                x, y = project(xyz, center, axes)
                # Omit selected observations to exercise missing-view cases.
                missing = ((camera == 'left' and second % 23 == 0) or
                           (camera == 'right' and second % 19 == 0))
                writer.writerow({
                    'recording_id': 'synthetic_flight',
                    'camera_id': camera,
                    'filename': f'synthetic_{camera}_{second:04d}.jpg',
                    'frame_index': second * 30,
                    'elapsed_time_s': float(second),
                    'width': IMAGE_WIDTH,
                    'height': IMAGE_HEIGHT,
                    'center_x_px': '' if missing else f'{x + rng.gauss(0, .3):.6f}',
                    'center_y_px': '' if missing else f'{y + rng.gauss(0, .3):.6f}',
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('synthetic_flight01'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    left_center = (-3., -37., 2.)
    right_center = (3., -37., 2.)
    write_srt(args.output_dir / 'flight_telemetry.srt', duration_seconds=220)
    write_metadata(args.output_dir / 'extractedVideoMetadata.txt', left_center, right_center)
    write_detections(args.output_dir / 'predictions_tiled_merged.csv', left_center,
                     right_center, random.Random(args.seed))
    (args.output_dir / 'run_info.json').write_text(json.dumps({
        'note': 'Synthetic 2D detections generated from camera projections',
        'manifest_sha256': {},
    }, indent=2), encoding='utf-8')
    print(f'Generated synthetic inputs in {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()

"""Run full-frame and tiled drone detection on exported stereo camera images.

The script compares three detection methods, exports bounding boxes in original
image coordinates, and records timings and candidate stereo pairs.
Developed with ultralytics 8.4.166; see requirements.txt.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from collections import defaultdict
from contextlib import ExitStack
from time import perf_counter
import csv
import hashlib
import json
import math
import platform
import shutil
import stat
import zipfile
import numpy as np

MANIFEST_FIELDS = ['recording_id', 'camera_id', 'filename', 'frame_index', 'pts',
                   'time_base_num', 'time_base_den', 'video_time_s',
                   'elapsed_time_s', 'elapsed_time_us', 'width', 'height']
METHODS = ('whole', 'tiled', 'tiled_merged')
DETECTION_FIELDS = MANIFEST_FIELDS + [
    'time_reference', 'method', 'detection_status', 'detection_id',
    'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2',
    'center_x_px', 'center_y_px', 'center_x_norm', 'center_y_norm',
    'synchronization_verified', 'cross_view_correspondence_verified', 'ready_for_3d']
FRAME_FIELDS = MANIFEST_FIELDS + [
    'status', 'whole_count', 'tiled_count', 'tiled_merged_count',
    'image_load_ms', 'whole_ms', 'tiled_total_ms', 'union_merge_ms',
    'merged_total_ms', 'tile_count']
PAIR_FIELDS = ['recording_id', 'frame_index', 'pair_status',
               'left_filename', 'right_filename',
               'left_elapsed_time_s', 'right_elapsed_time_s',
               'nominal_export_time_difference_ms', 'left_count', 'right_count',
               'left_center_x_px', 'left_center_y_px',
               'right_center_x_px', 'right_center_y_px',
               'synchronization_verified', 'cross_view_correspondence_verified',
               'ready_for_3d']

DEFAULT_CONFIG = {
    'input_root': '.', 'output_root': 'outputs',
    'export_zip': None, 'left_csv': None, 'right_csv': None,
    'left_images': None, 'right_images': None, 'weights': None,
    'frame_step': 30, 'max_pairs': None, 'preview_pairs': 12,
    'device': '0', 'require_gpu': True, 'warmup_frames': 3,
    'imgsz': 1280, 'conf': 0.25, 'iou': 0.7, 'tile_size': 640,
    'tile_imgsz': 960, 'overlap': 0.2, 'merge_iou': 0.5,
    'union_iou': 0.25, 'max_det': 300,
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def read_manifest(path, expected_camera):
    """Validate manifest fields and timestamp consistency."""
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(MANIFEST_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f'Missing frames.csv columns: {path}')
        rows = [{k: r[k] for k in MANIFEST_FIELDS} for r in reader]
    if not rows:
        raise ValueError(f'Empty manifest: {path}')
    seen_indices, seen_names, recordings, origins = set(), set(), set(), []
    int_fields = ['frame_index', 'pts', 'time_base_num', 'time_base_den',
                  'elapsed_time_us', 'width', 'height']
    for r in rows:
        for key in int_fields:
            r[key] = int(r[key])
        for key in ['video_time_s', 'elapsed_time_s']:
            r[key] = float(r[key])
            if not math.isfinite(r[key]):
                raise ValueError(f'Non-finite {key}: {path}')
        name = r['filename']
        if (not name or name in {'.', '..'} or '/' in name or '\\' in name
                or ':' in name or not name.lower().endswith(('.jpg', '.jpeg', '.png'))):
            raise ValueError(f'Unsafe or unsupported image filename: {name!r}')
        if r['camera_id'] != expected_camera or not r['recording_id']:
            raise ValueError(f'Wrong camera/recording in {path}')
        if r['frame_index'] in seen_indices or name in seen_names:
            raise ValueError(f'Duplicate frame_index or filename: {path}')
        seen_indices.add(r['frame_index']); seen_names.add(name)
        recordings.add(r['recording_id'])
        if (min(r['width'], r['height'], r['time_base_num'], r['time_base_den']) <= 0
                or min(r['frame_index'], r['elapsed_time_s'], r['elapsed_time_us']) < 0):
            raise ValueError(f'Invalid dimensions/time/index: {path}')
        pts_time = r['pts'] * r['time_base_num'] / r['time_base_den']
        if abs(pts_time - r['video_time_s']) > 1e-6:
            raise ValueError(f'PTS and video_time_s disagree: {path}')
        if abs(r['elapsed_time_us'] / 1e6 - r['elapsed_time_s']) > 1.1e-6:
            raise ValueError(f'Elapsed seconds/microseconds disagree: {path}')
        origins.append(r['video_time_s'] - r['elapsed_time_s'])
    if len(recordings) != 1 or max(origins) - min(origins) > 1e-6:
        raise ValueError(f'Inconsistent recording/time origin: {path}')
    rows.sort(key=lambda r: r['frame_index'])
    if any(a['video_time_s'] >= b['video_time_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError(f'Non-increasing frame times: {path}')
    return rows


def discover_manifests(root):
    matches = {'left': [], 'right': []}
    for p in Path(root).rglob('*frames.csv'):
        try:
            with p.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if not set(MANIFEST_FIELDS).issubset(reader.fieldnames or []):
                    continue
                first = next(reader, {})
                camera = first.get('camera_id')
                if camera in matches:
                    matches[camera].append(p.resolve())
        except (OSError, UnicodeError, csv.Error):
            continue
    return {k: sorted(set(v)) for k, v in matches.items()}


def extract_export_zip(archive, output_root):
    """Validate ZIP paths and reuse an unchanged extracted archive."""
    archive = Path(archive)
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        total = 0
        for item in entries:
            p = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (p.is_absolute() or '..' in p.parts or '\\' in item.filename
                    or ':' in item.filename or stat.S_ISLNK(mode)):
                raise ValueError(f'Unsafe ZIP entry: {item.filename!r}')
            total += item.file_size
        identity = [str(archive.resolve()), archive.stat().st_size,
                    archive.stat().st_mtime_ns,
                    [(i.filename, i.CRC, i.file_size) for i in entries]]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
        dest = Path(output_root) / ('flight01_extracted_' + key)
        marker = dest / '.extraction_complete.json'
        if marker.is_file():
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(dest).free < total + 512 * 1024**2:
            raise OSError('Not enough free disk to extract ZIP. Upload extracted left/right folders instead.')
        print(f'Extracting {len(entries):,} entries, {total / 1024**3:.2f} GiB...', flush=True)
        z.extractall(dest)
        write_json(marker, {'archive': str(archive), 'entries': len(entries), 'bytes': total})
        return dest


def locate_inputs(cfg):
    root = Path(cfg['input_root'])
    if not root.is_dir():
        raise FileNotFoundError(f'Input folder not found: {root}')
    if cfg['export_zip']:
        export_root = extract_export_zip(cfg['export_zip'], cfg['output_root'])
        candidates = discover_manifests(export_root)
    else:
        candidates = discover_manifests(root)
        if any(not candidates[c] and not cfg[c + '_csv'] for c in candidates):
            archives = sorted(root.rglob('*flight01_export*.zip'))
            if len(archives) == 1:
                export_root = extract_export_zip(archives[0], cfg['output_root'])
                extra = discover_manifests(export_root)
                for c in candidates:
                    if not candidates[c]:
                        candidates[c] = extra[c]
            elif len(archives) > 1:
                raise ValueError('Multiple flight01_export ZIPs. Set EXPORT_ZIP explicitly.')
    manifests, folders, rows = {}, {}, {}
    for camera in ('left', 'right'):
        explicit = cfg[camera + '_csv']
        options = [Path(explicit)] if explicit else candidates[camera]
        if len(options) != 1:
            raise ValueError(f'{camera}: need one frames.csv, found {len(options)}: {options}. '
                             'Add the new export dataset or set LEFT_CSV / RIGHT_CSV.')
        manifests[camera] = options[0]
        rows[camera] = read_manifest(options[0], camera)
        folders[camera] = Path(cfg[camera + '_images']) if cfg[camera + '_images'] else options[0].parent / 'images'
        if not folders[camera].is_dir():
            raise FileNotFoundError(f'{camera} images folder missing: {folders[camera]}. '
                                    'Upload the images too, not only frames.csv.')
        missing = [r['filename'] for r in rows[camera]
                   if not (folders[camera] / r['filename']).is_file()]
        if missing:
            raise FileNotFoundError(f'{camera}: {len(missing)} images listed in CSV are missing; '
                                    f'first examples: {missing[:3]}')
    if rows['left'][0]['recording_id'] != rows['right'][0]['recording_id']:
        raise ValueError('Left/right recording_id differs. Use one export version.')
    weights = ([Path(cfg['weights'])] if cfg['weights'] else
               sorted(set(p.resolve() for p in root.rglob('anti_uav_best.pt'))))
    if len(weights) != 1 or not weights[0].is_file():
        raise ValueError('Add anti_uav_best.pt or set WEIGHTS to its exact path.')
    return manifests, folders, rows, weights[0]


def tile_positions(length, size, overlap):
    if length <= size:
        return [0]
    coords = list(range(0, length - size + 1, max(1, round(size * (1 - overlap)))))
    if coords[-1] != length - size:
        coords.append(length - size)
    return coords


def class_aware_nms(data, threshold):
    data = np.asarray(data, dtype=np.float32).reshape(-1, 6)
    data = data[(data[:, 2] > data[:, 0]) & (data[:, 3] > data[:, 1])]
    order = np.argsort(-data[:, 4], kind='stable')
    keep = []
    while len(order):
        best = int(order[0]); keep.append(best)
        remaining = order[1:]
        if not len(remaining):
            break
        a, b = data[best], data[remaining]
        wh = np.maximum(0, np.minimum(a[2:4], b[:, 2:4]) - np.maximum(a[:2], b[:, :2]))
        intersection = wh[:, 0] * wh[:, 1]
        union = ((a[2] - a[0]) * (a[3] - a[1]) +
                 (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - intersection)
        iou = intersection / np.maximum(union, 1e-9)
        order = remaining[(iou <= threshold) | (b[:, 5] != a[5])]
    return data[keep].copy()


def merge_prediction_boxes(data, overlap_threshold=0.25):
    """Merge connected, overlapping boxes of the same class."""
    data = np.asarray(data, dtype=np.float32).reshape(-1, 6)
    if not 0 < overlap_threshold <= 1:
        raise ValueError('Invalid union threshold')
    if not len(data):
        return data.copy()
    parent = list(range(len(data)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    area = (data[:, 2] - data[:, 0]) * (data[:, 3] - data[:, 1])
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i, 5] != data[j, 5]:
                continue
            wh = np.maximum(0, np.minimum(data[i, 2:4], data[j, 2:4]) -
                            np.maximum(data[i, :2], data[j, :2]))
            inter = float(wh[0] * wh[1])
            if inter / max(float(area[i] + area[j] - inter), 1e-12) >= overlap_threshold:
                parent[root(j)] = root(i)
    groups = defaultdict(list)
    for i in range(len(data)):
        groups[root(i)].append(i)
    out = []
    for ids in groups.values():
        b = data[ids]
        out.append([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max(),
                    b[:, 4].max(), b[0, 5]])
    out = np.asarray(out, dtype=np.float32)
    return out[np.argsort(-out[:, 4], kind='stable')]


def detect_three_methods(model, image, cfg):
    """Run the three inference methods sequentially and record timings."""
    import torch
    gpu = str(cfg['device']).lower() != 'cpu'
    def synchronize():
        if gpu:
            torch.cuda.synchronize(torch.device('cuda:' + str(cfg['device'])))
    def predict(array, size):
        result = model.predict(source=array, imgsz=size, conf=cfg['conf'],
                               iou=cfg['iou'], device=cfg['device'], max_det=cfg['max_det'],
                               verbose=False, save=False, quantize=None, rect=True)[0]
        return result.boxes.data.cpu().numpy().copy()
    height, width = image.shape[:2]
    synchronize(); start = perf_counter()
    whole = predict(image, cfg['imgsz'])
    synchronize(); whole_end = perf_counter()
    collected = [whole.copy()]
    tile_count = 0
    for top in tile_positions(height, cfg['tile_size'], cfg['overlap']):
        for left in tile_positions(width, cfg['tile_size'], cfg['overlap']):
            crop = np.ascontiguousarray(image[top:top + cfg['tile_size'], left:left + cfg['tile_size']])
            boxes = predict(crop, cfg['tile_imgsz'])
            if len(boxes):
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] + left, 0, width)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] + top, 0, height)
                collected.append(boxes)
            tile_count += 1
    tiled = class_aware_nms(np.concatenate(collected, axis=0), cfg['merge_iou'])
    synchronize(); tiled_end = perf_counter()
    merged = merge_prediction_boxes(tiled, cfg['union_iou'])
    merged_end = perf_counter()
    return {'whole': whole, 'tiled': tiled, 'tiled_merged': merged}, {
        'whole_ms': (whole_end - start) * 1000,
        'tiled_total_ms': (tiled_end - start) * 1000,
        'union_merge_ms': (merged_end - tiled_end) * 1000,
        'merged_total_ms': (merged_end - start) * 1000,
        'tile_count': tile_count,
    }


def prediction_rows(frame, method, boxes, names):
    base = dict(frame, time_reference='export_relative_not_original_MOV', method=method,
                synchronization_verified=False, cross_view_correspondence_verified=False,
                ready_for_3d=False)
    if not len(boxes):
        yield dict(base, detection_status='no_detection')
    for i, (x1, y1, x2, y2, confidence, class_id) in enumerate(boxes, 1):
        cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
        yield dict(base, detection_status='detected', detection_id=i, class_id=int(class_id),
                   class_name=names[int(class_id)], confidence=float(confidence),
                   x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                   center_x_px=cx, center_y_px=cy,
                   center_x_norm=cx / frame['width'], center_y_norm=cy / frame['height'])


def make_pair_rows(records, indices, recording):
    """Keep candidate pairs and mark ambiguous frames for later screening."""
    out = []
    for index in indices:
        pair = dict(recording_id=recording, frame_index=index,
                    synchronization_verified=False, cross_view_correspondence_verified=False,
                    ready_for_3d=False)
        views = [records.get((c, index)) for c in ('left', 'right')]
        for camera, view in zip(('left', 'right'), views):
            if view is None:
                continue
            frame, boxes = view
            pair[camera + '_filename'] = frame['filename']
            pair[camera + '_elapsed_time_s'] = frame['elapsed_time_s']
            pair[camera + '_count'] = len(boxes)
            if len(boxes) == 1:
                pair[camera + '_center_x_px'] = float((boxes[0, 0] + boxes[0, 2]) / 2)
                pair[camera + '_center_y_px'] = float((boxes[0, 1] + boxes[0, 3]) / 2)
        if any(v is None for v in views):
            pair['pair_status'] = 'missing_view_in_manifest'
        else:
            counts = [len(v[1]) for v in views]
            pair['nominal_export_time_difference_ms'] = abs(
                views[0][0]['elapsed_time_s'] - views[1][0]['elapsed_time_s']) * 1000
            pair['pair_status'] = ('missing_detection' if min(counts) == 0 else
                                   'multiple_boxes_ambiguous' if max(counts) > 1 else
                                   'one_box_each_unverified_candidate')
        out.append(pair)
    return out


def save_preview(image, boxes, row, names, path):
    import cv2
    canvas = image.copy()
    for x1, y1, x2, y2, confidence, class_id in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(canvas, p1, p2, (0, 220, 255), 2)
        cv2.circle(canvas, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 3, (0, 0, 255), -1)
        label = f'{names[int(class_id)]} {confidence:.2f}'
        cv2.putText(canvas, label, (p1[0], max(60, p1[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 2)
    title = (f"{row['camera_id']} | frame {row['frame_index']} | export {row['elapsed_time_s']:.3f}s"
             f" | tiled+union: {len(boxes)} predictions")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (25, 25, 25), -1)
    cv2.putText(canvas, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f'Could not write preview: {path}')


README = '''# Flight01 detection results

## Prediction files

- `predictions_whole.csv`: detections on full images.
- `predictions_tiled.csv`: full-image and overlapping-tile detections combined with global class-aware NMS.
- `predictions_tiled_merged.csv`: detections after an additional overlap-merging step (union IoU 0.25).
- `frame_summary.csv`: frame-level detection counts and timing measurements.
- `stereo_pair_candidates.csv`: candidate left/right pairs grouped by exported frame index.

Bounding boxes and centers use original exported-image pixel coordinates. Frames
without detections have one row with empty box fields. Preview images use the
merged detection results.

## Timing measurements

The three methods run sequentially within one shared inference pass, with GPU
synchronization at stage boundaries. `frame_summary.csv` includes:

- `whole_ms`: full-image inference, preprocessing/postprocessing, and CPU box handling.
- `tiled_total_ms`: `whole_ms` plus tile inference, coordinate mapping, and global NMS.
- `union_merge_ms`: additional CPU time for overlap merging.
- `merged_total_ms`: `tiled_total_ms` plus `union_merge_ms`.
- `image_load_ms`: image reading time, recorded separately.

Stage timings cover inference and processing. `processing_wall_s` includes image
reading, inference, CSV writing, and previews. Setup, model loading, warmup,
summaries, and ZIP packaging are measured outside this value.

## Evaluation and localization

The predictions use export-relative frame times. Camera time offsets and 3D
positions are estimated in the localization stage. Detection precision, recall,
and mAP require manual reference annotations.

The union IoU threshold of 0.25 was selected from exploratory samples. Compare
`predictions_tiled.csv` with `predictions_tiled_merged.csv` to inspect its effect
on adjacent boxes.
'''


def run_detection(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f'Unknown config keys: {sorted(unknown)}')
        cfg.update(config)
    for key in ('frame_step', 'tile_size', 'tile_imgsz', 'imgsz', 'max_det'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['max_pairs'] is not None and (not isinstance(cfg['max_pairs'], int) or cfg['max_pairs'] < 1):
        raise ValueError('max_pairs must be None or a positive integer')
    if (not isinstance(cfg['preview_pairs'], int) or cfg['preview_pairs'] < 0
            or not isinstance(cfg['warmup_frames'], int) or cfg['warmup_frames'] < 1):
        raise ValueError('preview_pairs >= 0 and warmup_frames >= 1 required')
    if not 0 <= cfg['overlap'] < 1 or any(not 0 < cfg[k] <= 1 for k in ('conf', 'iou', 'merge_iou', 'union_iou')):
        raise ValueError('Invalid confidence/IoU/overlap configuration')
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    if ultralytics.__version__ != '8.4.166':
        raise RuntimeError('I tested this script with ultralytics==8.4.166; install requirements.txt')
    cfg['device'] = str(cfg['device'])
    if cfg['device'] != 'cpu' and not cfg['device'].isdigit():
        raise ValueError('device must be cpu or a single CUDA index such as 0')
    if cfg['require_gpu'] and (not torch.cuda.is_available() or cfg['device'] == 'cpu'):
        raise RuntimeError('This configuration requires a CUDA GPU. To use CPU, set require_gpu=false and device=cpu.')
    if cfg['device'] != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose a GPU or set device=cpu, require_gpu=False.')
    Path(cfg['output_root']).mkdir(parents=True, exist_ok=True)
    manifests, folders, rows, weights = locate_inputs(cfg)
    indices = sorted({r['frame_index'] for camera in rows for r in rows[camera]
                      if r['frame_index'] % cfg['frame_step'] == 0})
    if cfg['max_pairs']:
        indices = indices[:cfg['max_pairs']]
    if not indices:
        raise ValueError('No frames selected. Reduce frame_step.')
    selected_ids = set(indices)
    by_camera = {c: {r['frame_index']: r for r in rr} for c, rr in rows.items()}
    selected = [(c, by_camera[c][i]) for i in indices for c in ('left', 'right') if i in by_camera[c]]
    previews = set(indices[int(j)] for j in np.linspace(
        0, len(indices) - 1, min(cfg['preview_pairs'], len(indices)), dtype=int))
    out = Path(cfg['output_root']) / ('flight01_detection_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    metadata = {}
    for camera, path in manifests.items():
        meta = path.parent / 'video_metadata.json'
        metadata[camera] = json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else None
    info = {
        'status': 'initializing', 'config': cfg, 'output_directory': str(out),
        'weights': str(weights), 'weights_sha256': sha256_file(weights),
        'manifest_paths': {c: str(p) for c, p in manifests.items()},
        'manifest_sha256': {c: sha256_file(p) for c, p in manifests.items()},
        'export_metadata': metadata,
        'manifest_rows': {c: len(rr) for c, rr in rows.items()},
        'manifest_index_gaps': {c: len(rr) != rr[-1]['frame_index'] - rr[0]['frame_index'] + 1 for c, rr in rows.items()},
        'selected_candidate_indices': len(indices), 'selected_images': len(selected),
        'processed_images': 0, 'time_reference': 'new_export_relative_only',
        'synchronization_verified': False, 'cross_view_correspondence_verified': False,
        'ready_for_3d': False, 'accuracy_evaluated': False,
        'versions': {'python': platform.python_version(), 'ultralytics': ultralytics.__version__,
                     'torch': torch.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__},
        'device_name': torch.cuda.get_device_name(int(cfg['device'])) if cfg['device'] != 'cpu' else 'CPU',
    }
    write_json(out / 'run_info.json', info)
    (out / 'README.txt').write_text(README, encoding='utf-8')
    print(f"Preflight OK: left={len(rows['left']):,}, right={len(rows['right']):,} images present.")
    print(f'Selected {len(indices):,} candidate indices / {len(selected):,} images; output: {out}', flush=True)
    stats, pair_records = [], {}
    try:
        model = YOLO(str(weights))
        if model.task != 'detect':
            raise ValueError(f'Expected detection weights, got task={model.task}')
        camera, frame = selected[0]
        warm_image = cv2.imread(str(folders[camera] / frame['filename']))
        if warm_image is None or warm_image.shape[:2] != (frame['height'], frame['width']):
            raise ValueError('Warmup image unreadable or dimensions differ from manifest')
        for _ in range(cfg['warmup_frames']):
            detect_three_methods(model, warm_image, cfg)
        info['model_names'] = model.names
        info['precision_fp16'] = bool(getattr(model.predictor.model, 'fp16', False))
        if info['precision_fp16']:
            raise RuntimeError('Unexpected FP16 model; this comparison requires FP32.')
        info['status'] = 'running'; write_json(out / 'run_info.json', info)
        print('Warmup complete; starting measured processing.', flush=True)
        start_wall = perf_counter()
        with ExitStack() as stack:
            handles, writers = {}, {}
            for method in METHODS:
                handles[method] = stack.enter_context((out / f'predictions_{method}.csv').open('w', encoding='utf-8-sig', newline=''))
                writers[method] = csv.DictWriter(handles[method], fieldnames=DETECTION_FIELDS)
                writers[method].writeheader()
            summary_handle = stack.enter_context((out / 'frame_summary.csv').open('w', encoding='utf-8-sig', newline=''))
            summary_writer = csv.DictWriter(summary_handle, fieldnames=FRAME_FIELDS)
            summary_writer.writeheader()
            for n, (camera, frame) in enumerate(selected, 1):
                start_load = perf_counter()
                image = cv2.imread(str(folders[camera] / frame['filename']))
                load_ms = (perf_counter() - start_load) * 1000
                if image is None or image.shape[:2] != (frame['height'], frame['width']):
                    raise ValueError(f"Unreadable image or wrong dimensions: {camera}/{frame['filename']}")
                predictions, timing = detect_three_methods(model, image, cfg)
                for method in METHODS:
                    writers[method].writerows(prediction_rows(frame, method, predictions[method], model.names))
                summary = dict(frame, status='processed', image_load_ms=load_ms, **timing,
                               **{m + '_count': len(predictions[m]) for m in METHODS})
                summary_writer.writerow(summary); stats.append(summary)
                pair_records[(camera, frame['frame_index'])] = (frame, predictions['tiled_merged'])
                if frame['frame_index'] in previews:
                    save_preview(image, predictions['tiled_merged'], frame, model.names,
                                 out / 'preview_images' / camera / frame['filename'])
                info['processed_images'] = n
                if n % 10 == 0 or n == len(selected):
                    for handle in handles.values():
                        handle.flush()
                    summary_handle.flush()
                    write_json(out / 'run_info.json', info)
                    elapsed = perf_counter() - start_wall
                    print(f'{n:,}/{len(selected):,} images | processing wall {elapsed:.1f}s', flush=True)
        info['processing_wall_s'] = perf_counter() - start_wall
        pairs = make_pair_rows(pair_records, indices, rows['left'][0]['recording_id'])
        with (out / 'stereo_pair_candidates.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
            writer.writeheader(); writer.writerows(pairs)
        count_summary = []
        for camera in ('left', 'right', 'all'):
            ss = stats if camera == 'all' else [s for s in stats if s['camera_id'] == camera]
            for method in METHODS:
                count_summary.append({'camera': camera, 'method': method, 'processed_images': len(ss),
                                      'frames_with_predictions': sum(s[method + '_count'] > 0 for s in ss),
                                      'prediction_boxes': sum(s[method + '_count'] for s in ss)})
        with (out / 'counts_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(count_summary[0]))
            writer.writeheader(); writer.writerows(count_summary)
        timings = {}
        for key in ('image_load_ms', 'whole_ms', 'tiled_total_ms', 'union_merge_ms', 'merged_total_ms'):
            values = np.array([s[key] for s in stats])
            timings[key] = {'mean': float(values.mean()), 'median': float(np.median(values)),
                            'p95': float(np.percentile(values, 95)), 'total_ms': float(values.sum())}
        timings['tiled_vs_whole_ratio'] = timings['tiled_total_ms']['total_ms'] / timings['whole_ms']['total_ms']
        timings['tiled_added_mean_ms_per_image'] = timings['tiled_total_ms']['mean'] - timings['whole_ms']['mean']
        timings['timing_protocol'] = 'same_pass_cumulative_stages_FP32_GPU_synchronized_IO_excluded'
        timings['processed_images'] = len(stats)
        write_json(out / 'timing_summary.json', timings)
        info.update(status='complete', count_summary=count_summary, timing_summary=timings,
                    candidate_pair_status_counts={k: sum(p['pair_status'] == k for p in pairs)
                                                  for k in sorted({p['pair_status'] for p in pairs})})
        write_json(out / 'run_info.json', info)
    except Exception as exc:
        info.update(status='failed', error=repr(exc))
        write_json(out / 'run_info.json', info)
        print(f'Run failed; partial CSVs and diagnostics retained at {out}', flush=True)
        raise
    zip_path = out.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(out))
    print(f'Complete. Results ZIP: {zip_path}', flush=True)
    return {'output_dir': str(out), 'zip_path': str(zip_path), 'run_info': info}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Optional JSON configuration file')
    args = parser.parse_args()
    run_detection(json.loads(Path(args.config).read_text()) if args.config else None)

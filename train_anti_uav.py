"""Train and evaluate YOLO26s on Anti-UAV300 RGB video sequences.

The script samples frames, writes YOLO labels, splits data by whole video,
trains the detector, evaluates the held-out subset, and exports the weights.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import zipfile


HF_REPO_ID = "VoyageWang/antiuav"
HF_FILENAME = "Anti-UAV300.zip"


def get_annotation_arrays(annotation):
    """Read common Anti-UAV annotation formats."""
    existence = annotation.get("exist")
    boxes = annotation.get("gt_rect") or annotation.get("gt") or annotation.get("bbox")
    if existence is None:
        raise KeyError("The annotation JSON does not contain an 'exist' field")
    if boxes is None:
        raise KeyError("The annotation JSON has no supported bounding-box field")
    return existence, boxes


def bbox_xywh_to_yolo(box, image_width, image_height):
    """Clip a pixel-space box to the image and convert it to YOLO xywh."""
    if box is None or len(box) < 4 or image_width <= 0 or image_height <= 0:
        return None

    x, y, width, height = map(float, box[:4])
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None

    # Clip corners first, then calculate the normalized center and size.
    x1 = max(0.0, min(x, float(image_width)))
    y1 = max(0.0, min(y, float(image_height)))
    x2 = max(0.0, min(x + width, float(image_width)))
    y2 = max(0.0, min(y + height, float(image_height)))
    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2.0) / image_width
    yc = ((y1 + y2) / 2.0) / image_height
    wn = (x2 - x1) / image_width
    hn = (y2 - y1) / image_height
    return xc, yc, wn, hn


def get_archive(archive_path, work_dir):
    if archive_path is not None:
        path = Path(archive_path)
        if not path.is_file():
            raise FileNotFoundError(f"Anti-UAV300 archive not found: {path}")
        return path
    # Download the public dataset when no local archive is supplied.
    from huggingface_hub import hf_hub_download

    download_dir = work_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME,
                                repo_type="dataset", local_dir=str(download_dir)))


def discover_sequences(archive, max_sequences, seed):
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
    videos = sorted(name for name in names
                    if "/train/" in name.lower() and name.lower().endswith("/visible.mp4"))
    pairs = []
    for video_name in videos:
        annotation_name = video_name[:-len("visible.mp4")] + "visible.json"
        if annotation_name in names:
            pairs.append({"sequence": Path(video_name).parent.name,
                          "video_member": video_name, "json_member": annotation_name})
    if not pairs:
        raise RuntimeError("I could not find train/**/visible.mp4 + visible.json pairs")
    if max_sequences is not None:
        pairs = random.Random(seed).sample(pairs, k=min(max_sequences, len(pairs)))
    return pairs


def split_sequences(pairs, seed, train_ratio=.75, val_ratio=.15):
    if len(pairs) < 3:
        raise ValueError("At least three sequences are required for train, validation, and test")
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_ratio))
    n_val = max(1, int(len(shuffled) * val_ratio))
    if len(shuffled) - n_train - n_val < 1:
        n_train -= 1
    splits = {"train": shuffled[:n_train],
              "val": shuffled[n_train:n_train + n_val],
              "test": shuffled[n_train + n_val:]}
    for a in splits:
        for b in splits:
            if a != b:
                assert {p['video_member'] for p in splits[a]}.isdisjoint(
                    {p['video_member'] for p in splits[b]})
    return splits


def prepare_split(split_name, split_pairs, archive, dataset_dir, frame_stride):
    """Sample video frames and write single-class YOLO labels."""
    import cv2

    images_dir = dataset_dir / "images" / split_name
    labels_dir = dataset_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with zipfile.ZipFile(archive) as z:
        for index, item in enumerate(split_pairs, 1):
            print(f"[{split_name}] {index}/{len(split_pairs)} {item['sequence']}", flush=True)
            annotation = json.loads(z.read(item['json_member']).decode('utf-8'))
            exists, boxes = get_annotation_arrays(annotation)
            with tempfile.TemporaryDirectory() as tmp:
                # Extract the current video using a validated archive path.
                video_path = Path(tmp) / "visible.mp4"
                with z.open(item['video_member']) as source, video_path.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    stats['video_open_errors'] += 1
                    continue
                frame_id = 0
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if frame_id % frame_stride:
                            frame_id += 1
                            continue
                        height, width = frame.shape[:2]
                        image_name = f"{item['sequence']}_{frame_id:06d}.jpg"
                        image_path = images_dir / image_name
                        label_path = labels_dir / f"{Path(image_name).stem}.txt"
                        if not cv2.imwrite(str(image_path), frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, 92]):
                            raise OSError(f"I could not save {image_path}")
                        has_drone = frame_id < len(exists) and bool(exists[frame_id])
                        if has_drone and frame_id < len(boxes):
                            result = bbox_xywh_to_yolo(boxes[frame_id], width, height)
                            if result is None:
                                label_path.write_text('', encoding='utf-8')
                                stats['invalid_boxes'] += 1
                            else:
                                xc, yc, wn, hn = result
                                label_path.write_text(
                                    f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n", encoding='utf-8')
                                stats['positive_frames'] += 1
                                stats['drone_boxes'] += 1
                        else:
                            # Keep empty labels for background training examples.
                            label_path.write_text('', encoding='utf-8')
                            stats['negative_frames'] += 1
                        stats['images'] += 1
                        frame_id += 1
                finally:
                    cap.release()
    return stats


def audit_stats(stats_by_split):
    for split, stats in stats_by_split.items():
        if stats['positive_frames'] == 0:
            raise ValueError(f"{split} contains no positive drone frames")
        if stats['negative_frames'] == 0:
            print(f"Warning: {split} contains no negative/background frames")


def show_samples(dataset_dir, output_file, seed, count=6):
    """Save an image-and-label preview for checking the annotations."""
    import cv2
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    images_dir = dataset_dir / 'images' / 'train'
    labels_dir = dataset_dir / 'labels' / 'train'
    images = sorted(images_dir.glob('*.jpg'))
    chosen = random.Random(seed).sample(images, k=min(count, len(images)))
    if not chosen:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax in axes.flat:
        ax.axis('off')
    for ax, image_path in zip(axes.flat, chosen):
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        label_text = (labels_dir / f'{image_path.stem}.txt').read_text().strip()
        if label_text:
            _, xc, yc, bw, bh = map(float, label_text.split())
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            ax.add_patch(plt.Rectangle((x1, y1), bw * w, bh * h,
                                       fill=False, linewidth=2, edgecolor='red'))
            ax.set_title('drone')
        else:
            ax.set_title('background')
    fig.tight_layout()
    fig.savefig(output_file, dpi=130)
    plt.close(fig)


def write_dataset_yaml(dataset_dir, output_path):
    import yaml

    config = {'path': str(dataset_dir.resolve()), 'train': 'images/train',
              'val': 'images/val', 'test': 'images/test', 'names': {0: 'drone'}}
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')


def metrics_dict(metrics):
    return {'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
            'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map)}


def run_video_demo(model, test_pairs, archive, work_dir, imgsz, device, conf):
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / 'held_out_visible.mp4'
        with zipfile.ZipFile(archive) as z:
            with z.open(test_pairs[0]['video_member']) as source, video_path.open('wb') as dest:
                shutil.copyfileobj(source, dest)
        # Use a held-out public sequence for the optional preview.
        predictions = model.predict(source=str(video_path), imgsz=imgsz, conf=conf,
                                    iou=.45, device=device, save=True, stream=True,
                                    project=str(work_dir / 'video_demo'),
                                    name=test_pairs[0]['sequence'], exist_ok=True, verbose=False)
        return sum(1 for _ in predictions)


def run(args):
    import numpy as np
    import torch
    from ultralytics import YOLO
    import ultralytics

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = work_dir / 'dataset'
    if dataset_dir.exists():
        raise FileExistsError(f"Dataset directory already exists: {dataset_dir}. "
                              "Use a different --work-dir")
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else 'cpu')
    print(f"Python/PyTorch/Ultralytics: {torch.__version__} / {ultralytics.__version__}")
    print(f"Training device: {device}")

    archive = get_archive(args.archive, work_dir)
    sequences = discover_sequences(archive, args.max_sequences, args.seed)
    splits = split_sequences(sequences, args.seed)
    print('Sequence counts:', {split: len(items) for split, items in splits.items()})
    stats = {split: prepare_split(split, items, archive, dataset_dir, args.frame_stride)
             for split, items in splits.items()}
    audit_stats(stats)
    (work_dir / 'dataset_audit.json').write_text(json.dumps({
        'sequence_members': {split: [item['video_member'] for item in items]
                             for split, items in splits.items()},
        'frame_stride': args.frame_stride,
        'counts': {split: dict(values) for split, values in stats.items()},
    }, indent=2), encoding='utf-8')
    if not args.skip_preview:
        show_samples(dataset_dir, work_dir / 'sample_train.png', args.seed)
    yaml_path = work_dir / 'anti_uav_drone.yaml'
    write_dataset_yaml(dataset_dir, yaml_path)

    runs_dir = work_dir / 'runs'
    name = 'yolo26s_realistic_drone'
    model = YOLO(args.model)
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, patience=args.patience, device=device,
                project=str(runs_dir), name=name, exist_ok=True,
                pretrained=True, seed=args.seed, workers=2, plots=True,
                mosaic=1.0, close_mosaic=5, verbose=True)
    best_weights = runs_dir / name / 'weights' / 'best.pt'
    if not best_weights.is_file():
        raise FileNotFoundError(f"I could not find the trained checkpoint: {best_weights}")
    best_model = YOLO(str(best_weights))
    validation = best_model.val(data=str(yaml_path), split='val', imgsz=args.imgsz,
                                batch=args.batch, device=device, plots=True, verbose=True)
    test = best_model.val(data=str(yaml_path), split='test', imgsz=args.imgsz,
                          batch=args.batch, device=device, plots=True, verbose=True)
    scores = {'anti_uav_validation': metrics_dict(validation),
              'anti_uav_held_out_test': metrics_dict(test)}
    (work_dir / 'evaluation.json').write_text(json.dumps(scores, indent=2), encoding='utf-8')
    print('Evaluation:', json.dumps(scores, indent=2))
    if args.video_demo:
        count = run_video_demo(best_model, splits['test'], archive, work_dir,
                               args.imgsz, device, args.conf)
        print(f"Processed {count} frames in the optional public video demo")
    export_dir = work_dir / 'exported_model'
    export_dir.mkdir(exist_ok=True)
    exported = export_dir / 'anti_uav_best.pt'
    shutil.copy2(best_weights, exported)
    print('Exported trained checkpoint:', exported)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Existing Anti-UAV300.zip; downloaded if omitted')
    parser.add_argument('--work-dir', type=Path, default=Path('anti_uav_work'))
    parser.add_argument('--model', default='yolo26s.pt')
    parser.add_argument('--frame-stride', type=int, default=5)
    parser.add_argument('--max-sequences', type=int)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--device', default=None, help='YOLO device, for example 0 or cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-preview', action='store_true')
    parser.add_argument('--video-demo', action='store_true')
    args = parser.parse_args()
    if args.frame_stride <= 0:
        parser.error('--frame-stride must be positive')
    run(args)


if __name__ == '__main__':
    main()

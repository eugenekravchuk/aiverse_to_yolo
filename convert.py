#!/usr/bin/env python3
"""
Convert AI Verse dataset scenes to YOLO (object detection) format and emit data.yaml.

Input structure (per scene):
  <scene_dir>/
    scene_instances.json
    beauty.XXXX.png / .jpg  (and other optional images)

Output:
  <out_dir>/
    images/
      <image files copied here>
    labels/
      <same basenames>.txt (YOLO format: class_id x_center y_center w h)
    data.yaml   (Ultralytics dataset config; train/val both point to images/)
"""

import argparse
import json
import shutil
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True,
                   help="Dataset root containing scene folders (each with scene_instances.json).")
    p.add_argument("--out", dest="out", type=Path, required=True,
                   help="Output dir (YOLO). Will create images/ and labels/ inside.")
    p.add_argument("--label-field", default="class",
                   choices=["class", "subclass", "superclass", "path", "class_subclass"],
                   help="Which field to use for class names.")
    p.add_argument("--image-key", default="file_name",
                   help="Key in images[] for the filename (default: file_name).")
    p.add_argument("--image-extensions", nargs="+", default=[".png", ".jpg", ".jpeg"],
                   help="Allowed image extensions if file_name is missing or needs resolving.")
    p.add_argument("--strict", action="store_true",
                   help="Error on missing files/fields (default: skip with warning).")
    p.add_argument("--yaml-name", default="data.yaml",
                   help="Name of the generated Ultralytics YAML file (default: data.yaml).")
    p.add_argument("--clean", action="store_true",
               help="Delete out/images, out/labels, data.yaml and classes.txt before running.")
    return p.parse_args()

def yolo_bbox(xmin, ymin, xmax, ymax, w, h):
    '''
    (x_min, y_min, x_max, y_max) -> (x_center, y_center, width, height)
    '''
    xmin = max(0, min(xmin, w - 1))
    xmax = max(0, min(xmax, w - 1))
    ymin = max(0, min(ymin, h - 1))
    ymax = max(0, min(ymax, h - 1))
    bw = max(0.0, xmax - xmin)
    bh = max(0.0, ymax - ymin)
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h

def label_from_instance(inst, mode="class"):
    '''
    Getting label for yolo data.yaml, based on "mode"
    '''
    if mode == "class":
        return inst.get("class")
    if mode == "subclass":
        return inst.get("subclass")
    if mode == "superclass":
        return inst.get("superclass")
    if mode == "path":
        p = inst.get("path")
        return p.split("/")[-1] if isinstance(p, str) else None
    if mode == "class_subclass":
        c = inst.get("class")
        s = inst.get("subclass")
        if c and s:
            return f"{c}/{s}"
        return c or s
    return None

def find_scene_dirs(root: Path):
    return [p for p in root.rglob("scene_instances.json")]

def ensure_dirs(out_root: Path):
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

def resolve_image_path(scene_dir: Path, fname: str | None, exts):
    '''
    1. Looking directly by path 
    2. Running through all alowed extensions
    '''
    if fname:
        p = (scene_dir / fname)
        if p.exists():
            return p
        p = scene_dir / Path(fname).name
        if p.exists():
            return p
    for ext in exts:
        cand = list(scene_dir.glob(f"*{ext}"))
        beauty = [c for c in cand if c.name.startswith("beauty.")]
        if beauty:
            return beauty[0]
        if cand:
            return cand[0]
    return None

def write_data_yaml(out_root: Path, class_list: list[str], yaml_name: str):
    """
    Write a minimal Ultralytics YAML with no split: both train and val point to 'images/'.
    """
    content_lines = [
        f"path: {out_root.as_posix()}",
        "train: images",
        "val: images",
        "names:"
    ]
    
    for cls_num in range(len(class_list)):
        content_lines.append(f"  {cls_num}: {class_list[cls_num]}")
    (out_root / yaml_name).write_text("\n".join(content_lines) + "\n", encoding="utf-8")

def clean_output(out_root: Path, yaml_name: str):
    '''
    Cleaning input and output dirs
    '''
    for sub in ["images", "labels"]:
        d = out_root / sub
        if d.exists() and d.is_dir():
            shutil.rmtree(d)
    for fname in ["classes.txt", yaml_name]:
        f = out_root / fname
        if f.exists():
            f.unlink()

def main():
    args = parse_args()
    in_root: Path = args.inp
    out_root: Path = args.out

    if args.clean:
        clean_output(out_root, args.yaml_name)

    ensure_dirs(out_root)

    classes = {}
    next_cid = 0
    stats = {"images": 0, "instances": 0, "skipped_instances": 0, "scenes": 0}

    # iterating through each scene_insatnces.json
    for json_path in find_scene_dirs(in_root):
        scene_dir = json_path.parent
        stats["scenes"] += 1
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            msg = f"[WARN] Failed to read {json_path}: {e}"
            if args.strict: raise RuntimeError(msg)
            print(msg); continue

        images = data.get("images", [])
        instances = data.get("instances", [])

        # map - images:meta
        img_map = {}
        for im in images:
            iid = im.get("id")
            file_name = im.get(args.image_key)
            width = im.get("width")
            height = im.get("height")
            img_map[iid] = {"file_name": file_name, "width": width, "height": height}

        # map - image_id:list_of_instances
        per_image = {iid: [] for iid in img_map.keys()}
        for inst in instances:
            iid = inst.get("image_id")
            if iid not in per_image:
                stats["skipped_instances"] += 1
                continue
            per_image[iid].append(inst)

        # for every image:
        # 1. find by path
        # 2. create new name
        # 3. copy it to output
        # 4. create txt file 
        for iid, meta in img_map.items():
            fname = meta.get("file_name")
            w = meta.get("width")
            h = meta.get("height")

            src_img = resolve_image_path(scene_dir, fname, args.image_extensions)
            if src_img is None or not src_img.exists():
                msg = f"[WARN] Missing image for iid={iid} (declared '{fname}') in {scene_dir}"
                if args.strict: raise FileNotFoundError(msg)
                print(msg); continue

            new_name = f"{scene_dir.name}_{src_img.name}"
            out_img = out_root / "images" / new_name
            out_lbl = out_root / "labels" / (Path(new_name).stem + ".txt")

            if not out_img.exists():
                shutil.copy2(src_img, out_img)

            # getting data for txt file
            # (cls_id, ...yolo_bbox)
            lines = []
            for inst in per_image[iid]:
                # getting label
                lab = label_from_instance(inst, args.label_field)
                if lab is None:
                    stats["skipped_instances"] += 1
                    continue
                if lab not in classes:
                    classes[lab] = next_cid
                    next_cid += 1
                cid = classes[lab]
                
                # getting bbox
                bbox = inst.get("bbox")
                if not bbox or len(bbox) != 4:
                    stats["skipped_instances"] += 1
                    continue

                xmin, ymin, xmax, ymax = bbox
                if w is None or h is None:
                    try:
                        from PIL import Image
                        with Image.open(src_img) as im:
                            w, h = im.size
                    except Exception:
                        msg = f"[WARN] Missing width/height for {src_img}, and PIL not available."
                        if args.strict: raise RuntimeError(msg)
                        print(msg); continue

                x, y, bw, bh = yolo_bbox(float(xmin), float(ymin), float(xmax), float(ymax), float(w), float(h))
                if bw <= 0 or bh <= 0:
                    stats["skipped_instances"] += 1
                    continue

                lines.append(f"{cid} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

            out_lbl.write_text("\n".join(lines), encoding="utf-8")
            stats["images"] += 1
            stats["instances"] += len(lines)


    inv = [None] * len(classes)
    for k, v in classes.items():
        inv[v] = k

    # here yaml file gets written 
    write_data_yaml(out_root, inv, args.yaml_name)

    print(f"[DONE] Scenes: {stats['scenes']}, Images: {stats['images']}, "
          f"YOLO instances: {stats['instances']}, Skipped instances: {stats['skipped_instances']}")
    print(f"[INFO] Wrote: {out_root/'images'}  and  {out_root/'labels'}")
    print(f"[INFO] Dataset YAML -> {out_root/args.yaml_name}")

if __name__ == "__main__":
    main()

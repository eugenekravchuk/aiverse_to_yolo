# AI Verse to YOLO Dataset Converter

A Python tool to convert AI Verse dataset format to YOLO (You Only Look Once) object detection format for training computer vision models.

##  Dataset Structure

### Input Format (AI Verse)
```
input/
├── scene.{scene_id}/
│   ├── scene_instances.json    # Scene annotations
│   ├── beauty.0001.png        # Scene images
│   ├── beauty.0002.png
│   └── ...
└── scene.{another_id}/
    └── ...
```

### Output Format (YOLO)
```
output/
├── images/                    # All training images
│   ├── scene.{id}_beauty.0001.png
│   └── ...
├── labels/                    # YOLO format annotations
│   ├── scene.{id}_beauty.0001.txt
│   └── ...
└── data.yaml                 # Dataset configuration
```

## Quick Start

### Basic Usage
```bash
python convert.py --in input/ --out output/
```

### Usage with clenaing 
```bash
python convert.py \
  --in input/ \
  --out output/ \
  --clean
```

## Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--in` | Input directory with AI Verse scenes | Required | - |
| `--out` | Output directory for YOLO dataset | Required | - |
| `--label-field` | Field to use for class labels | `class` | `class`, `subclass`, `superclass`, `path`, `class_subclass` |
| `--image-key` | Key for image filename in JSON | `file_name` | - |
| `--image-extensions` | Allowed image file extensions | `[.png, .jpg, .jpeg]` | - |
| `--strict` | Error on missing files/fields | `False` | - |
| `--yaml-name` | Name of output YAML file | `data.yaml` | - |
| `--clean` | Clean output directory before conversion | `False` | - |


## Example Conversion

### Input: AI Verse Scene
```json
{
  "images": [
    {
      "id": "img-001",
      "file_name": "beauty.0001.png",
      "width": 1280,
      "height": 960
    }
  ],
  "instances": [
    {
      "image_id": "img-001",
      "class": "house_north_america",
      "subclass": "medium",
      "superclass": "building",
      "bbox": [76, 0, 281, 128]
    }
  ]
}
```

### Output: YOLO Format
**Image**: `scene.example_beauty.0001.png`

**Label** (`scene.example_beauty.0001.txt`):
```
0 0.139453 0.066667 0.160156 0.133333
```

**Dataset Config** (`data.yaml`):
```yaml
path: output
train: images
val: images
names:
  0: house_north_america
```
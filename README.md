# Wildlife Detection using Computer Vision

Real-time animal detection from a camera feed, built with **YOLOv10** and **OpenCV**. It is the detection core of a system meant to warn people when a dangerous animal comes near homes in high-wildlife areas, to help reduce human-wildlife conflict.

The script reads frames from a webcam and runs a YOLOv10-medium detector on each one. It draws a labelled bounding box around every object it finds and prints the class and confidence. Animals the model can recognise include elephants, bears, zebras, giraffes, horses, cows, sheep, dogs, cats and birds.

---

## How it works

```
Webcam (640×480) ──► YOLOv10m (Ultralytics) ──► boxes + classes + confidences ──► OpenCV overlay
```

1. **Capture:** OpenCV grabs frames from the default camera at 640×480.
2. **Detect:** each frame goes through a pretrained `yolov10m.pt` model in streaming mode.
3. **Annotate:** every detection gets a bounding box and its class name drawn on the frame. The confidence (rounded to two decimals) and the class are printed to the console.
4. **Display:** the annotated feed is shown in a window. Press **`q`** to quit.

**Why YOLO:** a single-stage detector predicts boxes and classes in one pass. That makes it fast enough for live video on ordinary hardware, which an alert system needs.

## Getting started

```bash
pip install ultralytics opencv-python
```

Download the YOLOv10-medium weights and place them at `yolo-Weights/yolov10m.pt`. Alternatively, change the path in the script to `"yolov10m.pt"` and Ultralytics will download the weights automatically. Then run:

```bash
python "Python Code.py"
```

## Known issue: class labels

The script uses its own hard-coded `classNames` list. That list adds `"snake"` and `"tiger"` to the standard COCO class order, so from index 23 onwards the names no longer match what the model predicts. For example, a giraffe is labelled "snake".

The pretrained model was trained on COCO, which has no snake or tiger class, so it cannot detect them anyway. The simplest fix is to use the model's own label names:

```python
cls = int(box.cls[0])
label = model.names[cls]   # instead of classNames[cls]
```

Detecting snakes, tigers, leopards and similar animals needs a model fine-tuned on a wildlife dataset. That is the first item in the roadmap below.

## Roadmap

- [ ] Fine-tune YOLO on a wildlife dataset, for example camera-trap images, so it covers locally dangerous species such as leopards, tigers, snakes and wild boar.
- [ ] Define a "dangerous species" list with a confidence threshold, and trigger an **alert** (sound, SMS or a push notification) when one is detected.
- [ ] Add tracking across frames (for example SORT, with a Kalman filter and Hungarian matching) so one animal raises one alert, not one per frame.
- [ ] Improve night-time performance using IR cameras and low-light augmentation.
- [ ] Deploy on an edge device such as a Raspberry Pi or Jetson Nano, with an outdoor camera.

## Tech stack

`Python` · `Ultralytics YOLOv10` · `OpenCV` · `PyTorch`

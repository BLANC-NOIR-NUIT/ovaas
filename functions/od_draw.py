import cv2
import numpy as np


classes = [
        "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
        "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush", "hair brush"
    ]
colors = cv2.applyColorMap(
        src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
        colormap=cv2.COLORMAP_RAINBOW,
        ).squeeze()

def draw_boxes(frame, boxes):
        for label, score, box in boxes:
            # Choose color for the label.
            color = tuple(map(int, colors[label]))
            # Draw a box.
            x2 = box[0] + box[2]
            y2 = box[1] + box[3]
            cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)

            # Draw a label name inside the box.
            cv2.putText(
                img=frame,
                text=f"{classes[label]} {score:.2f}",
                org=(box[0] + 10, box[1] + 30),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=frame.shape[1] / 1000,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return frame
    



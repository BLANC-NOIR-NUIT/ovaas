import cv2


def process_results(frame, results, thresh=0.6):
        # The size of the original frame.
        h, w = frame.shape[:2]
        # The 'results' variable is a [1, 1, 100, 7] tensor.
        results = results.squeeze()
        boxes = []
        labels = []
        scores = []
        for _, label, score, xmin, ymin, xmax, ymax in results:
            # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
            boxes.append(
                tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
            )
            labels.append(int(label))
            scores.append(float(score))


        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6
        )

        # If there are no boxes.
        if len(indices) == 0:
            return []

        # Filter detected objects.
        return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

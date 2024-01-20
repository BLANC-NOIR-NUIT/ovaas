import datetime
import cv2
import numpy as np
from openvino.runtime import Core
import collections
import time
from pathlib import Path
import functions.notebook_utils as utils
from functions.od_result import process_results
from functions.od_draw import draw_boxes

# ...


def object_detection(video_path, model_name):

    model_path = f"./model/{model_name}-0001.xml"
    # Initialize OpenVINO Core.
    core = Core()
    # Read the network and corresponding weights from a file.
    model = core.read_model(model=model_path)
    # Compile the model for the specified device or let the engine choose the best available device (AUTO).
    compiled_model = core.compile_model(model=model, device_name="CPU")
    # Get the input and output nodes.
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    # Get the input size.
    height, width = list(input_layer.shape)[1:3]

  
    try:  
        # Open the video file using OpenCV.
        cap = cv2.VideoCapture(video_path)
        # Create a VideoWriter object to save the output video.
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        #データタイムを取得
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        #videoの縦横取得
        video_width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        video_height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #作成するvideo
        result_video = cv2.VideoWriter(f"./result/result_object_detection{timestamp}.mp4", codec,30,(video_width, video_height))
        processing_times = collections.deque()

        while True:
            # Read the frame from the video.
            ret, frame = cap.read()
            if not ret:
                print("Source ended")
                break

            # If the frame is larger than full HD, reduce size to improve performance.
            scale = video_width / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            # Resize the image and change dimensions to fit neural network input.
            # データの形状を変更し、モデルの入力形状に合わせる

            input_img = cv2.resize(
                src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA
            )
            #print(input_img.shape)
            # input_img = input_img.transpose(0, 1, 2)[np.newaxis, ...]
            input_img = input_img[np.newaxis, ...]

            #print(input_img.shape)

            # Measure processing time.
            start_time = time.time()
            # Get the results.
            results = compiled_model([input_img])[output_layer]

            #print(type(results))

            stop_time = time.time()
            # Get object detection results.
            boxes = process_results(frame=frame, results=results)

            # Draw boxes on the frame.
            frame = draw_boxes(frame=frame, boxes=boxes)

            processing_times.append(stop_time - start_time)
            # Use processing times from the last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            # Write the frame to the output video.
            result_video.write(frame)

    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(e)
    finally:
        # Release the video capture and output video.
        cap.release()
        result_video.release()
        cv2.destroyAllWindows()

    return result_video

    player = None




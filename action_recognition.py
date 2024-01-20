'''
import collections
import datetime
import os
import sys
import time
from typing import Tuple, List

import cv2
import numpy as np
from IPython import display
from openvino.runtime import Core
from openvino.runtime.ie_api import CompiledModel
from functions.act_rec_fun import decoder, display_text_fnc, encoder, model_init, preprocessing

sys.path.append("../utils")
import functions.notebook_utils as utils

def action_recognition(file_path):
    flip=False, 
    skip_first_frames=600
    model_path_decoder = f"./model/action-recognition-0001-decoder.xml"
    model_path_encoder = f"./model/action-recognition-0001-encoder.xml"
    # Encoder initialization
    input_key_en, output_keys_en, compiled_model_en = model_init(model_path_encoder, 'CPU')
    # Decoder initialization
    input_key_de, output_keys_de, compiled_model_de = model_init(model_path_decoder, 'CPU')

    # Get input size - Encoder.
    height_en, width_en = list(input_key_en.shape)[2:]
    # Get input size - Decoder.
    frames2decode = list(input_key_de.shape)[0:][1]

    
    size = height_en  # Endoder input size - From Cell 5_9
    sample_duration = frames2decode  # Decoder input size - From Cell 5_7
    # Select frames per second of your source.
    fps = 30
    player = None

    try:
        # Open the video file using OpenCV.
        cap = cv2.VideoCapture(file_path)
        # Create a VideoWriter object to save the output video.
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        #データタイムを取得
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        #videoの縦横取得
        video_width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        video_height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        #作成するvideo
        output_name = f"./result/{timestamp}result_action_recognition.mp4"
        result_video = cv2.VideoWriter(output_name, codec,fps,(video_width, video_height))
        processing_times = collections.deque()
        processing_time = 0
        encoder_output = []
        decoded_labels = [0, 0, 0]
        decoded_top_probs = [0, 0, 0]
        counter = 0
        # Create a text template to show inference results over video.
        text_inference_template = "Infer Time:{Time:.1f}ms,{fps:.1f}FPS"
        text_template = "{label},{conf:.2f}%"

        while True:
            counter = counter + 1

            ret, frame = cap.read()
            if not ret:
                print("Source ended")
                break

            if video_width>video_height:
                max_frame=video_width
            else:
                max_frame=video_height
            # If the frame is larger than full HD, reduce size to improve performance.
            scale = max_frame / max(frame.shape)

            # Adaptative resize for visualization.
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Select one frame every two for processing through the encoder.
            # After 16 frames are processed, the decoder will find the action,
            # and the label will be printed over the frames.

            if counter % 2 == 0:
                # Preprocess frame before Encoder.
                (preprocessed, _) = preprocessing(frame, size)

                # Measure processing time.
                start_time = time.time()

                # Encoder Inference per frame
                encoder_output.append(encoder(preprocessed, compiled_model_en))
               # Decoder inference per set of frames
                # Wait for sample duration to work with decoder model.
                if len(encoder_output) == sample_duration:
                    decoded_labels, decoded_top_probs = decoder(encoder_output, compiled_model_de)
                    encoder_output = []

                # Inference has finished. Display the results.
                stop_time = time.time()

                # Calculate processing time.
                processing_times.append(stop_time - start_time)

                # Use processing times from last 200 frames.
                if len(processing_times) > 200:
                    processing_times.popleft()

                # Mean processing time [ms]
                processing_time = np.mean(processing_times) * 1000
                fps = 1000 / processing_time

            # Visualize the results.
            for i in range(0, 3):
                display_text = text_template.format(
                    label=decoded_labels[i],
                    conf=decoded_top_probs[i] * 100,
                )
                display_text_fnc(frame, display_text, i)

            display_text = text_inference_template.format(Time=processing_time, fps=fps)
            display_text_fnc(frame, display_text, 3)

            # Use this workaround if you experience flickering.
            result_video.write(frame)


    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # Any different error
    except RuntimeError as e:
        print(e)
    finally:
        # Release the video capture and output video.
        cap.release()
        result_video.release()
        cv2.destroyAllWindows()
'''

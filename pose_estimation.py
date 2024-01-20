import collections
import datetime
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from IPython import display
from numpy.lib.stride_tricks import as_strided
from openvino.runtime import Core
from functions.Decoder import OpenPoseDecoder
from functions.notebook_utils import VideoPlayer
from functions.ProcessResults import process_results
from functions.DrawPoseOverlays import draw_poses
from functions import notebook_utils as utils
sys.path.append("../utils")

#姿勢推定
def PoseEstimation(file_path):

    
    # Initialize OpenVINO Runtime
    core = Core()

    # Read the network from a file.
    path_xml = f"./model/human-pose-estimation-0001.xml"
    path_bin = f"./model/human-pose-estimation-0001.bin"
    model = core.read_model(model=path_xml, weights=path_bin)

    # Let the AUTO device decide where to load the model (you can use CPU, GPU as well).
    compiled_model = core.compile_model(model=model, device_name="CPU", config={"PERFORMANCE_HINT": "LATENCY"})

    # Get the input and output names of nodes.
    input_layer = compiled_model.input(0)
    output_layers = compiled_model.outputs

    # Get the input size.
    height, width = list(input_layer.shape)[2:]
    input_layer.any_name, [o.any_name for o in output_layers]
    
    decoder = OpenPoseDecoder()
    
    # Main processing function to run pose estimation.
    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")

# made by uchida
    try:
        cap = cv2.VideoCapture(file_path)
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        video_width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        video_height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        output_name = f"./result/{timestamp}result_human_pose_estimation.mp4"
        result_video = cv2.VideoWriter(output_name, codec, fps, (video_width, video_height))
        processing_times = collections.deque()
        while True:
            # Grab the frame.
            ret, frame = cap.read()
            if not ret :
                print("Source ended")
                break

            if video_width>video_height:
                max_frame=video_width
            else:
                max_frame=video_height
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = max_frame / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            #print(f"frame1: {type(frame)}")
            
            # Resize the image and change dims to fit neural network input.
            # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # Create a batch of images (size = 1).
            input_img = input_img.transpose((2,0,1))[np.newaxis, ...]

            # Measure processing time.
            start_time = time.time()
            # Get results.
            results = compiled_model([input_img])
            stop_time = time.time()

            pafs = results[pafs_output_key]
            heatmaps = results[heatmaps_output_key]
            # Get poses from network results.
            poses, scores = process_results(frame, pafs, heatmaps, compiled_model=compiled_model)

            # Draw poses on a frame.
            frame = draw_poses(frame, poses, 0.1)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # mean processing time [ms]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)

            
            #print(f"frame2: {type(frame)}")
            # Use this workaround if there is flickering.
            # if use_popup:
            
            result_video.write(frame)
            

         
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:

        result_video.release()
        cv2.destroyAllWindows()   
    return result_video    

def run_pose_estimation(source=0, flip=False, use_popup=False, skip_first_frames=0):
            # ランタイムの初期化
        core = Core()

        # ファイルの読み込み
        path_xml = f"./model/human-pose-estimation-0001.xml"
        path_bin = f"./model/human-pose-estimation-0001.bin"
        model = core.read_model(model=path_xml, weights=path_bin)

        # モデル読み込み
        compiled_model = core.compile_model(model=model, device_name="CPU", config={"PERFORMANCE_HINT": "LATENCY"})

        # Get the input and output names of nodes.
        input_layer = compiled_model.input(0)
        output_layers = compiled_model.outputs

        # Get the input size.
        height, width = list(input_layer.shape)[2:]
        input_layer.any_name, [o.any_name for o in output_layers]

        decoder = OpenPoseDecoder()
        pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
        heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")
        player = None
        try:
            # Create a video player to play with target fps.
            player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
            # Start capturing.
            player.start()
            if use_popup:
                title = "Press ESC to Exit"
                cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

            processing_times = collections.deque()

            while True:
                # Grab the frame.
                frame = player.next()
                if frame is  None :
                    print("Source ended")
                    break
                # If the frame is larger than full HD, reduce size to improve the performance.
                scale = 1280 / max(frame.shape)
                if scale < 1:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                # Resize the image and change dims to fit neural network input.
                # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
                input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                # Create a batch of images (size = 1).
                input_img = input_img.transpose((2,0,1))[np.newaxis, ...]

                # Measure processing time.
                start_time = time.time()
                # Get results.
                results = compiled_model([input_img])
                stop_time = time.time()

                pafs = results[pafs_output_key]
                heatmaps = results[heatmaps_output_key]
                # Get poses from network results.
                poses, scores = process_results(frame, pafs, heatmaps,compiled_model=compiled_model)

                # Draw poses on a frame.
                frame = draw_poses(frame, poses, 0.1)

                processing_times.append(stop_time - start_time)
                # Use processing times from last 200 frames.
                if len(processing_times) > 200:
                    processing_times.popleft()

                _, f_width = frame.shape[:2]
                # mean processing time [ms]
                processing_time = np.mean(processing_times) * 1000
                fps = 1000 / processing_time
                cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
                            cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)

                # Use this workaround if there is flickering.
                "ここを変えたい！"
                if use_popup:
                    cv2.imshow(title, frame)
                    key = cv2.waitKey(1)
                    # escape = 27
                    if key == 27:
                        break
                else:
                    # Encode numpy array to jpg.
                    _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                    # Create an IPython image.
                    i = display.Image(data=encoded_img)
                    # Display the image in this notebook.
                    display.clear_output(wait=True)
                    display.display(i)
        # ctrl-c
        except KeyboardInterrupt:
            print("Interrupted")
        # any different error
        except RuntimeError as e:
            print(e)
        finally:
            if player is not None:
                # Stop capturing.
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()

run_pose_estimation(source=0, flip=True, use_popup=True)




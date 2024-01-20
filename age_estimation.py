

import datetime
import cv2
from openvino.runtime import Core
#@@@#from openvino.inference_engine import IECore
import numpy as np
import os
import shutil
 
def Age_Estimation(video_path, model_name):
    # Initialize OpenVINO Inference Engine
    #@@@#ie = IECore()
    core = Core()
    # Load the pre-trained models for face detection
    #@@@#face_detection_model = ie.read_network(model="model/face-detection-adas-0001.xml")
    face_detection_model = core.read_model(model="model/face-detection-adas-0001.xml")
    #@@@#compiled_face_detection_model = ie.load_network(network=face_detection_model, device_name="CPU")
    compiled_face_detection_model = core.compile_model(face_detection_model, device_name="CPU")

    # Load the pre-trained models for age and gender recognition
    #@@@#age_gender_model = ie.read_network(model="model/age-gender-recognition-retail-0013.xml")
    age_gender_model =  core.read_model(model="model/age-gender-recognition-retail-0013.xml")
    #@@@#compiled_age_gender_model = ie.load_network(network=age_gender_model, device_name="CPU")
    compiled_age_gender_model = core.compile_model(age_gender_model, device_name="CPU")

    input_layer = compiled_face_detection_model.input(0)
    output_layer = compiled_face_detection_model.outputs

    # Get the input size.
    height, width = list(input_layer.shape)[2:]

    #追加
    try:
        # Open the input video file
        cap = cv2.VideoCapture(video_path)
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        # Get video frame properties
        #@@@#frame_width = int(cap.get(3))
        video_width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        #@@@frame_height = int(cap.get(4))
        video_height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #@@@#frame_fps = int(cap.get(5))
        #timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        # Define the codec and create a VideoWriter object for the output video
        output_video_path = video_path.replace(".mp4", "_result_face_age.mp4")
        out = cv2.VideoWriter(output_video_path,codec, 30, (video_width, video_height))



        while True:
            # Read a frame from the input video
            ret, frame = cap.read()
            if not ret:
                print("Source ended")
                break

            # Get the shape of the input for face detection
            #@@@#input_shape = face_detection_model.input_info["data"].input_data.shape
            #@@@#height, width = input_shape[2], input_shape[3]

            # Preprocess the input image for face detection
            input_image = cv2.resize(frame, (width, height))
            input_image = input_image.transpose((2, 0, 1))  # Change the channel order
            input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

            # Perform inference for face detection
            #@@@#face_detection_result = compiled_face_detection_model.infer(inputs={"data": input_image})
            face_detection_result = compiled_face_detection_model([input_image])[output_layer[0]]
            # Get the coordinates of detected faces
            detections = face_detection_result[0][0][np.where(face_detection_result[0][0][:,2] > 0.5)]

            for detection in detections:
                confidence = detection[2]
                if confidence > 0.3:  # Adjust the confidence threshold as needed
                    x1, y1, x2, y2 = (detection[3:7] * np.array([video_width, video_height, video_width, video_height])).astype(int)

                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Crop the detected face
                    face = frame[y1:y2, x1:x2]

                    # Resize the face to the input size expected by the age-gender model
                    input_face = cv2.resize(face, (62, 62))

                    # Preprocess the input image for age and gender recognition
                    input_face = input_face.transpose((2, 0, 1))  # Change the channel order
                    input_face = np.expand_dims(input_face, axis=0)  # Add batch dimension

                    # Perform inference on the face for age and gender
                    #@@@#age_gender_result = compiled_age_gender_model.infer(inputs={"data": input_face})
                    age_gender_result = compiled_age_gender_model(inputs={"data": input_face})

                    # Get age and gender result
                    gender = "Male" if age_gender_result["prob"][0][0][0] < 0.7 else "Female"
                    age = age_gender_result["age_conv3"][0][0][0][0] * 100

                    # Display age and gender on the frame
                    text = f"Age: {age:.1f}, {gender}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the frame to the output video
            out.write(frame)
    finally:
        # Release the input video capture and output video writer
        cap.release()
        out.release()

    # Move the output video file to the specified output directory



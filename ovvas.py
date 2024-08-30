
#from age_gender_estimation import age_gender_estimation
from pose_estimation import PoseEstimation, pose_estimation_camera
from object_d import object_detection, object_detection_camera


def do_ovaas(file_path, model_name):
        if model_name == "object-detection":
            object_detection(file_path)

        elif model_name == "human-pose-estimation":
            PoseEstimation(file_path)

        #elif model_name == "face-age-estimation":
        #   age_gender_estimation(file_path)


def do_ovaas_camera(model_name):
        if model_name == "object-detection":
            object_detection_camera()

        elif model_name == "human-pose-estimation":
            pose_estimation_camera()

        #elif model_name == "face-age-estimation":
         #   age_gender_estimation(file_path)


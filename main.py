from src.input_feeder import InputFeeder
from src.face_detection import Face_Detect_Model
from src.facial_landmarks_detection import Landmark_Detect_Model
from src.head_pose_estimation import Pose_Detect_Model
from src.gaze_estimation import Gaze_Estimate_Model
from src.mouse_controller import MouseController
from argparse import ArgumentParser
from math import sin, cos, pi
import numpy as np
import cv2, time, sys, logging

# Create and parse command line arguments
def build_argparser():
    """
    Parse command line arguments for model paths, mouse precision/speed config, target device, etc..
    """
    
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file, or 'cam' for using the PC's webcam.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, or MYRIAD is acceptable. Default: CPU")
    parser.add_argument("-fm", "--face_model", type=str, 
                        default=r"models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001",
                        help="Path to Face Detection model (without extension). Default: models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 ")
    parser.add_argument("-lm", "--landmarks_model", type=str, 
                        default=r"models\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009",
                        help="Path to Facial Landmarks Detection model (without extension). Default: models\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009 ")
    parser.add_argument("-gm", "--gaze_model", type=str, 
                        default=r"models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002",
                        help="Path to Gaze Estimation model (without extension). Default: models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002 ")
    parser.add_argument("-pm", "--pose_model", type=str, 
                        default=r"models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001",
                        help="Path to Head Pose Estimation model (without extension). Default: models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold used at face detection step. Default: 0.5.")
    parser.add_argument("-sp", "--mouse_speed", type=str, default='fast',
                        help="Speed at which the mouse will move when changing location ('fast, 'medium', 'slow'). Default: medium")
    parser.add_argument("-prec", "--mouse_precision", type=str, default='medium',
                        help="Controls how long the mouse will move in the inferred direction ('low', 'medium', 'high'). Default: medium")
                        

    return parser

def crop_image(img, x_min, y_min, x_max, y_max):
    '''
    Return only the subset of the image with the specified bounding box coordinates.
    '''

    return img[y_min:y_max, x_min:x_max]


def main():

    # Init logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    # Get command line arguments
    args = build_argparser().parse_args()

    # Start InputFeeder to handle stream
    model_input = args.input
    input_extension = model_input.split('.')[-1]

    if input_extension in ['jpg', 'png']:
        feeder = InputFeeder('image', model_input)
    elif input_extension in ['mp4', 'avi']:
        feeder = InputFeeder('video', model_input)
    elif model_input == 'cam':
        feeder = InputFeeder('cam')
    else:
        logger.critical('Invalid input type supplied, terminating the app ... ')
        sys.exit()

    # Load the data
    feeder.load_data()

    # Initialize all the models
    face_model = Face_Detect_Model(model_name= args.face_model, device=args.device, threshold=args.prob_threshold)
    landmarks_model = Landmark_Detect_Model(model_name=args.landmarks_model, device=args.device)
    pose_model = Pose_Detect_Model(model_name=args.pose_model, device=args.device)
    gaze_model = Gaze_Estimate_Model(model_name=args.gaze_model, device=args.device)

    # Start Mouse Controller
    mouse_controller = MouseController(precision=args.mouse_precision, speed=args.mouse_speed)

    # Run inference on the streams
    # Iterate over batches and apply the different models.
    for i, batch in enumerate(feeder.next_batch()):
        if batch is None:
            # End of Stream
            logger.info('End of Stream, terminating the app ...')
            feeder.close()
            sys.exit()
            break
        else:
            # Run face detection model and get cropped face image
            x_min_face, y_min_face, x_max_face, y_max_face = face_model.predict(batch)

            if x_min_face is not None:
                
                try:
                    # Crop face image and visualize bounding box in original frame
                    face_img = crop_image(batch, x_min_face, y_min_face, x_max_face, y_max_face)
                    batch = cv2.rectangle(batch, (x_min_face, y_min_face), (x_max_face, y_max_face), (0,255,0), 1)

                    # Run landmark detection and visualize eye bounding boxes
                    left_eye_box, right_eye_box, landmark_coords = landmarks_model.predict(face_img)
                    x_min_left, y_min_left, x_max_left, y_max_left = left_eye_box
                    x_min_right, y_min_right, x_max_right, y_max_right = right_eye_box
                    batch = cv2.rectangle(batch, (x_min_left + x_min_face , y_min_left + y_min_face  ), (x_max_left + x_min_face, y_max_left + y_min_face), (255, 0, 0), 1)
                    batch = cv2.rectangle(batch, (x_min_right + x_min_face , y_min_right + y_min_face  ), (x_max_right + x_min_face, y_max_right + y_min_face), (255,0,0), 1)

                    # Run head pose detection 
                    head_pose_angles = pose_model.predict(face_img)

                    # Get cropped eye images and run gaze estimation
                    left_eye_img = crop_image(face_img, *left_eye_box)
                    right_eye_img = crop_image(face_img, *right_eye_box)
                    gaze_vector = gaze_model.predict(left_eye_img, right_eye_img, head_pose_angles)

                    # Show gaze vector direction
                    x_left_eye, y_left_eye = landmark_coords[0]
                    x_right_eye, y_right_eye = landmark_coords[1]
                    x_left_orig, y_left_orig = x_left_eye + x_min_face, y_left_eye + y_min_face
                    x_right_orig, y_right_orig = x_right_eye + x_min_face, y_right_eye + y_min_face
                    batch = cv2.arrowedLine(batch, (x_left_orig, y_left_orig), (int(x_left_orig + 200*gaze_vector[0]), int(y_left_orig - 200*gaze_vector[1]) )  , (0,255,255), 1) 
                    batch = cv2.arrowedLine(batch, (x_right_orig, y_right_orig), (int(x_right_orig + 200*gaze_vector[0]), int(y_right_orig - 200*gaze_vector[1]) )  , (0,255,255), 1) 
                    
                    # Move mouse and show original image to see if it makes sense
                    mouse_controller.move(gaze_vector[0], gaze_vector[1])
                    cv2.imshow('test',batch)
                    cv2.waitKey()
                except:
                    logger.critical('Error when running the inference pipeline, terminating the app ...')
                    sys.exit()

            else:
                # Print warning
                logger.warning('Face not detected in current frame, mouse will stay in its position.')


main()
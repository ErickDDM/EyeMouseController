# Mouse Computer Pointer Controller

This proyect uses several deep learning models to estimate in which direction a person is looking at (either from a prerecorded video, a webcam stream or an image) and then moves the mouse some distance in the estimated looking direction. Overall, the app allows us to control our mouse location using only our head pose and our eyes. Additionally the app show the frame with annotations to show interesting model outputs like the estimated gaze direction vector and face/eye bounding boxes detections. The proyect was developed using the OpenVINO toolkit, which allows us to run trained models efficiently on the edge in several different computing devices.

<img src="imgs/demo.png" alt="drawing" width="800"/>

All the models that were used to develop this proyect belong to the OpenVINO pre-trained model zoo and can be consulted in the following URLS:
* [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Recognition](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [Gaze Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)


 The general logic that is being executed underneath with this deep learning models (leaving aside other details like input and output preproccesing) is the following:

<img src="https://video.udacity-data.com/topher/2020/April/5e923081_pipeline/pipeline.png" alt="drawing" width="500"/>

## Project Set Up and Installation

For running this proyect it is necessary to execute the following steps:

* Have a working python installtion in your system.
* [Install the OpenVINO toolkit on your computer](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html).
* Install virtualenv:  `pip install virtualenv`.
* Create a new python virtual environment for installing all the necessary dependencies: `virtualenv <env-name>`
* Activate the virtual environment: `<env-name>\Scripts\activate` (windows)
* Install the required dependencies using the supplied `requirements.txt` file: `pip install -r requirements.txt`
* [Intialize the OpenVINO toolkit environment variables](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html#set-the-environment-variables).
* Execute `main.py` with the appropriate commandline arguments (use -h when executing the app for getting more information about this arguments).

To save memory the pre-trained model files are not included in this repository. You need to download this models yourself  into the `models` directory by using the [model downloader tool](https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html) provided with the OpenVINO toolkit. For this you have to use the `downloader.py` script in the following way:
 
 `python downloader.py --name face-detection-adas-binary-0001 -o <path-to-models-folder>`

 To download the other models you only have to change the `--name` argument with its corresponding value. You can get the model names by looking at the model's documentation linked in the previous section.

The final proyect structure that you should have after downloading the required pre-trained models is the following (with the exception of the folder that contains the python virtual environment):

<img src="imgs/folder_paths.png" alt="drawing" width="500"/>

## Demo
The simplest way to run the demo application is to download all the required models with the recommended proyect structure and then use one of the following commands to execute the application (after initializing the OpenVINO environment variables and activating the virtual environment):
* `python main.py -i bin\demo.mp4` (for running the app with a supplied pre-recorded video of a person).
* `python main.py -i cam` (for running the app with a live video stream of your webcam).

If you used the suggested proyect structure and execute one of the previous commands, the default value of the command line arguments will be such that the app will be executed in the CPU with FP32 precision models. See the next section for a description of all the available command line arguments.

## Documentation

The model uses several command line arguments to control various aspects of it's behaviour. The application has to be executed in the following way:

`main.py [-h] -i INPUT [-d DEVICE] [-fm FACE_MODEL]
               [-lm LANDMARKS_MODEL] [-gm GAZE_MODEL] [-pm POSE_MODEL]
               [-pt PROB_THRESHOLD] [-sp MOUSE_SPEED] [-prec MOUSE_PRECISION]`

The different command line arguments available in the application and their default values are the following:
  * -h, --help            show this help message and exit
  * -i INPUT, --input INPUT
                        Path to image or video file, or 'cam' for using the
                        PC's webcam.
  * -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, or
                        MYRIAD is acceptable. Default: CPU
  * -fm FACE_MODEL, --face_model FACE_MODEL
                        Path to Face Detection model (without extension).
                        Default: models\intel ace-detection-adas-
                        binary-0001\FP32-INT1 ace-detection-adas-binary-0001
  * -lm LANDMARKS_MODEL, --landmarks_model LANDMARKS_MODEL
                        Path to Facial Landmarks Detection model (without
                        extension). Default: models\intel\landmarks-
                        regression-retail-0009\FP32\landmarks-regression-
                        retail-0009
  * -gm GAZE_MODEL, --gaze_model GAZE_MODEL
                        Path to Gaze Estimation model (without extension).
                        Default: models\intel\gaze-estimation-
                        adas-0002\FP32\gaze-estimation-adas-0002
  * -pm POSE_MODEL, --pose_model POSE_MODEL
                        Path to Head Pose Estimation model (without
                        extension). Default: models\intel\head-pose-
                        estimation-adas-0001\FP32\head-pose-estimation-
                        adas-0001
  * -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold used at face detection step.
                        Default: 0.5.
  * -sp MOUSE_SPEED, --mouse_speed MOUSE_SPEED
                        Speed at which the mouse will move when changing
                        location ('fast, 'medium', 'slow'). Default: fast
  * -prec MOUSE_PRECISION, --mouse_precision MOUSE_PRECISION
                        Controls how long the mouse will move in the inferred
                        direction ('low', 'medium', 'high'). Default: medium


## Benchmarks

To perform some simple benchmarks each model's load time and execution time (inference + preprocessing) was analyzed with both 32 and 16 bits float precision models using the CPU and the IGPU of [my computer](https://support.hp.com/mx-es/document/c05599277) as target devices. The time measurements are in miliseconds unless stated otherwise and are approximates obtained by averaging and rounding the first 4 inferences from the models.

### CPU

| Model              | Load Time (32 Bits) | Execution Time (32 Bits) | Load Time (16 Bits) | Execution Time (16 Bits) |
|--------------------|---------------------|--------------------------|---------------------|--------------------------|
| Face Detection     | 252                 | 11                       | -                   | -                        |
| Landmark Detection | 154                 | 2                        | 150                 | 1                        |
| Pose Estimation    | 139                 | 2                        | 142                 | 2                        |
| Gaze Estimation    | 153                 | 2                        | 165                 | 2                        |


### IGPU

| Model              | Load Time (32 Bits) | Execution Time (32 Bits) | Load Time (16 Bits) | Execution Time (16 Bits) |
|--------------------|---------------------|--------------------------|---------------------|--------------------------|
| Face Detection     | 26 s                | 15                       | -                   | -                        |
| Landmark Detection | 4 s                 | 3                        | 5 s                 | 2                     |
| Pose Estimation    | 5 s                 | 3                        | 6 s                 | 2                        |
| Gaze Estimation    | 7 s                 | 3                        | 7 s                 | 2                        |


## Results
As expected, the benchmarks show that the model loading time is much bigger in the IGPU case in comparison to the CPU case (around 2 orders of magnitude). Even tough in general the load time required for running our app in the device IGPU's is much bigger, being able to run models on this device can bevery useful because:
* It could be that the target device CPU's is already saturated, in which case using the IGPU for running our models will free the CPU for performing other tasks.
* Many applications don't need to be able to start instantly. 

In general we can see that the whole iference pipeline takes around 10/20 ms to be completed, which are really good numbers if we are trying to work in developing a real time processing edge application. There are not very clear differences when using 16 vs 32 bit precision models. These could be either because this differences are not really big, or because the preprocessing time (that is the same in both cases) is taking much longer to run (additional testing would be necessary for analyzing this possibilities). Even tough in theory the 16 bit models should run in less time, the wide difference between the optimizations that can be performed in different hardware devices make this a rule that is not set in stone.



## Maximal Confidence Face Prediction Detection
The way that the face model detection is being called is such that even if there are various persons in a frame the model will only process the detection that has the highest confidence. We also check that this 'best prediction' has a confidence that is above a user specifed probability threshold. 

### Edge Cases
The modification that was described in the previous section make our app more robust to cases in which there are several people in the frame (the 'main' user should be the one that is closer to the cam and generally will be the one one with the highest confidence prediction). Testing also if this maximum confidence is above a user specified threshold allows the application to not move the mouse if if has not detected any face with that not has at least the required confidence.

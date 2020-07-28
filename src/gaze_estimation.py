from openvino.inference_engine import IECore
import numpy as np
import cv2

class Gaze_Estimate_Model:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU'):
        '''
        Intialize instance variables and load the model with the supplied CL arguments.
        '''

        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device

        try:
            self.core = IECore()
            self.model= self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network.")

        # Handle multiple model inputs
        self.left_eye_input="left_eye_image"
        self.right_eye_input="right_eye_image"
        self.angles_input="head_pose_angles"
        self.eye_input_shape=self.model.input_info['left_eye_image'].input_data.shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        self.load_model()

    def load_model(self):
        '''
        Load the already read model with the specified device type.
        '''
        
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def predict(self, left_eye_img, right_eye_img, pose_angles):
        '''
        Modularize the whole preprocess input / run inference / preprocess output cycle.
        '''

        # Preprocess the input, run the net, and return the angles
        proc_left_eye, proc_right_eye, proc_pos_angles = self.preprocess_input(left_eye_img, right_eye_img, pose_angles)
        input_dict={self.left_eye_input:proc_left_eye, self.right_eye_input:proc_right_eye, self.angles_input:proc_pos_angles}
        out = self.net.infer(input_dict)[self.output_name]
        gaze_vector = self.preprocess_output(out)

        return gaze_vector


    def preprocess_input(self, left_eye_img, right_eye_img, pose_angles):
        '''
        Preprocess the inputs so that we can run inference on them.
        '''

        # Left Eye:
        proc_left_eye = cv2.resize(left_eye_img, (self.eye_input_shape[3], self.eye_input_shape[2]))
        proc_left_eye = np.transpose(proc_left_eye, (2,0,1))
        proc_left_eye = proc_left_eye[np.newaxis, :]

        # Right Eye:
        proc_right_eye = cv2.resize(right_eye_img, (self.eye_input_shape[3], self.eye_input_shape[2]))
        proc_right_eye = np.transpose(proc_right_eye, (2,0,1))
        proc_right_eye = proc_right_eye[np.newaxis, :]

        # Angles
        proc_pose_angles = np.array(pose_angles).reshape(1,3)

        return proc_left_eye, proc_right_eye, proc_pose_angles


    def preprocess_output(self, outputs):
        '''
        Flatten the output into a typical 1D array.
        '''
        
        return outputs.flatten()

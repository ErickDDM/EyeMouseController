from openvino.inference_engine import IECore
import numpy as np
import cv2

class Pose_Detect_Model:
    '''
    Class for the Head Pose Estimation Model.
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

        # Handle multiple model outputs later
        self.yaw_name="angle_y_fc"
        self.pitch_name="angle_p_fc"
        self.roll_name="angle_r_fc"
        self.input_name=next(iter(self.model.input_info))
        self.input_shape=self.model.input_info[self.input_name].input_data.shape

        self.load_model()


    def load_model(self):
        '''
        Load the already read model with the specified device type.
        '''
        
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def predict(self, image):
        '''
        Modularize the whole preprocess input / run inference / preprocess output cycle.
        '''

        # Preprocess the input, run the net, and return the angles
        proc_img = self.preprocess_input(image)
        input_dict={self.input_name:proc_img}
        out = self.net.infer(input_dict)
        yaw_angle, pitch_angle, roll_angle = self.preprocess_output(out)

        return yaw_angle, pitch_angle, roll_angle


    def preprocess_input(self, image):
        '''
        Preprocess the input image so that we can run the network on it.
        '''

        # Remember  that the resize function takes the width first
        proc_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        proc_frame = np.transpose(proc_frame, (2,0,1))
        proc_frame = proc_frame[np.newaxis, :]

        return proc_frame


    def preprocess_output(self, outputs):
        '''
        Return a tuple that contains the 3 output angles.
        '''
        
        return float(outputs[self.yaw_name]), float(outputs[self.pitch_name]), float(outputs[self.roll_name])

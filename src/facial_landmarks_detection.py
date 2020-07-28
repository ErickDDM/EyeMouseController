from openvino.inference_engine import IECore
import numpy as np
import cv2

class Landmark_Detect_Model:
    '''
    Class for the Face Landmark Detection Model.
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

        self.input_name=next(iter(self.model.input_info))
        self.input_shape=self.model.input_info[self.input_name].input_data.shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        self.load_model()


    def load_model(self):
        '''
        Load the already read model with the specified device type.
        '''
        
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def predict(self, image):
        '''
        Modularize the whole preprocess input / infer / preprocess output cycle.
        '''

        # Preprocess the input, run the net, and return the face coordinates
        proc_img = self.preprocess_input(image)
        input_dict={self.input_name:proc_img}
        out = self.net.infer(input_dict)[self.output_name].flatten()
        left_eye_box, right_eye_box, landmark_coords = self.preprocess_output(out, image.shape)
        
        return left_eye_box, right_eye_box, landmark_coords


    def preprocess_input(self, image):
        '''
        Preprocess the input image so that we can run the network on it.
        '''
        # Remember  that the resize function takes the width first
        proc_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        proc_frame = np.transpose(proc_frame, (2,0,1))
        proc_frame = proc_frame[np.newaxis, :]

        return proc_frame


    def preprocess_output(self, outputs, orig_input_shape, eye_width_factor=0.3):
        '''
        Get the landmark coordinates in the original input image space and output the bounding boxes of the 
        left and right eyes.
        '''

        # PD: I had to play with some code before to find out what X,Y pair corresponds to each landmark.
        # The first tuple is the subject's right eye and the second tuple is the subject's left eye.

        # Get width and height of original image
        orig_height = orig_input_shape[0]
        orig_width = orig_input_shape[1]

        # Generate list of landmark X,Y pairs
        landmark_coords = []
        for i in range(0,10,2):
            landmark_coords.append(outputs[i:i+2])

        # Transform Coords to original input dimensions
        landmark_orig_coords = [ (int(landmark[0] * orig_width), int(landmark[1] * orig_height))  for landmark in landmark_coords]

        # Get Left and right eye coords
        right_eye_coords = landmark_orig_coords[0]
        left_eye_coords = landmark_orig_coords[1]

        # Crop left and right eye by creating square boxes around the landmarks
        # The eye_width_factor variable is used for defining the size of the square size relative to the total width size
        # Min and max are for ensuring that we don't go outside the possible axes indices
        half_sqr_size = int(orig_width*eye_width_factor/2)
        left_eye_box = [max(left_eye_coords[0] - half_sqr_size, 0), max(left_eye_coords[1] - half_sqr_size, 0),
                        min(left_eye_coords[0] + half_sqr_size, orig_width-1), min(left_eye_coords[1] + half_sqr_size, orig_height-1)]
        right_eye_box = [max(right_eye_coords[0] - half_sqr_size,0), max(right_eye_coords[1] - half_sqr_size, 0),
                        min(right_eye_coords[0] + half_sqr_size, orig_width -1),  min(right_eye_coords[1] + half_sqr_size, orig_height-1)]

        return left_eye_box, right_eye_box, landmark_orig_coords

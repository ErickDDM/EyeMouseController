from openvino.inference_engine import IECore
import numpy as np
import cv2

class Face_Detect_Model:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU',threshold=0.5):
        '''
        Intialize instance variables and load the model with the supplied CL arguments.
        '''

        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.core = IECore()
            self.model= self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network.")

        self.input_name=next(iter(self.model.input_info))
        self.input_shape=self.model.input_info[self.input_name].input_data.shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        # Load the model
        self.load_model()


    def load_model(self):
        '''
        Load the already read model with the specified device type.
        '''
        
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def predict(self, image):
        '''
        Modularize the whole process input / make inference / process output cycle.
        '''

        # Preprocess the input, run the net, and return the face coordinates
        proc_img = self.preprocess_input(image)
        input_dict={self.input_name:proc_img}
        out = self.net.infer(input_dict)[self.output_name]
        x_min, y_min, x_max, y_max = self.preprocess_output(out, image.shape)
        
        return x_min, y_min, x_max, y_max


    def preprocess_input(self, image):
        '''
        Preprocess network input so that we can run the network correctly.
        '''

        # Remember  that the resize function takes the width first
        proc_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        proc_frame = np.transpose(proc_frame, (2,0,1))
        proc_frame = proc_frame[np.newaxis, :]

        return proc_frame

    def preprocess_output(self, outputs, orig_input_shape):
        '''
        Get detection with biggest confidence and output its bounding box in the oiriginal image coordinates space.
        if its confidence its bigger than the user specified CLI threshold.
        '''

        # Get width and height of original image
        orig_height = orig_input_shape[0]
        orig_width = orig_input_shape[1]
        
        # Get output with biggest confidence
        best_detection_id = np.argmax(outputs[0,0,:,2])
        best_detection = outputs[0,0, best_detection_id]
    
        # If detection is over our confidence threshold 
        if best_detection[2] > self.threshold:
            x_min, y_min, x_max, y_max = best_detection[3:]

            # Transform detection coordinates to the original image input space
            x_min = int(x_min * orig_width)
            x_max = int(x_max * orig_width)
            y_min = int(y_min * orig_height)
            y_max = int(y_max * orig_height)

            return x_min , y_min, x_max, y_max

        else:
            return None, None, None, None

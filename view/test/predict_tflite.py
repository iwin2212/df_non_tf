from tflite_runtime.interpreter import Interpreter
import numpy as np

class tflite_model:
    def __init__(self, model_path):
        # load model
        self.model = Interpreter(model_path=model_path)
        self.model.allocate_tensors()

    def predict(self, input_data):
        size = input_data.shape[0]
        # Get input and output tensors.
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        _, b, c, d = input_details[0]["shape"]
        self.model.resize_tensor_input(input_details[0]['index'], (size, b,c,d))
        self.model.allocate_tensors()
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        self.model.set_tensor(input_details[0]['index'], input_data)
        self.model.invoke()
        output_data = self.model.get_tensor(output_details[0]['index'])
        return output_data

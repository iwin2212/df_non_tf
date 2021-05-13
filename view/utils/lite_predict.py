from tflite_runtime.interpreter import Interpreter
from const import model_path, input_shape_x, input_shape_y
import time

# Load the TFLite model
interpreter = Interpreter(model_path)
# interpreter.allocate_tensors()


def predict_tfmodel(input_data):
    size = input_data.shape[0]
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(
        input_details[0]['index'], (size, input_shape_x, input_shape_y, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (size, 384))
    # Load the allocate tensors.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

import tensorflow as tf

def predict_tfmodel(tfmodel, input_data):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tfmodel)
    interpreter.allocate_tensors()
    size = input_data.shape[0]

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'], (size, 128, 128, 1))
    interpreter.resize_tensor_input(output_details[0]['index'], (size, 2))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
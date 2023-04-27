'''
Author: Aditi Shanmugam (aditishanmugam1@gmail.com)
Date: 25-04-2023

This script runs the inference pipeline to identify the image overlay type. 
'''

import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import os
import pandas as pd


class runInference:

    def __init__(self, model_path, data_path, csv_path):
        self.model_path = model_path
        self.data_path = data_path
        self.Image_Paths = os.listdir(data_path)
        self.Labels = []
        self.Predictions = []
        # self.Results = []
        self.confidence = []
        self.classid = []
        self.csv_path = csv_path 

    def load_image(self, image_path):
        img_pth = self.data_path + '/' + image_path
        input_data = tf.io.gfile.GFile(img_pth, 'rb').read()
        image = tf.io.decode_image(input_data, channels=3, dtype=tf.uint8)
        image = tf.image.resize(image, (224, 224),
                                method='bilinear',
                                antialias=True)

        return tf.expand_dims(tf.cast(image, tf.float32), 0).numpy()

    def modelInfer(self):
        prediction_mappings = ['Front', 'Frontleft', 'Frontright', 'Rear', 'Rearleft', 'Rearright']
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']
        count = 0
        print(self.Image_Paths)
        for i in range(len(self.Image_Paths)):
            if self.Image_Paths[i].lower().endswith(('.png', '.jpg', '.jpeg')):
                # self.curr_img = self.data_path+
                image = self.load_image(self.Image_Paths[i])
                input_index = interpreter.get_input_details()[0]['index']

                interpreter.set_tensor(input_index, image)
                interpreter.invoke()

                output_details = interpreter.get_output_details()
                output = interpreter.get_tensor(output_details[0]['index'])

                output_type = output_details[0]['dtype']
                if output_type == np.int8:
                    output_scale, output_zero_point = output_details[0][
                        'quantization']
                    print("Raw output scores:", output)
                    print("Output scale:", output_scale)
                    print("Output zero point:", output_zero_point)
                    output = output_scale * (output.astype(np.float32) -
                                             output_zero_point)

                # self.Results.append(np.argmax(output[0]))
                self.confidence.append((np.max(tf.nn.softmax(output[0]))))
                self.Labels.append(self.Image_Paths[i])
                self.classid.append(prediction_mappings[np.argmax(output[0])])
                print()
                count += 1

        self.save_results()
        print('Processed ' + str(count) + ' images.')

    def save_results(self):
        inference_data = {
            'Image_file': self.Labels,
            'Predicted_Labels': self.classid,
            'Prediction_confidence': self.confidence
        }
        df = pd.DataFrame(inference_data)
        df.to_csv(self.csv_path)
        print('Saved inference results at: ' + str(self.csv_path))


def main(args: argparse.Namespace):
    print("Creating CSV using data")
    print(os.listdir(args.input_data_path))

    runInferenceObj = runInference(args.inference_model_path,
                                   args.input_data_path, args.output_csv_name)
    runInferenceObj.modelInfer()

    print('Exiting inference script.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference-ImageOverlay')
    parser.add_argument('-inference_model_path',
                        help='Full path to the model',
                        required=True,
                        type=str)
    parser.add_argument('-input_data_path',
                        help='Full path to the folder containing Images',
                        required=True,
                        type=str)
    parser.add_argument(
        '-output_csv_name',
        help=
        'Path to output csv (stored in the current working directory if path not provided)',
        default='',
        required=False,
        type=str)
    args = parser.parse_args()

    main(args)
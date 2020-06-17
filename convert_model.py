import os
import argparse
import tensorflow as tf
from tensorflow import lite
import tensorflow.keras as k


def parse_args():
    # Parse input arguments
    parser = argparse.ArgumentParser(
        description="Convert keras model to tflite")

    parser.add_argument('--model_filename',
                        help='name of the h5 file',
                        default="hair_segmentation_mobile.h5",
                        type=str)

    parser.add_argument("--model_folder",
                        help="path of folder containing model file",
                        default="weights",
                        type=str)

    parser.add_argument('--tflite_filename',
                        help='name of the tflite file',
                        default="hair_segmentation_mobile.tflite",
                        type=str)

    parser.add_argument("--tflite_folder",
                        help="path of folder where converted model is to be saved",
                        default="weights",
                        type=str)

    parser.add_argument("--fp_16",
                        help="If 1,quantize to float16 bit, 0 (default)",
                        default=0,
                        choices=[0, 1],
                        type=int)

    return parser.parse_args()


def convert_model(model_path, tflite_path, convert_to_fp16=0):

    model = k.models.load_model(model_path, compile=False)
    converter = lite.TFLiteConverter.from_keras_model(model)

    if convert_to_fp16 == 1:

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_path = f"{tflite_path.split('.')[0]}_fp16.tflite"

    print(f"Saving model at {tflite_path}")
    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)


if __name__ == "__main__":

    args = parse_args()

    model_path = os.path.join(args.model_folder, args.model_filename)
    tflite_path = os.path.join(args.tflite_folder, args.tflite_filename)
    convert_to_fp16 = int(args.fp_16)

    print("Converting model...")
    convert_model(model_path, tflite_path, convert_to_fp16)

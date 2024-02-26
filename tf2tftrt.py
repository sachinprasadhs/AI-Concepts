import logging
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
logging.disable(logging.WARNING)

SAVED_MODEL_DIR = "saved_model" # replace with tensorflow saved model

def get_dummy_images(batch_size=32, img_shape=[256, 256, 3]):
  img = tf.cast(tf.random.uniform(
      shape=[batch_size] + img_shape, dtype=tf.float32),
      dtype=tf.float32)
  print("Generated input random images with shape (N, H, W, C) =", img.shape)
  return img



# Instantiate the TF-TRT converter
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir=SAVED_MODEL_DIR,
   precision_mode=trt.TrtPrecisionMode.FP16
)

# Convert the model into TRT compatible segments
trt_func = converter.convert()
converter.summary()


dummy_images = get_dummy_images(1, [512, 1024, 3])
def input_fn():
   yield dummy_images

converter.build(input_fn=input_fn)
OUTPUT_SAVED_MODEL_DIR="./material_form_tftrt"
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)

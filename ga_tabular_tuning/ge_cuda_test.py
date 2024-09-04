import tensorflow as tf

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# List available devices
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):
    print("Running on GPU")
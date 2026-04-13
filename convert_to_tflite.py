import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("kidney_cnn_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save model
with open("kidney_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted to TFLite successfully!")
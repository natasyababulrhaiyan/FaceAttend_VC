from tensorflow.keras.models import load_model

model = load_model("model-cnn-facerecognition.keras")

print("MODEL BERHASIL DIMUAT")

print(model.summary())
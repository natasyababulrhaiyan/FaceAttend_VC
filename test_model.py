from tensorflow.keras.models import load_model

model = load_model("model-cnn-facerecognition.h5")

print("MODEL BERHASIL DIMUAT")

print(model.summary())
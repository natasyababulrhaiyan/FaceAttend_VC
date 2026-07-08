import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from tensorflow.keras.models import load_model

model = load_model("best_model.keras")

print("MODEL BERHASIL DIMUAT")

print(model.summary())
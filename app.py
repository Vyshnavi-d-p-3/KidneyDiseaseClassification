import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import logging
from monai.networks.nets import resnet18
import torch.nn.functional as F

app = Flask(__name__)
logger = logging.getLogger("ensemble_logger")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Load models (trained on 2 classes: Normal and Tumor)
vgg_model = load_model("artifacts/training/model.h5")
effnet_b2_model = torch.hub.load('pytorch/vision:v0.14.0', 'efficientnet_b2', weights=None)
effnet_b2_model.classifier[1] = torch.nn.Linear(in_features=1408, out_features=2)
effnet_b2_model.load_state_dict(torch.load("artifacts/training/efficientnetb2_final_2classes.pt", map_location=torch.device('cpu')))
effnet_b2_model.eval()
logger.info("✅ Loaded EfficientNetB2 model (2-class) successfully.")

monai_model = resnet18(spatial_dims=2, n_input_channels=3, num_classes=2)
monai_model.load_state_dict(torch.load("artifacts/training/monai_resnet18_kidney_2class.pt", map_location=torch.device("cpu")))
monai_model.eval()
logger.info("✅ Loaded MONAI ResNet18 model (2-class) successfully.")

class_labels = ['Normal', 'Tumor']

def preprocess_image(img_path, target_size):
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def preprocess_for_torch(img_path):
    img = Image.open(img_path).convert("RGB").resize((288, 288))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img).unsqueeze(0)
    return img_tensor

@app.route('/')
def home():
    return open("templates/index.html").read()

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("🔍 Starting prediction...")

    if 'file' not in request.files:
        logger.warning("⚠️ No file found in request.")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    logger.info(f"📥 Image saved to: {file_path}")

    try:
        # VGG
        vgg_img = preprocess_image(file_path, target_size=(224, 224))
        vgg_pred = vgg_model.predict(vgg_img)[0]

        # EfficientNetB2 (PyTorch)
        torch_img = preprocess_for_torch(file_path)
        with torch.no_grad():
            effnetb2_pred = effnet_b2_model(torch_img)
            effnetb2_probs = F.softmax(effnetb2_pred, dim=1).numpy()[0]

        # MONAI ResNet18 (PyTorch)
        with torch.no_grad():
            monai_pred = monai_model(torch_img)
            monai_probs = F.softmax(monai_pred, dim=1).numpy()[0]

        # Individual predictions
        vgg_pred_idx = np.argmax(vgg_pred)
        vgg_confidence = vgg_pred[vgg_pred_idx] * 100
        effnetb2_pred_idx = np.argmax(effnetb2_probs)
        effnetb2_confidence = effnetb2_probs[effnetb2_pred_idx] * 100
        monai_pred_idx = np.argmax(monai_probs)
        monai_confidence = monai_probs[monai_pred_idx] * 100

        print(f"VGG16 thinks: {class_labels[vgg_pred_idx]} ({vgg_confidence:.2f}%)")
        print(f"EffNetB2 thinks: {class_labels[effnetb2_pred_idx]} ({effnetb2_confidence:.2f}%)")
        print(f"MONAI thinks: {class_labels[monai_pred_idx]} ({monai_confidence:.2f}%)")

        # Ensemble: 15% VGG + 55% EffNetB2 + 30% MONAI
        final_probs = (
            0.15 * vgg_pred +
            0.45 * effnetb2_probs +
            0.40 * monai_probs
        )
        predicted_idx = np.argmax(final_probs)
        predicted_class = class_labels[predicted_idx]
        confidence = final_probs[predicted_idx] * 100

        # 🗳️ Majority Voting Ensemble (optional, but useful)
        votes = [np.argmax(vgg_pred), np.argmax(effnetb2_probs), np.argmax(monai_probs)]
        majority_class_idx = max(set(votes), key=votes.count)
        majority_class = class_labels[majority_class_idx]
        vote_confidence = votes.count(majority_class_idx) / 3 * 100

        print(f"\n🗳️ Majority Voting Result:")
        print(f"Voted Class: {majority_class} ({vote_confidence:.2f}% agreement)")


        # 🔍 DEBUG: Show all individual model probabilities
        print("\n🔍 Individual model probabilities:")
        print(f"  VGG:        Normal={vgg_pred[0]*100:.2f}%, Tumor={vgg_pred[1]*100:.2f}%")
        print(f"  EffNetB2:   Normal={effnetb2_probs[0]*100:.2f}%, Tumor={effnetb2_probs[1]*100:.2f}%")
        print(f"  MONAI:      Normal={monai_probs[0]*100:.2f}%, Tumor={monai_probs[1]*100:.2f}%")

        # 🔍 DEBUG: Show ensemble breakdown
        print("\n🔗 Weighted Ensemble Result:")
        print(f"  Normal: {final_probs[0]*100:.2f}%")
        print(f"  Tumor : {final_probs[1]*100:.2f}%")
        print(f"✅ Ensemble thinks: {predicted_class} ({confidence:.2f}%)\n")


        return jsonify({
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "raw_probability": [f"{p:.4f}" for p in final_probs],
            "metrics": {
                "reliability": f"{np.max(final_probs) * 100:.2f}%",
                "threshold_used": "N/A"
            },
            "voting_prediction": majority_class,
            "voting_agreement": f"{vote_confidence:.2f}%"
        })

    except Exception as e:
        logger.error("❌ ERROR during prediction:", exc_info=True)
        return jsonify({"error": "Error processing image"}), 500

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)


# import os
# import torch
# import numpy as np
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.preprocessing import image as keras_image
# from PIL import Image
# import logging

# app = Flask(__name__)
# logger = logging.getLogger("ensemble_logger")
# handler = logging.StreamHandler()
# formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
# handler.setFormatter(formatter)
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)

# # Load models (trained on 2 classes: Normal and Tumor)
# vgg_model = load_model("artifacts/training/model.h5")
# effnet_b0_model = load_model("artifacts/training_effnet/model.keras")
# effnet_b2_model = torch.hub.load('pytorch/vision:v0.14.0', 'efficientnet_b2', weights=None)
# effnet_b2_model.classifier[1] = torch.nn.Linear(in_features=1408, out_features=2)
# effnet_b2_model.load_state_dict(torch.load("efficientnetb2_final_2classes.pt", map_location=torch.device('cpu')))
# effnet_b2_model.eval()
# logger.info("✅ Loaded EfficientNetB2 model (2-class) successfully.")

# # Only 2 classes now
# class_labels = ['Normal', 'Tumor']

# def preprocess_image(img_path, target_size):
#     img = keras_image.load_img(img_path, target_size=target_size)
#     img_array = keras_image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return preprocess_input(img_array)

# def preprocess_for_torch(img_path):
#     img = Image.open(img_path).convert("RGB").resize((288, 288))
#     img = np.array(img).astype(np.float32) / 255.0
#     img = np.transpose(img, (2, 0, 1))
#     img_tensor = torch.tensor(img).unsqueeze(0)
#     return img_tensor

# @app.route('/')
# def home():
#     return open("templates/index.html").read()

# @app.route('/predict', methods=['POST'])
# def predict():
#     logger.info("🔍 Starting prediction...")

#     if 'file' not in request.files:
#         logger.warning("⚠️ No file found in request.")
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files['file']
#     file_path = os.path.join('uploads', file.filename)
#     file.save(file_path)
#     logger.info(f"📥 Image saved to: {file_path}")

#     try:
#         # Step 1: VGG
#         vgg_img = preprocess_image(file_path, target_size=(224, 224))
#         vgg_pred = vgg_model.predict(vgg_img)[0]

#         # Step 2: EffNetB0
#         effnet0_img = preprocess_image(file_path, target_size=(224, 224))
#         effnet0_pred = effnet_b0_model.predict(effnet0_img)[0]

#         # Step 3: EffNetB2 (PyTorch)
#         torch_img = preprocess_for_torch(file_path)
#         with torch.no_grad():
#             effnetb2_pred = effnet_b2_model(torch_img)
#             effnetb2_probs = torch.nn.functional.softmax(effnetb2_pred, dim=1).numpy()[0]

#         # Individual model predictions
#         vgg_pred_idx = np.argmax(vgg_pred)
#         vgg_pred_class = class_labels[vgg_pred_idx]
#         vgg_confidence = vgg_pred[vgg_pred_idx] * 100

#         effnet0_pred_idx = np.argmax(effnet0_pred)
#         effnet0_pred_class = class_labels[effnet0_pred_idx]
#         effnet0_confidence = effnet0_pred[effnet0_pred_idx] * 100

#         effnetb2_pred_idx = np.argmax(effnetb2_probs)
#         effnetb2_pred_class = class_labels[effnetb2_pred_idx]
#         effnetb2_confidence = effnetb2_probs[effnetb2_pred_idx] * 100

#         # Print individual predictions
#         print(f"VGG16 thinks: {vgg_pred_class} ({vgg_confidence:.2f}%)")
#         print(f"EffNetB0 thinks: {effnet0_pred_class} ({effnet0_confidence:.2f}%)")
#         print(f"EffNetB2 thinks: {effnetb2_pred_class} ({effnetb2_confidence:.2f}%)")

#         # Ensemble (60% weight to EffNetB2, 20% to others)
#         final_probs = (
#             0.2 * vgg_pred +
#             0.2 * effnet0_pred +
#             0.6 * effnetb2_probs
#         )
#         predicted_idx = np.argmax(final_probs)
#         predicted_class = class_labels[predicted_idx]
#         confidence = final_probs[predicted_idx] * 100

#         print(f"Ensemble thinks: {predicted_class} ({confidence:.2f}%)")

#         return jsonify({
#             "prediction": predicted_class,
#             "confidence": f"{confidence:.2f}%",
#             "raw_probability": [f"{p:.4f}" for p in final_probs],
#             "metrics": {
#                 "reliability": f"{np.max(final_probs) * 100:.2f}%",
#                 "threshold_used": "N/A"
#             }
#         })

#     except Exception as e:
#         logger.error("❌ ERROR during prediction:", exc_info=True)
#         return jsonify({"error": "Error processing image"}), 500

# if __name__ == "__main__":
#     logger.info("Starting Flask app...")
#     os.makedirs('uploads', exist_ok=True)
#     app.run(debug=True)

# import os
# import torch
# import numpy as np
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.preprocessing import image as keras_image
# from PIL import Image
# import logging

# app = Flask(__name__)
# logger = logging.getLogger("ensemble_logger")
# handler = logging.StreamHandler()
# formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
# handler.setFormatter(formatter)
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)

# # Load models
# vgg_model = load_model("artifacts/training/model.h5")
# effnet_b0_model = load_model("artifacts/training_effnet/model.keras")
# effnet_b2_model = torch.hub.load('pytorch/vision:v0.14.0', 'efficientnet_b2', weights=None)
# effnet_b2_model.classifier[1] = torch.nn.Linear(in_features=1408, out_features=4)
# effnet_b2_model.load_state_dict(torch.load("efficientnetb2_final_2classes.pt", map_location=torch.device('cpu')))
# effnet_b2_model.eval()
# logger.info("✅ Loaded EfficientNetB2 model successfully.")

# class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

# def preprocess_image(img_path, target_size):
#     img = keras_image.load_img(img_path, target_size=target_size)
#     img_array = keras_image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return preprocess_input(img_array)

# def preprocess_for_torch(img_path):
#     img = Image.open(img_path).convert("RGB").resize((288, 288))
#     img = np.array(img).astype(np.float32) / 255.0
#     img = np.transpose(img, (2, 0, 1))
#     img_tensor = torch.tensor(img).unsqueeze(0)
#     return img_tensor

# def map_to_4class(prob_2class):
#     return np.array([0, prob_2class[0], 0, prob_2class[1]])

# @app.route('/')
# def home():
#     return open("templates/index.html").read()

# @app.route('/predict', methods=['POST'])
# def predict():
#     logger.info("🔍 Starting prediction...")

#     if 'file' not in request.files:
#         logger.warning("⚠️ No file found in request.")
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files['file']
#     file_path = os.path.join('uploads', file.filename)
#     file.save(file_path)
#     logger.info(f"📥 Image saved to: {file_path}")

#     try:
#         # Step 1: VGG
#         vgg_img = preprocess_image(file_path, target_size=(224, 224))
#         vgg_pred = vgg_model.predict(vgg_img)[0]
#         vgg_4class = map_to_4class(vgg_pred)

#         # Step 2: EffNetB0
#         effnet0_img = preprocess_image(file_path, target_size=(224, 224))
#         effnet0_pred = effnet_b0_model.predict(effnet0_img)[0]
#         effnet0_4class = map_to_4class(effnet0_pred)

#         # Step 3: EffNetB2 (PyTorch)
#         torch_img = preprocess_for_torch(file_path)
#         with torch.no_grad():
#             effnetb2_pred = effnet_b2_model(torch_img)
#             effnetb2_probs = torch.nn.functional.softmax(effnetb2_pred, dim=1).numpy()[0]
#         # Individual model predictions
#         vgg_pred_idx = np.argmax(vgg_4class)
#         vgg_pred_class = class_labels[vgg_pred_idx]
#         vgg_confidence = vgg_4class[vgg_pred_idx] * 100

#         effnet0_pred_idx = np.argmax(effnet0_4class)
#         effnet0_pred_class = class_labels[effnet0_pred_idx]
#         effnet0_confidence = effnet0_4class[effnet0_pred_idx] * 100

#         effnetb2_pred_idx = np.argmax(effnetb2_probs)
#         effnetb2_pred_class = class_labels[effnetb2_pred_idx]
#         effnetb2_confidence = effnetb2_probs[effnetb2_pred_idx] * 100

#         # Print individual predictions
#         print(f"VGG16 thinks: {vgg_pred_class} ({vgg_confidence:.2f}%)")
#         print(f"EffNetB0 thinks: {effnet0_pred_class} ({effnet0_confidence:.2f}%)")
#         print(f"EffNetB2 thinks: {effnetb2_pred_class} ({effnetb2_confidence:.2f}%)")

#         # Ensemble
#         final_probs = (
#             0.2 * vgg_4class +
#             0.2 * effnet0_4class +
#             0.6 * effnetb2_probs
#         )
#         predicted_idx = np.argmax(final_probs)
#         predicted_class = class_labels[predicted_idx]
#         confidence = final_probs[predicted_idx] * 100

#         print(f"Ensemble thinks: {predicted_class} ({confidence:.2f}%)")


#         return jsonify({
#             "prediction": predicted_class,
#             "confidence": f"{confidence:.2f}%",
#             "raw_probability": [f"{p:.4f}" for p in final_probs],
#             "metrics": {
#                 "reliability": f"{np.max(final_probs) * 100:.2f}%",
#                 "threshold_used": "N/A"
#             }
#         })

#     except Exception as e:
#         logger.error("❌ ERROR during prediction:", exc_info=True)
#         return jsonify({"error": "Error processing image"}), 500

# if __name__ == "__main__":
#     logger.info("Starting Flask app...")
#     os.makedirs('uploads', exist_ok=True)
#     app.run(debug=True)

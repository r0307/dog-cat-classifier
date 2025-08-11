from flask import Flask, render_template, request, jsonify
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import io
from PIL import Image
import torch
import torchvision.models as models
import os

app=Flask(__name__)
class ResNetTransfer(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features=self.model.fc.in_features
    self.model.fc=nn.Linear(num_features,2)
  def forward(self,x):
    return self.model(x)
  
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
  ])

MODEL_PATH="dog_cat_classifier.pt"
if not os.path.exists(MODEL_PATH):
  raise FileNotFoundError(f"モデルファイルが見つかりません: {MODEL_PATH}")
model=ResNetTransfer()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
  if "file" not in request.files:
    return jsonify({"error":"ファイルがアップロードされていません。"}),400
  file=request.files["file"]
  if file.filename=="":
    return jsonify({"error":"ファイルが選択されていません。"}),400
  
  try:
      img_bytes=file.read()
      img=Image.open(io.BytesIO(img_bytes)).convert("RGB")
      with torch.no_grad():
        img_tensor=transform(img).unsqueeze(0)
        outputs=model(img_tensor)
        probabilities=F.softmax(outputs,1)
        confidence,predicted=torch.max(probabilities, 1)

      class_names=["猫", "犬"]
      predicted_class = class_names[predicted.item()]
  
      return jsonify({
                "prediction": predicted_class,
                "confidence": confidence.item()
            })
  except Exception as e:
    return jsonify({"error":str(e)}),500

if __name__=="__main__":
  app.run(debug=True)
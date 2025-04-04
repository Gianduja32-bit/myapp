from flask import Flask, request, render_template, send_from_directory
from diffusers import StableDiffusionPipeline
import torch
import os

app = Flask(__name__)

# Création du dossier d'images générées si inexistant
os.makedirs("static", exist_ok=True)

# Charger Stable Diffusion avec support MPS (Mac M1/M2)
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("mps")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    prompt = request.form["question"]
    image = pipe(prompt).images[0]

    # Sauvegarder l'image générée
    image_path = os.path.join("static", "image_generee.png")
    image.save(image_path)

    return render_template("index.html", image_url=image_path, prompt=prompt)

if __name__ == "__main__":
    app.run(debug=True)
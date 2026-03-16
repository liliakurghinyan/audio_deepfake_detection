import json
import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from detector import AudioDeepfakeDetector

app = FastAPI()
templates = Jinja2Templates(directory="web/templates")

detector = AudioDeepfakeDetector(model_path="model.pt", sr=16000, n_mels=80)

def load_validation_stats():
    stats = {}

    if os.path.exists("results/val_metrics.json"):
        with open("results/val_metrics.json") as f:
            val = json.load(f)
            stats["validation_auc"] = val.get("roc_auc")

    if os.path.exists("results/metrics.json"):
        with open("results/metrics.json") as f:
            test = json.load(f)
            stats["test_auc"] = test.get("test_auc")
            stats["accuracy"] = test.get("accuracy")
            stats["dataset_size"] = test.get("dataset_size")
            stats["model_name"] = test.get("model_name")

    return stats

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    stats = load_validation_stats()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": stats,
            "github_url": "https://github.com/YOUR_GITHUB_USERNAME",
        },
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    audio_bytes = await file.read()
    stats = load_validation_stats()

    try:
        result = detector.predict_from_upload(audio_bytes, file.filename)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "filename": file.filename,
                "label": result["label"],
                "prob_fake": round(result["prob_fake"], 4),
                "prob_real": round(result["prob_real"], 4),
                "converted": result["converted"],
                "original_ext": result["original_ext"],
                "stats": stats,
                "github_url": "https://github.com/liliakurghinyan/audio_deepfake_detection",
                "error_message": None,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "stats": stats,
                "github_url": "https://github.com/liliakurghinyan/audio_deepfake_detection",
                "error_message": str(e),
            },
        )
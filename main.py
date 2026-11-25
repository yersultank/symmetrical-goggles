from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Wine Classifier API")
model = joblib.load("model.joblib")

CLASS_NAMES = ["class_0", "class_1", "class_2"]

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.get("/")
def root():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(features: WineFeatures):
    x = np.array([[
        features.alcohol, features.malic_acid, features.ash,
        features.alcalinity_of_ash, features.magnesium, features.total_phenols,
        features.flavanoids, features.nonflavanoid_phenols,
        features.proanthocyanins, features.color_intensity, features.hue,
        features.od280_od315_of_diluted_wines, features.proline,
    ]])
    pred = int(model.predict(x)[0])
    return {"prediction": pred, "class_name": CLASS_NAMES[pred]}
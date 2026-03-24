import joblib
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

# -- App setup ----------------------------------------------------------------
app = FastAPI(
    title="Student Math Performance Predictor",
    description="Predicts a student's final math grade (G3) based on socio-demographic and academic features.",
    version="1.0.0"
)

# -- CORS Middleware -----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8080",
        "https://bianca255.github.io",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# -- Load model artifacts -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model  = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

with open(os.path.join(BASE_DIR, "feature_columns.json")) as f:
    FEATURE_COLUMNS = json.load(f)

# Exact 26 columns:
# ['sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','reason',
#  'guardian','traveltime','studytime','failures','schoolsup','paid',
#  'nursery','higher','internet','romantic','famrel','goout',
#  'Dalc','Walc','health','G1','G2']


# -- Input schema -------------------------------------------------------------
class StudentInput(BaseModel):
    # Demographic
    sex: int = Field(..., ge=0, le=1, description="Sex: 0=Female, 1=Male")
    age: int = Field(..., ge=15, le=22, description="Age (15-22)")
    address: int = Field(..., ge=0, le=1, description="Address: 0=Rural, 1=Urban")
    famsize: int = Field(..., ge=0, le=1, description="Family size: 0=LE3, 1=GT3")
    Pstatus: int = Field(..., ge=0, le=1, description="Parent cohabitation: 0=Apart, 1=Together")

    # Parent education (0=none to 4=higher)
    Medu: int = Field(..., ge=0, le=4, description="Mother education (0-4)")
    Fedu: int = Field(..., ge=0, le=4, description="Father education (0-4)")

    # Mother job (encoded 0-4)
    Mjob: int = Field(..., ge=0, le=4, description="Mother job (encoded 0-4)")

    # School info
    reason: int = Field(..., ge=0, le=3, description="Reason for school choice (0-3)")
    guardian: int = Field(..., ge=0, le=2, description="Guardian (0-2)")
    traveltime: int = Field(..., ge=1, le=4, description="Travel time to school (1-4)")
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time (1-4)")
    failures: int = Field(..., ge=0, le=4, description="Past class failures (0-4)")

    # Support flags (0=No, 1=Yes)
    schoolsup: int = Field(..., ge=0, le=1, description="Extra educational support (0=No, 1=Yes)")
    paid: int = Field(..., ge=0, le=1, description="Extra paid classes (0=No, 1=Yes)")
    nursery: int = Field(..., ge=0, le=1, description="Attended nursery school (0=No, 1=Yes)")
    higher: int = Field(..., ge=0, le=1, description="Wants higher education (0=No, 1=Yes)")
    internet: int = Field(..., ge=0, le=1, description="Internet access at home (0=No, 1=Yes)")
    romantic: int = Field(..., ge=0, le=1, description="In a romantic relationship (0=No, 1=Yes)")

    # Lifestyle (1-5 scale)
    famrel: int = Field(..., ge=1, le=5, description="Family relationship quality (1-5)")
    goout: int = Field(..., ge=1, le=5, description="Going out with friends (1-5)")
    Dalc: int = Field(..., ge=1, le=5, description="Workday alcohol consumption (1-5)")
    Walc: int = Field(..., ge=1, le=5, description="Weekend alcohol consumption (1-5)")
    health: int = Field(..., ge=1, le=5, description="Current health status (1-5)")

    # Academic
    G1: int = Field(..., ge=0, le=20, description="First period grade (0-20)")
    G2: int = Field(..., ge=0, le=20, description="Second period grade (0-20)")

    class Config:
        json_schema_extra = {
            "example": {
                "sex": 1, "age": 17, "address": 1, "famsize": 1,
                "Pstatus": 1, "Medu": 3, "Fedu": 2, "Mjob": 2,
                "reason": 0, "guardian": 0, "traveltime": 2,
                "studytime": 2, "failures": 0, "schoolsup": 0,
                "paid": 0, "nursery": 1, "higher": 1,
                "internet": 1, "romantic": 0, "famrel": 4,
                "goout": 2, "Dalc": 1, "Walc": 2, "health": 4,
                "G1": 10, "G2": 10
            }
        }


class RetrainInput(BaseModel):
    data: list = Field(
        ...,
        description="List of student records with all 26 features plus G3 as target"
    )


# -- Routes -------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Student Math Performance Predictor API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": type(model).__name__}


@app.post("/predict")
def predict(student: StudentInput):
    try:
        input_df = pd.DataFrame([student.dict()])
        input_df = input_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        scaled          = scaler.transform(input_df)
        prediction      = model.predict(scaled)[0]
        predicted_grade = round(float(np.clip(prediction, 0, 20)), 2)

        return {
            "predicted_G3": predicted_grade,
            "interpretation": (
                "At risk - needs support" if predicted_grade < 10
                else "Passing"            if predicted_grade < 14
                else "Good performance"
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def retrain(payload: RetrainInput):
    """
    Accepts new student records and retrains the model.
    Each record must include all 26 features plus G3 as the target.
    """
    global model, scaler

    try:
        new_df = pd.DataFrame(payload.data)

        if "G3" not in new_df.columns:
            raise HTTPException(
                status_code=400,
                detail="Each record must include 'G3' as the target column."
            )

        X_new = new_df.drop(columns=["G3"])
        y_new = new_df["G3"]

        X_new = X_new.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        scaler.fit(X_new)
        X_scaled = scaler.transform(X_new)
        model.fit(X_scaled, y_new)

        joblib.dump(model,  os.path.join(BASE_DIR, "best_model.pkl"))
        joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

        return {
            "message": "Model retrained successfully",
            "records_used": len(new_df),
            "model": type(model).__name__
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

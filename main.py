from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import cv2
import numpy as np
import base64
from pathlib import Path
import os
import pickle
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model


class Component1Type(BaseModel):
    base64Image: str


class Component2Type(BaseModel):
    menopause: int
    agegrp: int
    density: int
    race: int
    Hispanic: int
    bmi: int
    agefirst: int
    nrelbc: int
    brstproc: int
    surgmeno: int
    hrt: int


class Component3Type(BaseModel):
    ageAtDiagnosis: int
    typeOfBreastSurgery: int
    cancerType: int
    cancerTypeDetailed: int
    Cellularity: int
    Chemotherapy: int
    Pam50ClaudinLowSubtype: int
    cohort: float
    erStatusMeasuredByIHC: int
    erStatus: int
    neoplasmHistologicGrade: float
    her2StatusMeasuredBySNP6: int
    her2Status: int
    tumorOtherHistologicSubtype: int
    hormoneTherapy: int
    inferredMenopausalState: int
    integrativeCluster: int
    primaryTumorLaterality: int
    lymphNodesExaminedPositive: float
    mutationCount: float
    nottinghamPrognosticIndex: float
    oncotreeCode: int
    overallSurvivalMonths: int
    overallSurvivalStatus: int
    prStatus: int
    radioTherapy: int
    relapseFreeStatusMonths: int
    relapseFreeStatus: int
    numberOfSamplesPerPatient: int
    sex: int
    threeGeneClassifierSubtype: int
    tmbNonsynonymous: float
    tumorSize: float
    tumorStage: float


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def get_root():
    return {"message": "Hello World"}


model1_path = Path("assets/model1.h5")


@app.post("/component/1")
async def get_component1(file: UploadFile = File(...)):

    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())
    
    with open(file.filename, "rb") as image_file:

        img_bytes = image_file.read()
    
        if os.path.isfile(model1_path):
            new_model = load_model(model1_path)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            resize = tf.image.resize(img, (256, 256))
            yhat = new_model.predict(np.expand_dims(resize/255, 0))
            type = ''
            if yhat > 0.3:
                type = 'Malignant'
            else:
                type = 'Benign'
            return {"type": type}
        else:
            return ({"model_loaded": False, "error": "Model file not found"})


model2_path = Path("assets/model2.pkl")


@app.post("/component/2")
def get_component2(data: Component2Type):
    data_fields = {
        "menopause": data.menopause,
        "agegrp": data.agegrp,
        "density": data.density,
        "race": data.race,
        "Hispanic": data.Hispanic,
        "bmi": data.bmi,
        "agefirst": data.agefirst,
        "nrelbc": data.nrelbc,
        "brstproc": data.brstproc,
        "surgmeno": data.surgmeno,
        "hrt": data.hrt
    }
    if os.path.isfile(model2_path):
        with open(model2_path, 'rb') as f:
            loaded_model = pickle.load(f)
            dfToCheck = pd.DataFrame(data_fields, index=[0])
            return {"probabilityOfSurvice": loaded_model.predict_proba(dfToCheck)[:, 1][0]}
    else:
        return ({"model_loaded": False, "error": "Model file not found"})


model3_path = Path("assets/model3.pkl")


@app.post("/component/3")
def get_component3(data: Component3Type):
    dict_to_predict = {
        'Age at Diagnosis': data.ageAtDiagnosis,
        'Type of Breast Surgery': data.typeOfBreastSurgery,
        'Cancer Type': data.cancerType,
        'Cancer Type Detailed': data.cancerTypeDetailed,
        'Cellularity': data.Cellularity,
        'Chemotherapy': data.Chemotherapy,
        'Pam50 + Claudin-low subtype': data.Pam50ClaudinLowSubtype,
        'Cohort': data.cohort,
        'ER status measured by IHC': data.erStatusMeasuredByIHC,
        'ER Status': data.erStatus,
        'Neoplasm Histologic Grade': data.neoplasmHistologicGrade,
        'HER2 status measured by SNP6': data.her2StatusMeasuredBySNP6,
        'HER2 Status': data.her2Status,
        'Tumor Other Histologic Subtype': data.tumorOtherHistologicSubtype,
        'Hormone Therapy': data.hormoneTherapy,
        'Inferred Menopausal State': data.inferredMenopausalState,
        'Integrative Cluster': data.integrativeCluster,
        'Primary Tumor Laterality': data.primaryTumorLaterality,
        'Lymph nodes examined positive': data.lymphNodesExaminedPositive,
        'Mutation Count': data.mutationCount,
        'Nottingham prognostic index': data.nottinghamPrognosticIndex,
        'Oncotree Code': data.oncotreeCode,
        'Overall Survival (Months)': data.overallSurvivalMonths,
        'Overall Survival Status': data.overallSurvivalStatus,
        'PR Status': data.prStatus,
        'Radio Therapy': data.radioTherapy,
        'Relapse Free Status (Months)': data.relapseFreeStatusMonths,
        'Relapse Free Status': data.relapseFreeStatus,
        'Number of Samples Per Patient': data.numberOfSamplesPerPatient,
        'Sex': data.sex,
        '3-Gene classifier subtype': data.threeGeneClassifierSubtype,
        'TMB (nonsynonymous)': data.tmbNonsynonymous,
        'Tumor Size': data.tumorSize,
        'Tumor Stage': data.tumorStage
    }
    if os.path.isfile(model3_path):
        with open(model3_path, 'rb') as f:
            loaded_model = pickle.load(f)
            dfToCheck = pd.DataFrame(dict_to_predict, index=[0])
            return {"prob": 1-loaded_model.predict_proba(dfToCheck)[:, 1][0]}
    else:
        return ({"model_loaded": False, "error": "Model file not found"})

import os
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download


# =====================================================
# CONFIGURATION
# =====================================================

# CHANGE THIS
HF_REPO_ID = "alo234/AI-Color-Palette-Generator-model"

MODEL_FILE = "palettesense_model.pkl"


# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(
    title="PaletteSense AI",
    description="AI-powered design system generator",
    version="1.0.0"
)


# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =====================================================
# LOAD TRANSFORMER
# =====================================================

print("Loading Transformer...")

encoder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Transformer loaded.")


# =====================================================
# DOWNLOAD YOUR KAGGLE MODEL
# FROM HUGGING FACE
# =====================================================

print("Downloading PaletteSense model...")

model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=MODEL_FILE
)

print(
    "Model downloaded:",
    model_path
)


# =====================================================
# LOAD MODEL PACKAGE
# =====================================================

loaded_package = joblib.load(
    model_path
)

print(
    "Model package loaded."
)

print(
    loaded_package.keys()
)


# =====================================================
# GET MODELS
# =====================================================

industry_model = loaded_package[
    "industry_model"
]

style_model = loaded_package[
    "style_model"
]

mood_model = loaded_package[
    "mood_model"
]

color_systems = loaded_package[
    "color_systems"
]


# =====================================================
# REQUEST MODEL
# =====================================================

class DesignRequest(BaseModel):

    text: str


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "name": "PaletteSense AI",
        "version": "1.0.0"
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =====================================================
# PREDICT
# =====================================================

@app.post("/predict")
def predict(
    request: DesignRequest
):

    text = request.text.strip()


    if not text:

        return {
            "error": "Text cannot be empty."
        }


    # ---------------------------------------------
    # Transformer
    # ---------------------------------------------

    embedding = encoder.encode(
        [text]
    )


    # ---------------------------------------------
    # Industry
    # ---------------------------------------------

    industry = industry_model.predict(
        embedding
    )[0]


    # ---------------------------------------------
    # Style
    # ---------------------------------------------

    style = style_model.predict(
        embedding
    )[0]


    # ---------------------------------------------
    # Mood
    # ---------------------------------------------

    mood = mood_model.predict(
        embedding
    )[0]


    # ---------------------------------------------
    # Confidence
    # ---------------------------------------------

    industry_confidence = max(
        industry_model.predict_proba(
            embedding
        )[0]
    )

    style_confidence = max(
        style_model.predict_proba(
            embedding
        )[0]
    )

    mood_confidence = max(
        mood_model.predict_proba(
            embedding
        )[0]
    )


    # ---------------------------------------------
    # COLOR SYSTEM
    # ---------------------------------------------

    try:

        palette = color_systems[
            industry
        ][style]

    except (KeyError, TypeError):

        palette = {

            "primary": "#4F46E5",

            "secondary": "#06B6D4",

            "background": "#F8FAFC",

            "surface": "#FFFFFF",

            "accent": "#8B5CF6"

        }


    # ---------------------------------------------
    # RESPONSE
    # ---------------------------------------------

    return {

        "input": text,

        "analysis": {

            "industry": str(
                industry
            ),

            "style": str(
                style
            ),

            "mood": str(
                mood
            )

        },

        "confidence": {

            "industry": round(
                float(
                    industry_confidence
                ),
                4
            ),

            "style": round(
                float(
                    style_confidence
                ),
                4
            ),

            "mood": round(
                float(
                    mood_confidence
                ),
                4
            )

        },

        "palette": palette

    }


# =====================================================
# LOCAL RUN
# =====================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )

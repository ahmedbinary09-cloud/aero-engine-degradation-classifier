import pickle
from typing import Dict, List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from health import STATES, rul_to_state, state_probabilities


# request: one engine reading (one cycle) = 3 operating settings + 21 sensors
class Reading(BaseModel):
    unit: int = Field(..., ge=1, description="engine id")
    cycle: int = Field(..., ge=1, description="flight cycle number")
    op1: float
    op2: float
    op3: float = 100.0
    s1: float = 518.67
    s2: float
    s3: float
    s4: float
    s5: float = 14.62
    s6: float = 21.61
    s7: float
    s8: float
    s9: float
    s10: float = 1.30
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float = 0.03
    s17: float
    s18: float = 2388.0
    s19: float = 100.0
    s20: float
    s21: float


# response
class PredictResponse(BaseModel):
    unit: int
    cycle: int
    rul_cycles: float
    state: str
    state_probabilities: Dict[str, float]


app = FastAPI(title="engine-rul-health")

with open("model.bin", "rb") as f_in:
    bundle = pickle.load(f_in)

model = bundle["model"]
features: List[str] = bundle["features"]
sigma: float = bundle["sigma"]
rul_cap: int = bundle["rul_cap"]


def predict_single(reading: dict) -> dict:
    X = pd.DataFrame([reading])[features]
    rul = float(min(max(model.predict(X)[0], 0.0), rul_cap))
    proba = state_probabilities(rul, sigma)[0]
    return {
        "rul_cycles": round(rul, 1),
        "state": rul_to_state(rul)[0],
        "state_probabilities": {s: round(float(p), 4) for s, p in zip(STATES, proba)},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(reading: Reading) -> PredictResponse:
    data = reading.model_dump()
    return PredictResponse(unit=data["unit"], cycle=data["cycle"], **predict_single(data))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

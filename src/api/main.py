from fastapi import FastAPI
from src.api.pydantic_models import CustomerRequest, RiskResponse
import mlflow.pyfunc

app = FastAPI()

# Load model from MLflow Model Registry
MODEL_NAME = "CreditRiskBestModel"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/Production")

@app.post("/predict", response_model=RiskResponse)
def predict_risk(data: CustomerRequest):
    # Convert input to DataFrame
    input_df = data.to_df()
    # Predict probability
    prob = model.predict(input_df)[0]
    return RiskResponse(risk_probability=float(prob))
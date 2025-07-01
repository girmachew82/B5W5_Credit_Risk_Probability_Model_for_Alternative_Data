from pydantic import BaseModel
import pandas as pd

class CustomerRequest(BaseModel):
    # Add all model feature fields here, e.g.:
    Amount: float
    Value: float
    PricingStrategy: float
    day: int
    hour: int
    # ...add all other required fields...

    def to_df(self):
        return pd.DataFrame([self.dict()])

class RiskResponse(BaseModel):
    risk_probability: float
# service.py
import bentoml
import pandas as pd

model = bentoml.models.get("health_insurance_anomaly_detector:sebrf2cabow75qt2")

@bentoml.service
class HealthInsuranceService:
    @bentoml.api(input_spec=dict, output_spec=dict)
    def predict(self, payload: dict) -> dict:
        # Handle JSON data from Flask
        if "data" in payload:
            input_df = pd.DataFrame(payload["data"])
        else:
            input_df = pd.DataFrame(payload)
        
        predictions = model.predict(input_df)
        return {"predictions": predictions.tolist()}


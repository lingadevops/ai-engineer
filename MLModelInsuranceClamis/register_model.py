import bentoml
import pickle

#load the trained model
model_path = '/Users/linga/Documents/Linga_Dir/ai-engineer/MLModelInsuranceClamis/model.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)


#save the model to BentoML

bento_model = bentoml.sklearn.save_model("health_insurance_anomaly_detector", model)

print(f"Model registered with BentoML: {bento_model}")
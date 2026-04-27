import pandas as pd
import numpy as np

# set random seed for reproducibility
np.random.seed(42)

# generate synthetic health insurance claims data
num_records = 1000
data = {
    'claim_id': np.arange(1, num_records + 1),
    'claim_amount': np.random.normal(1000, 250, num_records),
    'num_services': np.random.randint(1, 10, num_records),
    'age': np.random.randint(18, 80, num_records),
    'provider_id': np.random.randint(1, 50, num_records),
    'days_since_last_claim': np.random.randint(0, 365, num_records),
}

# create a DataFrame
df = pd.DataFrame(data)

# introduce some anomalies

num_anomalies = 50
anomolies = {
    'claim_id': np.arange(num_records + 1, num_records + num_anomalies + 1),
    'claim_amount': np.random.normal(5000, 1000, num_anomalies),
    'num_services': np.random.randint(10, 20, num_anomalies),
    'age': np.random.randint(18, 80, num_anomalies),
    'provider_id': np.random.randint(1, 50, num_anomalies),
    'days_since_last_claim': np.random.randint(0, 365, num_anomalies),
}

df_anomalies = pd.DataFrame(anomolies)

# combine normal data with anomalies
df_combined = pd.concat([df, df_anomalies], ignore_index=True)
# save to CSV
df_combined.to_csv('synthetic_health_claims.csv', index=False)
print("Synthetic health insurance claims data generated and saved to 'synthetic_health_claims.csv'")
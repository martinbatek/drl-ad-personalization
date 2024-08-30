# %%
# Import packages
import pandas as pd
import numpy as np

# %%
log_dir = 'logs/'
datasets = ['kdd12','avazu','criteo']
for ds_name in datasets:
    filename = log_dir+ds_name+'_model_scores.csv'
    logs = pd.read_csv(filename)
    logs.columns = ['Model', 'Epochs', 'Log Loss', 'Accuracy', 'Precision','Recall','AUC']
    logs.Model = logs.Model.str.upper()
    for col in ['Log Loss', 'Accuracy', 'Precision','Recall','AUC']:
        logs[col] = round(logs[col],4)
    out_name = log_dir+ds_name+'_model_scores_cleaned.csv'
    logs.to_csv(out_name,index=False)



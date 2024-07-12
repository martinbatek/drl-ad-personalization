# Export models to pickle files
with open('./models/lr/kdd12_logistic_regression.pkl', 'wb') as f:
    pkl.dump(kdd12_pipeline, f, protocol=pkl.HIGHEST_PROTOCOL)

with open('./models/lr/avazu_logistic_regression.pkl', 'wb') as f:
    pkl.dump(avazu_pipeline, f, protocol=pkl.HIGHEST_PROTOCOL)

with open('./models/lr/criteo_logistic_regression.pkl', 'wb') as f:
    pkl.dump(criteo_pipeline, f, protocol=pkl.HIGHEST_PROTOCOL)
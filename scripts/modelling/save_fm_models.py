# Export models to pickle files
with open('./models/fm/kdd12_factorization_machine.pkl', 'wb') as f:
    pkl.dump(kdd12_fm, f, protocol=pkl.HIGHEST_PROTOCOL)

with open('./models/fm/avazu_factorization_machine.pkl', 'wb') as f:
    pkl.dump(avazu_fm, f, protocol=pkl.HIGHEST_PROTOCOL)

with open('./models/fm/criteo_factorization_machine.pkl', 'wb') as f:
    pkl.dump(criteo_fm, f, protocol=pkl.HIGHEST_PROTOCOL)
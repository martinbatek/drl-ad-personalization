from fastFM import sgd

# Get the categorical features from each dataset
kdd12_train_X_cat = kdd12_train_X[kdd12_categorical_columns]
kdd12_val_X_cat = kdd12_val_X[kdd12_categorical_columns]

avazu_train_X_cat = avazu_train_X[avazu_categorical_columns]
avazu_val_X_cat = avazu_val_X[avazu_categorical_columns]

criteo_train_X_cat = criteo_train_X[criteo_categorical_columns]
criteo_val_X_cat = criteo_val_X[criteo_categorical_columns]

# Get sparse csc matrices
kdd12_train_X_csc = sparse.csc_matrix(pd.get_dummies(kdd12_train_X, columns=kdd12_categorical_columns).to_numpy('float'))
kdd12_val_X_csc = sparse.csc_matrix(pd.get_dummies(kdd12_val_X, columns=kdd12_categorical_columns).to_numpy('float'))

avazu_train_X_csc = sparse.csc_matrix(pd.get_dummies(avazu_train_X, columns=avazu_categorical_columns).to_numpy('float'))
avazu_val_X_csc = sparse.csc_matrix(pd.get_dummies(avazu_val_X, columns=avazu_categorical_columns).to_numpy('float'))

criteo_train_X_csc = sparse.csc_matrix(pd.get_dummies(criteo_train_X, columns = criteo_categorical_columns).to_numpy('float'))
criteo_val_X_csc = sparse.csc_matrix(pd.get_dummies(criteo_val_X, columns = criteo_categorical_columns).to_numpy('float'))

# Change the target incoding
kdd12_train_y_fm = np.where(np.round(kdd12_train_y.to_numpy()) == 1., 1., -1.)
kdd12_val_y_fm = np.where(np.round(kdd12_val_y.to_numpy()) == 1., 1., -1.)

avazu_train_y_fm = np.where(avazu_train_y.to_numpy() == 1., 1., -1.)
avazu_val_y_fm = np.where(avazu_val_y.to_numpy() == 1., 1., -1.)

criteo_train_y_fm = np.where(criteo_train_y.to_numpy() == 1., 1., -1.)
criteo_val_y_fm = np.where(criteo_val_y.to_numpy() == 1., 1., -1.)

# fit the model as in the tutorial
kdd12_fm = sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0,
                          l2_reg_V=0, rank=2, step_size=0.1)
kdd12_fm.fit(kdd12_train_X_csc, kdd12_train_y_fm)

avazu_fm = sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0,
                          l2_reg_V=0, rank=2, step_size=0.1)
avazu_fm.fit(avazu_train_X_csc, avazu_train_y_fm)

criteo_fm = sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0,
                          l2_reg_V=0, rank=2, step_size=0.1)
criteo_fm.fit(criteo_train_X_csc, criteo_train_y_fm)
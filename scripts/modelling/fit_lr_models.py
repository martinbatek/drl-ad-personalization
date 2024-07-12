# Define One hot encoding preprocessing pipeline

## Define the one-hot encoding estimators
kdd12_oh_encoder = OneHotEncoder()
avazu_oh_encoder = OneHotEncoder()
criteo_oh_encoder = OneHotEncoder()

## Define the column transformer
kdd12_preprocessor = ColumnTransformer(
    transformers=[
        ('oh', kdd12_oh_encoder, kdd12_categorical_columns)
    ],
    remainder='passthrough'
)

avazu_preprocessor = ColumnTransformer(
    transformers=[
        ('oh', avazu_oh_encoder, avazu_categorical_columns)
    ],
    remainder='passthrough'
)

criteo_preprocessor = ColumnTransformer(
    transformers=[
        ('oh', criteo_oh_encoder, criteo_categorical_columns)
    ],
    remainder='passthrough'
)
# Define the logistic regression model pipelines
kdd12_pipeline = Pipeline([
    ('preprocessor', kdd12_preprocessor),
    ('classifier', LogisticRegression())
])

avazu_pipeline = Pipeline([
    ('preprocessor', avazu_preprocessor),
    ('classifier', LogisticRegression())
])

criteo_pipeline = Pipeline([
    ('preprocessor', criteo_preprocessor),
    ('classifier', LogisticRegression())
])
# Fit the models
kdd12_pipeline.fit(kdd12_train.drop(columns=['click']), np.round(kdd12_train['click'])) # Have to round the CTRs to 0 or 1
avazu_pipeline.fit(avazu_train.drop(columns=['click']), avazu_train['click'])
criteo_pipeline.fit(criteo_train.drop(columns=['click']), criteo_train['click'])
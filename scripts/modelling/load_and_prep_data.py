# Read in the encoded and standardized datasets
kdd12_standardized = pd.read_csv('./data/kdd12/kdd12_standardized.csv')
avazu_standardized = pd.read_csv('./data/avazu/avazu_standardized.csv').drop(columns=['id'])
criteo_standardized = pd.read_csv('./data/criteo/criteo_standardized.csv')

## Create lists of categorical colums for each dataset
kdd12_categorical_columns = [
    'DisplayURL',
    'AdID',
    'AdvertiserID',
    'QueryID',
    'KeywordID',
    'TitleID',
    'DescriptionID',
    'UserID'
]

avazu_categorical_columns = [
    'c1',
    'banner_pos',
    'site_id',
    'site_domain',
    'site_category',
    'app_id',
    'app_domain',
    'app_category',
    'device_id',
    'device_ip',
    'device_model',
    'device_type',
    'device_conn_type',
    'c14',
    'c15',
    'c16',
    'c17',
    'c18',
    'c19',
    'c20',
    'c21'
]

criteo_numerical_columns = [f'int_{i}' for i in np.arange(1,14)]
criteo_categorical_columns = [col for col in criteo_standardized.columns.tolist() if col not in criteo_numerical_columns+['click']]

## Cast categorical columns to string
kdd12_standardized[kdd12_categorical_columns] = kdd12_standardized[kdd12_categorical_columns].astype(str)
avazu_standardized[avazu_categorical_columns] = avazu_standardized[avazu_categorical_columns].astype(str)
criteo_standardized[criteo_categorical_columns] = criteo_standardized[criteo_categorical_columns].astype(str)

## Label Encode the categorical columns
for feat in kdd12_categorical_columns:
    kdd12_standardized[feat] = kdd12_standardized[feat].astype('category').cat.codes

for feat in avazu_categorical_columns:
    avazu_standardized[feat] = avazu_standardized[feat].astype('category').cat.codes

for feat in criteo_categorical_columns:
    criteo_standardized[feat] = criteo_standardized[feat].astype('category').cat.codes

# Split the datasets into training and validation sets
kdd12_train, kdd12_val = train_test_split(kdd12_standardized, test_size=0.2, random_state=42)
avazu_train, avazu_val = train_test_split(avazu_standardized, test_size=0.2, random_state=42)
criteo_train, criteo_val = train_test_split(criteo_standardized, test_size=0.2, random_state=42)

# Separate the X and y
kdd12_train_X = kdd12_train.drop(columns='click')
kdd12_train_y = kdd12_train['click']
kdd12_val_X = kdd12_val.drop(columns='click') 
kdd12_val_y = kdd12_val['click']

avazu_train_X = avazu_train.drop(columns='click')
avazu_train_y = avazu_train['click']
avazu_val_X = avazu_val.drop(columns='click')
avazu_val_y = avazu_val['click']

criteo_train_X = criteo_train.drop(columns='click')
criteo_train_y = criteo_train['click']
criteo_val_X = criteo_val.drop(columns='click')
criteo_val_y = criteo_val['click']
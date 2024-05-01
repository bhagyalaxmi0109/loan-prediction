# common constants
import certifi

ca = certifi.where()

TARGET_COLUMN = 'Loan_Status'
TRAIN_PIPELINE_NAME = 'train'
ARTIFACT_DIR = 'artifact'
FILE_NAME = 'loan-train-data.csv'

TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'


# MONGODB_URL_KEY = 'mongodb+srv://veenagabnave:aa5xcuda8yrgb6Zd@cluster0.otxvmcd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tlsCAFile=' + ca

MONGODB_KEY = "MONGODB_KEY"
DATABASE_NAME = 'loan-prediction'

# Data Ingestion Constants
TRAIN_DI_COLLECTION_NAME = 'loan-train-data'
DI_DIR_NAME = 'data_ingestion'
DI_FEATURE_STORE_DIR = 'feature_store'
DI_INGESTED_DIR = 'ingested'
DI_TRAIN_TEST_SPLIT_RATIO = 0.2
DI_COL_DROP_IN_CLEAN = ['_id','Loan_ID']

DI_MANDATORY_COL_LIST = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History',
                         'Property_Area', 'Loan_Status', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                         'Loan_Amount_Term']

DI_MANDATORY_COL_DATA_TYPE = {'Gender': 'object', 'Married': 'object', 'Dependents': 'object', 'Education': 'object',
                              'Self_Employed': 'object', 'Credit_History': 'object', 'Property_Area': 'object',
                              'Loan_Status': 'object', 'ApplicantIncome': 'float64', 'CoapplicantIncome': 'float64',
                              'LoanAmount': 'float64', 'Loan_Amount_Term': 'float64'}

# Data Validation Constants
DV_IMPUTATION_VALUES_FILE_NAME = "source/ml/imputation_values.csv"
DV_OUTLIERS_PARAMS_FILE = "source/ml/outliers_details.csv"
DV_DIR_NAME = 'data_validation'


# Data transformation constant
DT_MULTI_CLASS_COL = ['Dependents', 'Property_Area']
DT_BINARY_CLASS_COL = ['Gender', 'Married', 'Education', 'Self_Employed']
DT_ENCODER_PATH = 'source/ml/multi_class_encoder.pkl'
DT_DIR_NAME: str = "data_transformation"
MP_DIR_NAME = "model_prediction"

# model train and evaluate

MODEL_PATH = "source/ml/artifact"
FINAL_MODEL_PATH = "source/ml/final_model"

# Prediction constant
PREDICT_PIPELINE_NAME = 'predict'
PREDICT_DATA_FILE_NAME = 'predict_data.csv'
PREDICT_FILE = 'predict.csv'
PREDICT_DI_COLLECTION_NAME = "loan-predict-data"

FINAL_MODEL_FILE_NAME = "GradientBoostingClassifier.pkl"
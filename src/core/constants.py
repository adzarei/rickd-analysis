"""Constants module for the RICKD analysis project."""

RICKD_ROOT_DATA_FOLDER = "/Users/adrianzapaterreig/Documents/Personal/TFM/data/Running Injury Clinic Kinematic Dataset" 
RICKD_SOURCE_DATA_FOLDER = f"{RICKD_ROOT_DATA_FOLDER}/source_data"
RICKD_PROCESSED_DATA_FOLDER = f"{RICKD_ROOT_DATA_FOLDER}/processed_data"
RICKD_RESULTS_FOLDER = f"{RICKD_ROOT_DATA_FOLDER}/results"

RICKD_RUNNING_METADATA_FILE = f"{RICKD_ROOT_DATA_FOLDER}/run_data_meta.csv"
RICKD_RUNNING_METADATA_CLEANED_FILE = f"{RICKD_PROCESSED_DATA_FOLDER}/run_data_meta_cleaned.csv"
RICKD_SESSION_DATA_FULL_FILE = f"{RICKD_PROCESSED_DATA_FOLDER}/session_data_full.csv"

RICKD_PROY_RESOURCE_FOLDER = "/Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis/resources"

# Mapping files
RICKD_MAP_INJURY_CODE = f"{RICKD_PROY_RESOURCE_FOLDER}/injury_code_mapping.csv"
RICKD_MAP_INJURY_DESC = f"{RICKD_PROY_RESOURCE_FOLDER}/injury_desc_mapping.csv"
RICKD_MAP_SELF_INJURY_CODE = f"{RICKD_PROY_RESOURCE_FOLDER}/self_select_injury_code_mapping.csv"

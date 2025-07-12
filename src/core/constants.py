"""Constants module for the RICKD analysis project."""

RICKD_ROOT_DATA_FOLDER = "/Users/adrianzapaterreig/Documents/Personal/TFM/data/Running Injury Clinic Kinematic Dataset"
RICKD_SOURCE_DATA_FOLDER = f"{RICKD_ROOT_DATA_FOLDER}/source_data"
RICKD_PROCESSED_DATA_FOLDER = f"{RICKD_ROOT_DATA_FOLDER}/processed_data"
RICKD_RESULTS_FOLDER = f"{RICKD_ROOT_DATA_FOLDER}/results"
RICKD_PROY_RESOURCE_FOLDER = "/Users/adrianzapaterreig/Documents/Personal/TFM/rickd-analysis/resources"

# Processed data files
RICKD_RUNNING_METADATA_FILE = f"{RICKD_ROOT_DATA_FOLDER}/run_data_meta.csv"
RICKD_RUNNING_METADATA_CLEANED_FILE = f"{RICKD_PROCESSED_DATA_FOLDER}/run_data_meta_cleaned.csv"
RICKD_DESCRIPTIVE_VARIABLES_FILE = f"{RICKD_PROCESSED_DATA_FOLDER}/descriptive_variables.csv"
RICKD_MARKER_CENTERS_FILE = f"{RICKD_PROCESSED_DATA_FOLDER}/marker_centers.csv"
RICKD_MARKER_DATA_120HZ_FILE = f"{RICKD_PROCESSED_DATA_FOLDER}/marker_data_120hz.csv"
RICKD_MARKER_DATA_200HZ_FILE = f"{RICKD_PROCESSED_DATA_FOLDER}/marker_data_200hz.csv"
RICKD_SESSION_DATA_FULL_FILE = f"{RICKD_PROCESSED_DATA_FOLDER}/session_data_full.csv"
RICKD_SESSION_DATA_FULL_CLEANED_FILE = f"{RICKD_PROCESSED_DATA_FOLDER}/cleaned_session_data_full.csv"

# Mapping files
RICKD_MAP_INJURY_CODE = f"{RICKD_PROY_RESOURCE_FOLDER}/injury_code_mapping.csv"
RICKD_MAP_INJURY_DESC = f"{RICKD_PROY_RESOURCE_FOLDER}/injury_desc_mapping.csv"
RICKD_MAP_SELF_INJURY_CODE = f"{RICKD_PROY_RESOURCE_FOLDER}/self_select_injury_code_mapping.csv"
RICKD_MAP_INJURED_JOINT = f"{RICKD_PROY_RESOURCE_FOLDER}/injured_joint_mapping.csv"
RICKD_MAP_INJURED_JOINT_SIDE = f"{RICKD_PROY_RESOURCE_FOLDER}/injured_joint_side_mapping.csv"
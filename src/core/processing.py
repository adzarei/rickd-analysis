import os
import json
from collections import defaultdict
from typing import Dict, Tuple
from pydantic import BaseModel


class MarkerPoint(BaseModel):
    x: float
    y: float
    z: float

    @classmethod
    def from_coordinates_list(cls, lst: list) -> "MarkerPoint":
        if len(lst) != 3:
            raise ValueError(f"Expected 3D coordinates, got {len(lst)}")
        return cls(x=lst[0], y=lst[1], z=lst[2])


class SessionData(BaseModel):
    session_id: str
    subject_id: int
    session_hz: int
    desc_variables: dict
    marker_center_data: dict
    marker_data: dict


def get_dv_var_name(side, orig_name) -> str:
    return side + "_" + str.lower(orig_name)


def get_marker_center_var_name(group, orig_name) -> str:
    return group + "_" + str.lower(orig_name)


def get_marker_var_name(orig_name) -> str:
    return str.lower(orig_name)


def process_row(row: Tuple[int, Dict], source_data_folder: str) -> SessionData:
    idx, session = row
    row_id = session['id']
    subject_id = session['sub_id']
    relative_session_path = session['session_file_path']

    json_session = json.load(open(os.path.join(source_data_folder, relative_session_path)))
    frequency_hz = json_session['hz_r']

    # Extract Descriptive Variable data
    dv_data = {}
    dv_data['id'] = row_id
    dv_data['subject_id'] = subject_id
    dv_data['session_file_path'] = relative_session_path

    dv_r = json_session['dv_r']
    variables = {
        'l': dv_r['left'],
        'r': dv_r['right']
    }
    
    for side, data in variables.items():
        for name, value in data.items():
            variable_name = get_dv_var_name(side, name)
            dv_data[variable_name] = value
    
    # Extract Marker center data
    marker_center_data = defaultdict(list)
    marker_center_data['id'] = row_id
    marker_center_data['subject_id'] = subject_id
    marker_center_data['session_file_path'] = relative_session_path

    marker_vars = {
        "indiv": json_session['joints'],
        "cluster": json_session['neutral']
    }
    
    for group, data in marker_vars.items():
        for keypoint, coordinates in data.items():
            marker = MarkerPoint.from_coordinates_list(coordinates)
            variable_prefix = get_marker_center_var_name(group, keypoint)
            marker_center_data[variable_prefix + "_x"] = marker.x
            marker_center_data[variable_prefix + "_y"] = marker.y
            marker_center_data[variable_prefix + "_z"] = marker.z
    
    # Extract Marker data
    marker_data = defaultdict(list)
    marker_vars = json_session['running']

    is_first_keypoint = True
    for keypoint, data in marker_vars.items():
        for coordinates in data:
            marker = MarkerPoint.from_coordinates_list(coordinates)
            variable_prefix = get_marker_var_name(keypoint)
            if is_first_keypoint:
                marker_data['id'].append(row_id)
                marker_data['subject_id'].append(subject_id)
                marker_data['session_file_path'].append(relative_session_path)
            marker_data[variable_prefix + "_x"].append(marker.x)
            marker_data[variable_prefix + "_y"].append(marker.y)
            marker_data[variable_prefix + "_z"].append(marker.z)
        is_first_keypoint = False

    return SessionData(
        session_id=row_id, 
        subject_id=subject_id, 
        session_hz=frequency_hz, 
        desc_variables=dv_data, 
        marker_center_data=marker_center_data, 
        marker_data=marker_data
    ) 
from django.shortcuts import render, redirect
from django.http import JsonResponse,FileResponse,HttpResponse
from bson import ObjectId
import pymongo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt  
from pymongo import MongoClient
import json
import pandas as pd
from django.core.files.storage import default_storage
from django.conf import settings
from scipy import signal
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import math
import plotly.graph_objects as go
import plotly.io as pio
import traceback
import io
from .PQRST_detection_model import check_r_index, check_qs_index, check_pt_index, r_index_model, pt_index_model
import random
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator

client = pymongo.MongoClient("mongodb://192.168.1.65:27017/")
db = client["ecgarrhythmias"]

arrhythmias_dict = {
'Myocardial Infarction': ['T-wave abnormalities', 'Inferior MI', 'Lateral MI'],
'Atrial Fibrillation & Atrial Flutter': ['Afib', 'Aflutter'],
'HeartBlock': ['I Degree', 'II Degree', 'III Degree'],
'Junctional Rhythm': ['Junctional Bradycardia', 'Junctional Rhythm'],
'Premature Atrial Contraction': ['Isolated', 'Bigeminy', 'Couplet','Triplet', 'SVT','Trigeminy','Quadrigeminy'],
'Premature Ventricular Contraction': ['AIVR', 'Bigeminy', 'Couplet', 'Triplet', 'Isolated', 'NSVT', 'Quadrigeminy', 'Trigeminy','IVR','VT'],
'Ventricular Fibrillation and Asystole': ['VFIB','VFL','Asystole'],
'Noise':['Noise'],
'Others':['Others'],
'LBBB & RBBB':['Left Bundle Branch Block','Right Bundle Branch Block']
}

# Dashboard View
def index(request):
    return render(request, 'oom_ecg_data/ecg_data_index.html', {'arrhythmias_dict':arrhythmias_dict})#,data=data

def select_arrhythmia(request):
    selected_arrhythmia = request.GET.get('arrhythmia', '')  # Get the selected key
    return render(request, "your_template.html", {
        "arrhythmias_dict": arrhythmias_dict,
        "selected_arrhythmia": selected_arrhythmia,
    })

@csrf_exempt
def new_insert_data(request):
    if request.method == "POST":
        try:
            if "csv_file" not in request.FILES:
                return JsonResponse({"status": "error", "message": "File is not uploaded..."})

            patient_id = request.POST.get("patientId")
            arrhythmia_mi = request.POST.getlist("arrhythmia[]")  # Get as list
            sub_arrhythmia = request.POST.getlist("subArrhythmia[]")
            frequency = request.POST.get("newfrequency")
            lead_type = request.POST.get("lead")
            lead = lead_type.split(' ')[0]
            if len(arrhythmia_mi) != len(sub_arrhythmia):
                return JsonResponse({
                    "status": "error",
                    "message": "Mismatch between arrhythmia and sub-arrhythmia selections."
                })

            # Clean up strings
            collections_to_insert = [
                (arrhythmia_mi[i].strip(), sub_arrhythmia[i].strip())
                for i in range(len(arrhythmia_mi))
            ]
            file = request.FILES["csv_file"]
            file_path = os.path.join(settings.MEDIA_ROOT, "temp", file.name)
            default_storage.save(file_path, file)

            all_lead_data = pd.read_csv(file_path)
            column_count = all_lead_data.shape[1]

            # Validate lead-column match
            if lead_type == "2" and column_count > 2:
                return JsonResponse({"status": "error", "message": "Incorrect lead selected."})
            elif lead_type == "7" and column_count not in [7, 8]:
                return JsonResponse({"status": "error", "message": "Incorrect lead selected."})
            elif lead_type == "12" and column_count not in [12, 13]:
                return JsonResponse({"status": "error", "message": "Incorrect lead selected."})

            all_lead_data.columns = all_lead_data.columns.str.upper()

            # Drop time column if exists
            if lead_type == "12" and all_lead_data.shape[1] == 13:
                all_lead_data = all_lead_data.iloc[:, 1:13]
            elif lead_type == "7" and all_lead_data.shape[1] == 8:
                all_lead_data = all_lead_data.iloc[:, 1:8]
            elif lead_type == "2" and all_lead_data.shape[1] > 1:
                all_lead_data = all_lead_data.iloc[:, 1:2]

            # Handle header rows with alphabet values
            if any(str(_).isalpha() for _ in all_lead_data.iloc[0, :].values):
                if lead_type == "2":
                    all_lead_data = pd.read_csv(file_path, skiprows=1, usecols=[0]).fillna(0)
                    all_lead_data.columns = ['II']
                elif lead_type == "7":
                    col_indices = [0, 1, 2, 3, 4, 5, 6]
                    all_lead_data = pd.read_csv(file_path, skiprows=1, usecols=col_indices).fillna(0)
                    all_lead_data.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V5']
                elif lead_type == "12":
                    col_indices = list(range(12))
                    all_lead_data = pd.read_csv(file_path, skiprows=1, usecols=col_indices).fillna(0)
                    all_lead_data.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            else:
                # Rename to standard
                if lead_type == "2":
                    all_lead_data = all_lead_data.rename(columns={all_lead_data.columns[0]: 'II'})
                elif lead_type == "7":
                    all_lead_data = all_lead_data.rename(columns={
                        all_lead_data.columns[0]: 'I',
                        all_lead_data.columns[1]: 'II',
                        all_lead_data.columns[2]: 'III',
                        all_lead_data.columns[3]: 'aVR',
                        all_lead_data.columns[4]: 'aVL',
                        all_lead_data.columns[5]: 'aVF',
                        all_lead_data.columns[6]: 'V5'
                    })
                elif lead_type == "12":
                    all_lead_data = all_lead_data.rename(columns={
                        all_lead_data.columns[0]: 'I',
                        all_lead_data.columns[1]: 'II',
                        all_lead_data.columns[2]: 'III',
                        all_lead_data.columns[3]: 'aVR',
                        all_lead_data.columns[4]: 'aVL',
                        all_lead_data.columns[5]: 'aVF',
                        all_lead_data.columns[6]: 'V1',
                        all_lead_data.columns[7]: 'V2',
                        all_lead_data.columns[8]: 'V3',
                        all_lead_data.columns[9]: 'V4',
                        all_lead_data.columns[10]: 'V5',
                        all_lead_data.columns[11]: 'V6'
                    })

            # Check for duplicates
            for coll_name, sub_arr in collections_to_insert:
                collection = db[coll_name]
                exists = collection.count_documents({
                    'PatientID': patient_id,
                    'Arrhythmia': sub_arr,
                    'Lead': int(lead),
                    'Frequency': int(frequency)
                }) > 0
                if exists:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return JsonResponse({
                        "status": "error",
                        "message": f"Duplicate data found in collection '{coll_name}' with sub-arrhythmia '{sub_arr}'."
                    })
            # Chunk-wise insert
            chunk_size = 5000
            total_rows = len(all_lead_data)
            num_chunks = math.ceil(total_rows / chunk_size)
            columns = list(all_lead_data.columns)

            for chunk_index in range(num_chunks):
                start = chunk_index * chunk_size
                end = min((chunk_index + 1) * chunk_size, total_rows)
                chunk = all_lead_data.iloc[start:end]

                data_dict = {col.upper(): chunk[col].tolist() for col in columns}

                for coll_name, sub_arr in collections_to_insert:
                    db_insert_data = {
                        'Arrhythmia': sub_arr,
                        'Lead': int(lead),
                        'Frequency': int(frequency),
                        'PatientID': patient_id,
                        'Data': data_dict
                    }
                    db[coll_name].insert_one(db_insert_data)

            if os.path.exists(file_path):
                os.remove(file_path)

            return JsonResponse({
                "status": "success",
                "message": f"{num_chunks} chunks inserted successfully for PatientID {patient_id}."
            })

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})

    return JsonResponse({"status": "error", "message": "Invalid request!"})
 # API view for ECG data (Example)
def api_ecg_data(request):
    data = {"message": "ECG data API response"}
    return JsonResponse(data)

def new_table_data(request):
    return render(request,'oom_ecg_data/table.html')

def submit_form(request):
    # Your form handling logic here
    return render(request, 'oom_ecg_data/table.html')

@csrf_exempt
def fetch_ecg_data(request):
    # Handle POST request (form submission)
    if request.method == 'POST':
        try:
            patient_id = request.POST.get('patientId')
            lead_type = int(request.POST.get('leadType'))
            arrhythmia = request.POST.get('arrhythmia')
            frequency = int(request.POST.get('frequency'))

        except Exception as e:
            return JsonResponse({"status": "error", "message": "Invalid input: " + str(e)}, status=400)

        # Query without "Arrhythmia" field as it's used only for collection name
        query = {
            "PatientID": patient_id,
            "Lead": lead_type,
            "Frequency": frequency
        }

        # Try the arrhythmia-specific collection first
        collection = db[arrhythmia]
        total_count = collection.count_documents(query)

        # Fallback to default ECG_DATA collection if no results found
        if total_count == 0:
            collection = db["ECG_DATA"]
            total_count = collection.count_documents(query)

        total_pages = (total_count + 9) // 10

        if total_count == 0:
            # Return JSON response for no data
            return JsonResponse({
                "status": "error",
                "message": "No ECG data found for the given criteria",
                "total_count": total_count
            })

        first_page = list(collection.find(query).limit(10))
        page_obj = [
            {
                "object_id": str(r["_id"]),
                "PatientID": r.get("PatientID", ""),
                "Arrhythmia": r.get("Arrhythmia", ""),
                "Lead": r.get("Lead", ""),
                "Frequency": r.get("Frequency", "")
            }
            for r in first_page
        ]

        # Store query parameters in session for pagination
        request.session["ecg_query"] = {
            "patient_id": patient_id,
            "lead_type": lead_type,
            "frequency": frequency,
            "arrhythmia": arrhythmia
        }
        request.session["total_pages"] = total_pages
        request.session.modified = True

        # Return JSON response for success
        return JsonResponse({
            "status": "success",
            "data": page_obj,
            "total_pages": total_pages,
            "total_records": total_count,  # Added total_records for consistency
            "arrhythmia": arrhythmia
        })
    # Handle AJAX GET request for pagination
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        page_str = request.GET.get("page")
        if not page_str or not page_str.isdigit():
            return JsonResponse({"error": "Missing or invalid page parameter"}, status=400)

        page = int(page_str)
        ecg_query = request.session.get("ecg_query")

        if not ecg_query:
            return JsonResponse({"error": "No query stored"}, status=400)

        query = {
            "PatientID": ecg_query["patient_id"],
            "Lead": ecg_query["lead_type"],
            "Frequency": ecg_query["frequency"]
        }

        collection = db[ecg_query["arrhythmia"]]
        total_count = collection.count_documents(query)
        total_pages = (total_count + 9) // 10

        skip_count = (page - 1) * 10
        page_records = list(collection.find(query).skip(skip_count).limit(10))
        page_data = [
            {
                "object_id": str(r["_id"]),
                "PatientID": r.get("PatientID", ""),
                "Arrhythmia": r.get("Arrhythmia", ""),
                "Lead": r.get("Lead", ""),
                "Frequency": r.get("Frequency", "")
            }
            for r in page_records
        ]
        # Updated response to include total_records and total_pages
        return JsonResponse({
            "data": page_data,
            "total_records": total_count,  # Added to fix the frontend error
            "total_pages": total_pages    
        })
    return JsonResponse({"error": "Invalid request"}, status=400)

def fetch_random_ecg_data(request, arrhythmia):
    collection = db[arrhythmia]

    # Get query params
    page = int(request.GET.get("page", 1))
    page_size = 10
    skip = (page - 1) * page_size

    patient_id = request.GET.get("patientId", "").strip()
    lead = request.GET.get("lead", "").strip()
    arrhythmia_param = request.GET.get("arrhythmia", "").strip()
    frequency = request.GET.get("frequency", "").strip()

    query = {}
    if patient_id:
        query["PatientID"] = {"$regex": patient_id, "$options": "i"}
    if lead:
        query["Lead"] = lead if lead.isdigit() else 2 if lead == "II" else lead
    if arrhythmia_param:
        query["Arrhythmia"] = {"$regex": arrhythmia_param, "$options": "i"}
    if frequency:
        query["Frequency"] = {"$regex": frequency, "$options": "i"}

    total_count = collection.count_documents(query)
    total_pages = (total_count + page_size - 1) // page_size

    records = list(collection.find(query).skip(skip).limit(page_size))

    page_obj = [
        {
            "object_id": str(r["_id"]),
            "PatientID": r.get("PatientID", ""),
            "Arrhythmia": r.get("Arrhythmia", ""),
            "Lead": "II" if r.get("Lead", "") == 2 else r.get("Lead", ""),
            "LeadNumeric": r.get("Lead", ""),
            "Frequency": r.get("Frequency", ""),
            "collection_name": arrhythmia,
        }
        for r in records
    ]
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse({
            "status": "success",
            "data": page_obj,
            "total_records": total_count,
            "total_pages": total_pages,
            "current_page": page
        })

    return render(request, "oom_ecg_data/ecg_details.html", {
        "arrhythmia": arrhythmia,
        "page_obj": page_obj,
        "total_pages": total_pages,
        "current_page": page,
        "card_name": arrhythmia
    })

def get_object_id(request):

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = json.loads(request.body)
        patient_id = data.get("patientId")
        lead_type = str(data.get("lead"))
        arrhythmia_raw = data.get("selectedArrhythmia", "").strip()
        objectID = data.get("objectId")
        samples_taken = int(data.get("samplesTaken"))

        #Support multiple arrhythmias
        arrhythmia_list = [a.strip() for a in arrhythmia_raw.split(",") if a.strip()]

        if not objectID or not arrhythmia_list:
            return JsonResponse({"error": "Missing required parameters"}, status=400)

        objid = ObjectId(objectID)
        result = None
        found_collection = None

        #Try each arrhythmia until found
        for arrhythmia in arrhythmia_list:
            for key in arrhythmias_dict.keys():
                if key.lower() == arrhythmia.lower():
                    collection = db[key]
                    result = collection.find_one({"_id": objid})
                    if result:
                        found_collection = key
                        break
            if result:
                break

        # Fallback to ECG_DATA if not found anywhere
        if not result:
            collection = db["ECG_DATA"]
            result = collection.find_one({"_id": objid})

        if not result or "Data" not in result:
            return JsonResponse({"error": "ECG data not found"}, status=404)

        ecg_data_dict = result["Data"]

        # Normalize lead keys
        standard_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_mapping = {lead.lower(): lead for lead in standard_leads}

        normalized_ecg_data = {}
        for key, value in ecg_data_dict.items():
            norm_key = lead_mapping.get(str(key).lower().strip(), key)
            normalized_ecg_data[norm_key] = value

        # Return lead data
        if lead_type == "2":
            if "II" not in normalized_ecg_data:
                return JsonResponse({"error": "Lead II data not found"}, status=404)
            ecg_data = normalized_ecg_data["II"][:samples_taken]
            return JsonResponse({"x": list(range(len(ecg_data))), "ecgData": ecg_data})

        elif lead_type in ["7", "12"]:
            lead_sets = {
                "7": ["I", "II", "III", "aVR", "aVL", "aVF", "V5"],
                "12": ["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]
            }
            selected_leads = lead_sets.get(lead_type, [])
            extracted = {lead: normalized_ecg_data[lead][:samples_taken]
                         for lead in selected_leads if lead in normalized_ecg_data}
            if not extracted:
                return JsonResponse({"error": f"No valid leads found for {lead_type}-lead ECG"}, status=404)
            return JsonResponse({"ecgData": extracted})

        elif lead_type in normalized_ecg_data:
            ecg_data = normalized_ecg_data[lead_type][:samples_taken]
            return JsonResponse({"x": list(range(len(ecg_data))), "ecgData": ecg_data})

        else:
            return JsonResponse({"error": f"Lead {lead_type} data not found"}, status=404)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

@csrf_exempt
def edit_datas(request):
    db = client["ecgarrhythmias"]

    if request.method != "POST":
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)

    # CASE 1: Check if collection exists
    if 'collection_name' in data and len(data) == 1:
        collection_name = data['collection_name']
        existing_collections = db.list_collection_names()
        exists = collection_name in existing_collections
        return JsonResponse({'exists': exists})

    # CASE 2: Proceed to update
    patient_id = data.get('PatientID')
    object_id = data.get('object_id')
    old_collection_name = data.get('old_collection')
    new_collection_name = data.get('new_collection')
    lead = data.get('lead')

    if not all([object_id, old_collection_name, new_collection_name, lead]):
        return JsonResponse({'status': 'error', 'message': 'Missing required fields'}, status=400)

    try:
        lead = int(lead)  # Convert lead to integer
    except ValueError as e:
        return JsonResponse({'status': 'error', 'message': f'Invalid Lead value: {e}'}, status=400)

    try:
        obj_id = ObjectId(object_id)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Invalid ObjectId: {e}'}, status=400)

    find_collection = db[old_collection_name]
    new_collection = db[new_collection_name]

    # Try to fetch data from original collection
    fetched_data = find_collection.find_one({'_id': obj_id})
    if not fetched_data:
        # Try fallback collection
        fallback_collection = db["ECG_DATA"]  # Fixed: Use square bracket notation
        fetched_data = fallback_collection.find_one({'_id': obj_id})
        find_collection = fallback_collection  # for deletion
        if not fetched_data:
            return JsonResponse({'status': 'error', 'message': 'Data not found'}, status=404)

    # Use patient_id from DB if not passed
    if not patient_id:
        patient_id = fetched_data.get('PatientID')

    # Check for duplicates in the new collection
    duplicate = new_collection.find_one({
        'PatientID': patient_id,
        'Lead': lead,
        'Arrhythmia': new_collection_name
    })

    if duplicate:
        return JsonResponse({
            'status': 'error',
            'message': 'Duplicate data: This patient\'s ECG with same lead and arrhythmia already exists.'
        }, status=409)

    # Prepare new document
    update_arrhy_data = {
        'PatientID': patient_id,
        'Arrhythmia': new_collection_name,
        'Lead': lead,
        'Frequency': fetched_data.get('Frequency', 200),
        'Data': fetched_data.get('Data')
    }

    # Insert and Delete
    insert_result = new_collection.insert_one(update_arrhy_data)
    delete_result = find_collection.delete_one({'_id': obj_id})

    # Update session
    request.session["ecg_query"] = {
        "patient_id": patient_id,
        "lead_type": lead,
        "frequency": fetched_data.get('Frequency', 200),
        "arrhythmia": new_collection_name
    }
    request.session.modified = True

    return JsonResponse({
        'status': 'success',
        'message': 'Data edited and saved successfully',
        'new_object_id': str(insert_result.inserted_id)
    })
@csrf_exempt
def delete_data(request):

    if request.method != "POST":
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

    try:
        #  Extract values from form-data
        object_id = request.POST.get('object_id')
        if not object_id:
            return JsonResponse({'status': 'error', 'message': 'Missing object_id'}, status=400)

        #  Check all collections for this object_id
        found = False
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            result = collection.delete_one({'_id': ObjectId(object_id)})
            if result.deleted_count > 0:
                found = True
                break  # Stop after deleting the first match

        if not found:

            return JsonResponse({'status': 'error', 'message': 'Data not found'}, status=404)

        return JsonResponse({'status': 'success', 'message': 'Data deleted successfully'})

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
@csrf_exempt
def process_and_return_ecg(request):
    """
    Receives ECG data (x, y format), applies low-pass filtering and baseline correction,
    scales the signal between 0 and 4, and returns the processed ECG data as JSON.
    """
    try:
        # Ensure request is POST and contains JSON data
        if request.method != "POST":
            return JsonResponse({'error': 'Invalid request method'}, status=400)

        data = json.loads(request.body)  # Parse JSON request
        # Extract x and y values
        x_values = data.get("x", [])
        raw_ecg_data = data.get("y", [])  # Extract y values (ECG data)

        # Validate input
        if not raw_ecg_data or not x_values or len(x_values) != len(raw_ecg_data):
            return JsonResponse({'error': 'Invalid or missing ECG data'}, status=400)

        # Convert ECG data to NumPy array
        ecg_signal = np.array(raw_ecg_data, dtype=float)
        # Apply low-pass filter (cutoff frequency: 40 Hz)
        fs = 500  # Sampling frequency (adjust if needed)
        cutoff = 40
        b, a = signal.butter(3, cutoff / (fs / 2), btype='lowpass', analog=False)
        low_passed = signal.filtfilt(b, a, ecg_signal)

        # Baseline correction using median filter
        baseline = signal.medfilt(low_passed, kernel_size=131)
        corrected_ecg = low_passed - baseline

        # Scale the signal to range 0 to 4
        min_val = np.min(corrected_ecg)
        max_val = np.max(corrected_ecg)
        if max_val - min_val == 0:
            scaled_ecg = np.zeros_like(corrected_ecg)  # If constant signal, set to 0
        else:
            scaled_ecg = 4 * (corrected_ecg - min_val) / (max_val - min_val)
        
        # Return processed ECG data with original x-values
        return JsonResponse({'x': x_values, 'y': scaled_ecg.tolist()})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

def get_pqrst_data(request):
    if request.method != 'POST':
        return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

    try:
        data = json.loads(request.body)
        object_id = data.get("object_id")
        arrhythmia_raw = data.get("arrhythmia", "").strip()
        lead_config = data.get("lead_config")  # "2_lead", "7_lead", "12_lead"

        if not object_id or not arrhythmia_raw or not lead_config:
            return JsonResponse({
                "status": "error",
                "message": "Missing parameters: object_id, arrhythmia, or lead_config."
            }, status=400)

        if lead_config not in ["2_lead", "7_lead", "12_lead"]:
            return JsonResponse({
                "status": "error",
                "message": "Invalid lead_config. Must be '2_lead', '7_lead', or '12_lead'."
            }, status=400)

        #Handle multiple arrhythmias
        arrhythmia_list = [a.strip() for a in arrhythmia_raw.split(",") if a.strip()]
        record = None
        found_collection = None

        for arr in arrhythmia_list:
            # Find matching collection name
            collection_name = None
            for key in arrhythmias_dict.keys():
                if key.lower() == arr.lower():
                    collection_name = key
                    break

            if not collection_name:
                continue  # Try next arrhythmia

            collection = db[collection_name]
            record = collection.find_one({"_id": ObjectId(object_id)})
            if record and "Data" in record:
                found_collection = collection_name
                break  # Found ECG data, stop searching

        # Fallback to ECG_DATA if nothing found
        if not record:
            collection = db["ECG_DATA"]
            record = collection.find_one({"_id": ObjectId(object_id)})
            if record and "Data" in record:
                found_collection = "ECG_DATA"

        if not record or "Data" not in record:
            return JsonResponse({"status": "error", "message": "Invalid or missing ECG data."}, status=404)


        frequency = int(record.get("Frequency", 200))

        # Normalize Data keys to standard lead names
        standard_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_mapping = {lead.lower(): lead for lead in standard_leads}

        normalized_data = {}
        for key, value in record["Data"].items():
            norm_key = lead_mapping.get(str(key).lower().strip(), key)
            normalized_data[norm_key] = value

        # Expected leads by config
        if lead_config == "2_lead":
            expected_leads = ["II"]
        elif lead_config == "7_lead":
            expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V5"]
        else:  # 12_lead
            expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF",
                              "V1", "V2", "V3", "V4", "V5", "V6"]

        available_leads = [lead for lead in expected_leads if lead in normalized_data]
        if not available_leads:
            return JsonResponse({
                "status": "error",
                "message": f"No valid leads found. Expected: {expected_leads}"
            }, status=404)

        # Create DataFrame for processing
        lead_data = {lead: normalized_data[lead] for lead in available_leads}
        df = pd.DataFrame(lead_data)

        # Process ECG data
        r_index = check_r_index(df, lead_config, frequency, r_index_model)
        s_index, q_index = check_qs_index(df, r_index, lead_config)
        t_index, p_index, _, _, _ = check_pt_index(df, lead_config, r_index)

        return JsonResponse({
            "status": "success",
            "r_peaks": [int(i) for i in r_index],
            "q_points": [int(i) for i in q_index],
            "s_points": [int(i) for i in s_index],
            "p_points": [int(i) for i in p_index],
            "t_points": [int(i) for i in t_index],
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

# Download selected data
@csrf_exempt
def selecteddownload(request):
    try:
        data = json.loads(request.body)
        if not data:
            return JsonResponse({'error': 'No data received'}, status=400)

        # Initialize buffer for CSV
        buffer = io.StringIO()

        # Handle 2-lead ECG (single lead)
        if 'x' in data and 'y' in data:
            df = pd.DataFrame({
                'TimeIndex': data['x'],
                'II': data['y']
            })
            df.to_csv(buffer, index=False, encoding='utf-8')

        # Handle 7/12-lead ECG (multiple leads)
        elif 'leadDict' in data:
            lead_dict = data['leadDict']
            if not lead_dict:
                return JsonResponse({'error': 'No lead data provided'}, status=400)

            # Create DataFrame with all leads
            lead_names = list(lead_dict.keys())
            first_lead = lead_names[0]
            df_data = {
                'TimeIndex': lead_dict[first_lead]['x']
            }
            for lead in lead_names:
                if len(lead_dict[lead]['x']) == len(lead_dict[first_lead]['x']):
                    df_data[lead] = lead_dict[lead]['y']
                else:
                    return JsonResponse({'error': f'Inconsistent data length for lead {lead}'}, status=400)

            df = pd.DataFrame(df_data)
            df.to_csv(buffer, index=False, encoding='utf-8')

        else:
            return JsonResponse({'error': 'Invalid data format'}, status=400)

        buffer.seek(0)
        response = HttpResponse(buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="selected_ecg_data.csv"'
        return response

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)  

@csrf_exempt
@require_POST
def get_multiple_segments(request):
    try:
        data = json.loads(request.body)

        lead = int(data.get("lead"))
        frequency = int(data.get("frequency"))
        arrhythmia_data = data.get("arrhythmiaData", [])
      
        session_query_list = []

        for index, group in enumerate(arrhythmia_data):
            arrhythmia = group.get("arrhythmia")
            duration = int(group.get("duration"))
            sample_count = frequency * duration

            try:
                collection = db[arrhythmia]
                all_records = list(collection.find({"Lead": lead}))
                random.shuffle(all_records)

                total_collected = 0
                selected_segments = []
                seen_patients_in_group = set()

                required_channels = {
                    2: ["II"],
                    7: ["I", "II", "III", "AVR", "AVL", "AVF", "V5"],
                    12: ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                }.get(lead)

                if not required_channels:
                    continue

                for record in all_records:
                    patient_id = record.get("PatientID", "N/A")
                    if patient_id in seen_patients_in_group:
                        continue

                    if "Data" not in record:
                        continue
                    if not all(ch in record["Data"] for ch in required_channels):
                        continue

                    min_len = min(len(record["Data"][ch]) for ch in required_channels)
                    if min_len <= 0:
                        continue

                    remaining_needed = sample_count - total_collected
                    take = min(remaining_needed, min_len)

                    total_collected += take
                    seen_patients_in_group.add(patient_id)

                    selected_segments.append({
                        "object_id": str(record.get("_id", "")),
                        "PatientID": patient_id,
                        "Arrhythmia": record.get("Arrhythmia", arrhythmia),
                        "Lead": "II" if lead == 2 else lead,
                        "LeadNumeric": lead,
                        "Frequency": frequency,
                        "SamplesTaken": take,
                        "collection_name": arrhythmia,
                    })

                    if total_collected >= sample_count:
                        break

                if total_collected >= sample_count:
                    session_query_list.append({
                        "arrhythmia": arrhythmia,
                        "lead": lead,
                        "frequency": frequency,
                        "duration": duration,
                        "segments": selected_segments
                    })
                else:
                    print(f"Not enough data for arrhythmia {arrhythmia}, only got {total_collected} samples.")

            except Exception as inner_e:
                print(f"Error while processing arrhythmia '{arrhythmia}':", str(inner_e))

        # Store only lightweight session data (no ECG signals)
        request.session["multi_ecg_query"] = session_query_list
        request.session["segment_mode"] = True
        request.session.modified = True

        # Prepare frontend data (no duplication)
        flattened_segments = []
        seen_patients_global = set()

        for group in session_query_list:
            for seg in group.get("segments", []):
                patient_id = seg.get("PatientID", "N/A")
                if patient_id in seen_patients_global:
                    continue
                seen_patients_global.add(patient_id)

                flattened_segments.append({
                    "patient_id": patient_id,
                    "arrhythmia": group.get("arrhythmia", ""),
                    "lead": seg.get("Lead", group.get("lead", "")),
                    "frequency": seg.get("Frequency", group.get("frequency", "")),
                    "samples_taken": seg.get("SamplesTaken", 0),
                    # Still exclude "data" here to keep response lightweight
                })

        return JsonResponse({
            "status": "success" if flattened_segments else "error",
            "data": flattened_segments,
            "total_pages": 1 if flattened_segments else 0,
            "total_records": len(flattened_segments),
            "arrhythmia": session_query_list[0]['arrhythmia'] if flattened_segments else ""
        })

    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": str(e),
            "data": []
        })

def ecg_details(request, arrhythmia):
    normalized_arrhythmia = arrhythmia.strip().replace("_", " ")

    full_data = []

    ecg_data = request.session.get("multi_ecg_query", [])
    is_segment_mode = request.session.get("segment_mode", False)

    if is_segment_mode and ecg_data:
        arrhythmia_data = [
            entry for entry in ecg_data
            if entry.get("arrhythmia", "").lower() == normalized_arrhythmia
        ]
        for entry in arrhythmia_data:
            for seg in entry.get("segments", []):
                full_data.append({
                    "object_id": seg.get("object_id", ""),
                    "PatientID": seg.get("PatientID", ""),
                    "Arrhythmia": seg.get("Arrhythmia", ""),
                    "Lead": seg.get("Lead", ""),
                    "LeadNumeric": seg.get("LeadNumeric", ""),
                    "Frequency": seg.get("Frequency", ""),
                    "collection_name": seg.get("collection_name", normalized_arrhythmia),
                    "samples_taken": seg.get("SamplesTaken", 0),  #WORKING VERSION
                })
        # for entry in arrhythmia_data:
        #     for seg in entry.get("segments", []):
        #         full_data.extend(entry.get("segments", []))
    else:
        ecg_query = request.session.get("ecg_query", {})
        if not ecg_query:
            return render(request, "oom_ecg_data/ecg_details.html", {
                "page_obj": [], "total_pages": 1, "arrhythmia": arrhythmia, "show_alert": True
            })

        query = {
            "PatientID": ecg_query["patient_id"],
            "Lead": ecg_query["lead_type"],
            "Frequency": ecg_query["frequency"]
        }

        collection = db[normalized_arrhythmia]
        results = list(collection.find(query).limit(1000))
        for r in results:
            full_data.append({
                "object_id": str(r["_id"]),
                "PatientID": r.get("PatientID", ""),
                "Arrhythmia": r.get("Arrhythmia", arrhythmia),
                "Lead": "II" if r.get("Lead", "") == 2 else r.get("Lead", ""),
                # "LeadNumeric": r.get("Lead", ""),
                "Frequency": r.get("Frequency", ""),
                "collection_name": normalized_arrhythmia,
                "samples_taken": seg.get("SamplesTaken", 0)  #Added

            })

    # Pagination
    paginator = Paginator(full_data, 10)
    page = int(request.GET.get("page", 1))
    try:
        page_data = paginator.page(page)
    except:
        page_data = paginator.page(1)
    # AJAX: Return JSON
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        response_data = ({
            "status": "success",
            "data": list(page_data),
            "total_records": paginator.count,
            "total_pages": paginator.num_pages,
            "current_page": page
        })
        return JsonResponse(response_data)

    # Normal HTML render
    return render(request, "oom_ecg_data/ecg_details.html", {
        "page_obj": page_data,
        "total_pages": paginator.num_pages,
        "current_page": page,
        "arrhythmia": arrhythmia,
        "card_name": arrhythmia
    })
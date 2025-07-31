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
from django.views.decorators.http import require_POST
import plotly.graph_objects as go
import plotly.io as pio
import traceback
import io
from .PQRST_detection_model import check_r_index, check_qs_index, check_pt_index, r_index_model, pt_index_model


client = pymongo.MongoClient("mongodb://192.168.1.65:27017/")
db = client["ecgarrhythmias"]

arrhythmias_dict = {
'Myocardial Infarction': ['T-wave abnormalities', 'Inferior MI', 'Lateral MI'],
'Atrial Fibrillation & Atrial Flutter': ['Afib', 'Aflutter'],
'HeartBlock': ['I Degree', 'II Degree', 'III Degree'],
'Junctional Rhythm': ['Junctional Bradycardia', 'Junctional Rhythm'],
'Premature Atrial Contraction': ['Isolated', 'Bigeminy', 'Couplet','Triplet', 'SVT','Trigeminy','Quadrigeminy'],
'Premature Ventricular Contraction': ['AIVR', 'Bigeminy', 'Couplet', 'Triplet', 'Isolated', 'NSVT', 'Quadrigeminy', 'Trigeminy', 'LBBB','IVR','VT'],
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
            arrhythmia_mi = request.POST.get("newarrhythmiaMI")
            sub_arrhythmia = request.POST.get("subArrhythmia")
            frequency = request.POST.get("newfrequency")
            lead_type = request.POST.get("lead")
            lead = lead_type.split(' ')[0]
            collection = db[arrhythmia_mi]

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
            exists = collection.count_documents({
                'PatientID': patient_id,
                'Arrhythmia': sub_arrhythmia,
                'Lead': int(lead),
                'Frequency': int(frequency)
            }) > 0

            if exists:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return JsonResponse({"status": "error", "message": "Duplicate data found."})

            # Chunk-wise insert
            chunk_size = 5000
            total_rows = len(all_lead_data)
            num_chunks = math.ceil(total_rows / chunk_size)
            columns = list(all_lead_data.columns)

            for chunk_index in range(num_chunks):
                start = chunk_index * chunk_size
                end = min((chunk_index + 1) * chunk_size, total_rows)
                chunk = all_lead_data.iloc[start:end]

                data_dict = {}
                for col in columns:
                    data_dict[col.upper()] = chunk[col].tolist()

                db_insert_data = {
                    'Arrhythmia': sub_arrhythmia,
                    'Lead': int(lead),
                    'Frequency': int(frequency),
                    'PatientID': patient_id,
                    'Data': data_dict
                }

                collection.insert_one(db_insert_data)

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
    
def ecg_details(request, arrhythmia):
    # Retrieve query from session
    ecg_query = request.session.get("ecg_query", {})
    total_pages = request.session.get("total_pages", 1)
    
    # If no query, redirect or show empty page
    if not ecg_query:
        return render(request, "oom_ecg_data/ecg_details.html", {
            "page_obj": [],
            "total_pages": 1,
            "arrhythmia": arrhythmia,
            "show_alert": True
        })

    query = {
        "PatientID": ecg_query["patient_id"],
        "Lead": ecg_query["lead_type"],
        "Frequency": ecg_query["frequency"]
    }

    collection = db[arrhythmia]
    total_count = collection.count_documents(query)
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

    return render(request, "oom_ecg_data/ecg_details.html", {
        "page_obj": page_obj,
        "total_pages": total_pages,
        "arrhythmia": arrhythmia,
        "card_name": arrhythmia
    })

@csrf_exempt
def get_object_ids(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            patient_id = data.get("patientId")
            lead_type = data.get("lead")
            arrhythmia = data.get("selectedArrhythmia")
            objectID = data.get("objectId")

            collection = db[arrhythmia]
            objid = ObjectId(objectID)
            result = collection.find_one({"_id": objid})

            if result is None:
                collection = db["ECG_DATA"]
                result = collection.find_one({"_id": objid})

            if not result or "Data" not in result:
                return JsonResponse({"error": "ECG data not found"}, status=404)

            ecg_data_dict = result["Data"]

            # Normalize keys (case-insensitive)
            standard_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            lead_mapping = {lead.lower(): lead for lead in standard_leads}

            normalized_ecg_data = {}
            for key, value in ecg_data_dict.items():
                norm_key = lead_mapping.get(str(key).lower().strip(), key)
                normalized_ecg_data[norm_key] = value

            # Filter leads
            if lead_type == "2":
                if "II" in normalized_ecg_data:
                    ecg_data = normalized_ecg_data["II"][:2000]
                    return JsonResponse({
                        "x": list(range(len(ecg_data))),
                        "ecgData": ecg_data
                    })
                else:
                    return JsonResponse({"error": "Lead II data not found"}, status=404)

            elif lead_type in ["7", "12"]:
                lead_sets = {
                    "7": ["I", "II", "III", "aVR", "aVL", "aVF", "V5"],
                    "12": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                }

                selected_leads = lead_sets.get(lead_type, [])
                extracted = {
                    lead: normalized_ecg_data[lead][:2000]
                    for lead in selected_leads if lead in normalized_ecg_data
                }

                if not extracted:
                    return JsonResponse({"error": f"No valid leads found for {lead_type}-lead ECG"}, status=404)

                return JsonResponse({
                    "ecgData": extracted
                })

            elif lead_type in normalized_ecg_data:
                ecg_data = normalized_ecg_data[lead_type][:2000]
                return JsonResponse({
                    "x": list(range(len(ecg_data))),
                    "ecgData": ecg_data
                })
            else:
                return JsonResponse({"error": f"Lead {lead_type} data not found"}, status=404)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)

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

@csrf_exempt
def delete_file(request):
    try:
        data = json.loads(request.body)
        file_path = data.get('filePath')
        
        if not file_path:
            return JsonResponse({'status': 'error', 'message': 'File path is required'}, status=400)

        # Construct the full file path relative to the media directory
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        # Security: Prevent directory traversal
        if not full_path.startswith(settings.MEDIA_ROOT):
            return JsonResponse({'status': 'error', 'message': 'Invalid file path'}, status=400)

        if os.path.exists(full_path):
            os.remove(full_path)
            return JsonResponse({'status': 'success', 'message': 'File deleted successfully'})
        else:
            return JsonResponse({'status': 'error', 'message': 'File not found'}, status=404)
            
    except json.JSONDecodeError as e:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)     
@csrf_exempt
def get_pqrst_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            object_id = data.get("object_id")
            arrhythmia = data.get("arrhythmia")
            lead_config = data.get("lead_config")  # New parameter for lead configuration

            if not object_id or not arrhythmia or not lead_config:
                return JsonResponse({"status": "error", "message": "Missing parameters: object_id, arrhythmia, or lead_config."}, status=400)

            if lead_config not in ["2_lead", "7_lead", "12_lead"]:
                return JsonResponse({"status": "error", "message": "Invalid lead_config. Must be '2_lead', '7_lead', or '12_lead'."}, status=400)

            collection = db[arrhythmia]
            record = collection.find_one({"_id": ObjectId(object_id)})

            if not record or "Data" not in record:
                return JsonResponse({"status": "error", "message": "Invalid or missing ECG data."}, status=404)

            frequency = int(record.get("Frequency", 200))

            # Define expected leads based on configuration
            if lead_config == "2_lead":
                expected_leads = ["II"]
            elif lead_config == "7_lead":
                expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "v5"]
            else:  # 12_lead
                expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "v1", "v2", "v3", "v4", "v5", "v6"]

            # Verify required leads are present
            available_leads = [lead for lead in expected_leads if lead in record["Data"]]
            if not available_leads:
                return JsonResponse({"status": "error", "message": f"No valid leads found. Expected leads: {expected_leads}."}, status=404)

            # Create DataFrame with available leads
            lead_data = {lead: record["Data"][lead] for lead in available_leads if lead in record["Data"]}
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

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

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
        print("Error:", str(e))  # Debug
        return JsonResponse({'error': str(e)}, status=500)  
    
@csrf_exempt
@require_POST
def rytham_data_insert(request):
    try:
        # Extract FormData
        patient_id = request.POST.get("patientId")
        arrhythmia_mi = request.POST.get("newarrhythmiaMI_multiple")
        sub_arrhythmia = request.POST.get("subArrhythmia")
        frequency = request.POST.get("newfrequency")
        lead_type = request.POST.get("lead")
        ecg_data_str = request.POST.get("ecgData")

        # Validate required fields
        if not all([patient_id,arrhythmia_mi, sub_arrhythmia, frequency, lead_type, ecg_data_str]):
            return JsonResponse({"status": "error", "message": "Missing required fields."})

        # Validate lead type
        if lead_type not in ["2", "7", "12"]:
            return JsonResponse({"status": "error", "message": f"Invalid lead type: {lead_type}."})

        # Parse ecgData JSON
        try:
            ecg_data = json.loads(ecg_data_str)
        except json.JSONDecodeError:
            return JsonResponse({"status": "error", "message": "Invalid ecgData format. Must be valid JSON."})

        # Connect to MongoDB
        collection = db[arrhythmia_mi]

        # Validate ecgData structure
        expected_samples = 60000  # 10 seconds at 200 Hz
        if lead_type == "2":
            if not isinstance(ecg_data, dict) or "x" not in ecg_data or "y" not in ecg_data:
                return JsonResponse({"status": "error", "message": "Invalid ecgData for lead type 2. Must contain 'x' and 'y' arrays."})
            if not isinstance(ecg_data["x"], list) or not isinstance(ecg_data["y"], list):
                return JsonResponse({"status": "error", "message": "x and y must be arrays."})
            if len(ecg_data["x"]) != len(ecg_data["y"]) or len(ecg_data["y"]) > expected_samples:
                return JsonResponse({"status": "error", "message": f"Invalid data length. Expected up to {expected_samples} samples."})
            data_dict = {"II": ecg_data["y"]}  # Standardize column name
        else:  # lead_type == "7" or "12"
            if not isinstance(ecg_data, dict) or "leadDict" not in ecg_data:
                return JsonResponse({"status": "error", "message": "Invalid ecgData for lead type 7 or 12. Must contain 'leadDict'."})
            lead_dict = ecg_data["leadDict"]
            expected_lead_count = 7 if lead_type == "7" else 12
            if len(lead_dict) != expected_lead_count:
                return JsonResponse({"status": "error", "message": f"Expected {expected_lead_count} leads, but found {len(lead_dict)}."})
            data_dict = {}
            standard_leads = {
                "7": ["I", "II", "III", "aVR", "aVL", "aVF", "V5"],
                "12": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            }[lead_type]
            for i, (lead, values) in enumerate(lead_dict.items()):
                if not isinstance(values, list) or len(values) > expected_samples:
                    return JsonResponse({"status": "error", "message": f"Lead {lead} data invalid or exceeds {expected_samples} samples."})
                data_dict[standard_leads[i]] = values  # Map to standard lead names

        # Check for duplicates
        exists = collection.count_documents({
            "PatientID": patient_id,
            "Arrhythmia": sub_arrhythmia,
            "Lead": int(lead_type),
            "Frequency": int(frequency)
        }) > 0

        if exists:
            return JsonResponse({"status": "error", "message": "Duplicate data found."})

        # Insert data into MongoDB
        db_insert_data = {
            "Arrhythmia": sub_arrhythmia,
            "Lead": int(lead_type),
            "Frequency": int(frequency),
            "PatientID": patient_id,
            "Data": data_dict
        }
        collection.insert_one(db_insert_data)

        return JsonResponse({
            "status": "success",
            "message": f"ECG data inserted successfully for PatientID {patient_id}."
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})

    return JsonResponse({"status": "error", "message": "Invalid request!"})

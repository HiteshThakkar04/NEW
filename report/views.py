from django.shortcuts import render
from django.http import JsonResponse
from pymongo import MongoClient
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
 
client = MongoClient("mongodb://192.168.1.65:27017/")
db = client['ecgarrhythmias']
 
def process_collection_data(collection_name):
    """Process collection data asynchronously to speed up execution."""
    collection = db[collection_name]
    # Get distinct patient IDs for counting unique patients
    unique_patient_ids = collection.distinct("PatientID")
 
    # Get total document count for this collection
    total_documents = collection.count_documents({})
 
    # Aggregation to calculate total duration and length per patient
    pipeline = [
        {"$unwind": "$Data"},
        {"$match": {"Data.II": {"$exists": True, "$type": "array"}}},
        {"$group": {
            "_id": "$PatientID",
            "total_data_length": {"$sum": {"$size": "$Data.II"}},
            "document_count": {"$sum": 1},
        }},
    ]
    patient_data_list = list(collection.aggregate(pipeline))
 
    # Prepare patient data dictionary
    patient_data = {}
    for doc in patient_data_list:
        patient_id = doc["_id"]
        patient_data[patient_id] = {
            "document_count": doc["document_count"],
            "total_data_length": doc["total_data_length"]
        }
 
    # Fetch 10 recent documents with required fields
    recent_docs = list(
        collection.find({}, {"PatientID": 1, "Data": 1, "Frequency": 1, "Arrhythmia": 1})
        .sort("_id", -1)
        .limit(10)
    )
    return total_documents, unique_patient_ids, patient_data, recent_docs
 
def index(request):
    all_collections = db.list_collection_names()
    piechart_collections = ['Premature Ventricular Contraction', 'Ventricular Fibrillation and Asystole', 'HeartBlock', 'Atrial Fibrillation & Atrial Flutter', 'Myocardial Infarction', 'Premature Atrial Contraction', 'Junctional Rhythm','Noise','Others','LBBB & RBBB']
    total_documents = 0
    grand_total_seconds = 0
    patient_data = {}
    collection_data_count = {}
    recent_patient_records = OrderedDict()
 
    # ======================== PROCESS ALL COLLECTIONS IN PARALLEL ========================
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_collection_data, all_collections))
 
    # ======================== AGGREGATE DATA FROM RESULTS ========================
    for total_docs, unique_patient_ids, collection_patient_data, recent_docs in results:
        total_documents += total_docs
 
        for patient_id, data in collection_patient_data.items():
            if patient_id not in patient_data:
                patient_data[patient_id] = {"document_count": 0, "total_data_length": 0}
            patient_data[patient_id]["document_count"] += data["document_count"]
            patient_data[patient_id]["total_data_length"] += data["total_data_length"]
 
            # Add duration for each patient
            grand_total_seconds += data["total_data_length"] / 200  # Assuming default frequency = 200Hz
 
        # Process recent records for latest 5 patients
        for doc in recent_docs:
            patient_id = doc.get("PatientID", "Unknown")
            if patient_id not in recent_patient_records:
                ecg_data = doc.get("Data", {})
                lead_ii_data = []
 
                if isinstance(ecg_data, list) and len(ecg_data) > 0 and isinstance(ecg_data[0], dict):
                    lead_ii_data = ecg_data[0].get("II", [])
                elif isinstance(ecg_data, dict):
                    lead_ii_data = ecg_data.get("II", [])
 
                recording_time = round(len(lead_ii_data) / doc.get("Frequency", 200), 2) if isinstance(lead_ii_data, list) else 0
 
                recent_patient_records[patient_id] = {
                    "patient_id": patient_id,
                    "lead": "II" if isinstance(lead_ii_data, list) and len(lead_ii_data) > 0 else "Unknown",
                    "arrhythmia": doc.get("Arrhythmia", "Unknown"),
                    "recording_time": recording_time
                }
 
                if len(recent_patient_records) >= 10:
                    break
 
    # ======================== PIE CHART COLLECTION COUNT ========================
    for collection_name in piechart_collections:
        collection = db[collection_name]
        count = collection.count_documents({})
        if count > 0:
            collection_data_count[collection_name] = round((count * 10) / 60, 2)
 
    # ======================== FINAL CALCULATIONS ========================
    total_patients = len(patient_data)
    grand_total_minutes = round(grand_total_seconds / 60, 2)
 
    # Sort top 10 patients by total ECG data length
    top_10_patients = sorted(
        patient_data.items(), key=lambda x: x[1]["total_data_length"], reverse=True
    )[:10]
 
    # ======================== RENDER TEMPLATE ========================
    return render(request, "report/report_index.html", {
        "total_patients": total_patients,
        "total_records": total_documents,
        "total_time": grand_total_minutes,
        "top_10_patients": [
            {"patient_id": pid, "total_data_length": data["total_data_length"]}
            for pid, data in top_10_patients
        ],
        "recent_records": list(recent_patient_records.values()),
        "collection_data_count": collection_data_count,
    })
 
# API view for ECG data (Example)
def api_ecg_data(request):
    data = {"message": "ECG data API response"}
    return JsonResponse(data)
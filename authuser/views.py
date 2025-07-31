from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password, check_password
from django.conf import settings
from django.contrib import messages
from pymongo import MongoClient
from django.http import JsonResponse
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from django.views.decorators.csrf import csrf_protect
import json

# Connect to MongoDB
mongo_client = MongoClient("mongodb://192.168.1.65:27017/")
db = mongo_client['ecgarrhythmias']
users_collection = db["users"]

def home(request):
    return render(request, 'authuser/login.html')


# User Registration
@csrf_protect
def register(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')  # case-sensitive key!
        phone = request.POST.get('phone')
        password = request.POST.get('password')
        hashed_password = make_password(password)

        # Check if user already exists
        if users_collection.find_one({"username": username}):
            messages.error(request, "Username already exists.")
            return redirect('register')

        # Insert user into MongoDB
        users_collection.insert_one({
            "username": username,
            "email": email,
            "phone": phone,
            "password": hashed_password
        })

        messages.success(request, "Registration successful! You can log in now.")
        return redirect('login')  # Make sure 'login' is a valid URL name in your urls.py

    return render(request, 'authuser/register.html')

# User Login
def login(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = users_collection.find_one({"username": username})
        if user and check_password(password, user['password']):
           
            request.session['user_session'] = {
                "username": user['username'],
                "email": user['email'],
                "phone": user.get('phone', None)  # Use get() to handle missing phone
            }   # Store session
           
            # messages.success(request, "Login successful!")
            return redirect('dashboard')
 
        messages.error(request, "Invalid username or password.")
        return redirect('login')
 
    return render(request, 'authuser/login.html')

def profile(request):
    if 'user_session' not in request.session:
        messages.error(request, "You need to log in first.")
        return redirect('login')

    user_session = request.session['user_session']
    context = {
        "username": user_session['username'],
        "email": user_session['email'],
    }
    return render(request, 'authuser/profile.html', context)

@csrf_protect
def change_password(request):
    if 'user_session' not in request.session:
        return JsonResponse({"error": "You need to log in first."}, status=401)

    if request.method == "POST":
        try:
            data = json.loads(request.body)
            current_password = data.get('currentPassword')
            new_password = data.get('newPassword')

            user_session = request.session['user_session']
            user = users_collection.find_one({"username": user_session['username']})

            if not user or not check_password(current_password, user['password']):
                return JsonResponse({"error": "Current password is incorrect."}, status=400)

            hashed_new_password = make_password(new_password)
            users_collection.update_one(
                {"username": user_session['username']},
                {"$set": {"password": hashed_new_password}}
            )

            return JsonResponse({"message": "Password changed successfully."}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)

# User Dashboard
def dashboard(request):
    return redirect('/ommecgdata/')

# User Logout
def logout(request):
    request.session.flush()  # Clear session
    messages.success(request, "Logged out successfully.")
    return redirect('login')

# ======================== process_collection_data ========================
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

# ======================== TOTAL DATA FUNCTION ========================
def total_data(request):

    all_collections = db.list_collection_names()
    total_documents = 0
    grand_total_seconds = 0
    patient_data = {}

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

    total_patients = len(patient_data)
    grand_total_minutes = round(grand_total_seconds / 60, 2)
    context = {
        "total_patients": total_patients,
        "total_records": total_documents,
        "total_time": grand_total_minutes,
    }

    # Return JSON if requested via AJAX
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse(context)

    return render(request, "authuser/login.html", context)

@csrf_protect
def update_profile(request):
    if 'user_session' not in request.session:
        return JsonResponse({"error": "You need to log in first."}, status=401)

    if request.method == "POST":
        try:
            data = json.loads(request.body)
            username = data.get('username')
            email = data.get('email')
            phone = data.get('phone')  # Phone can be None or a valid value

            # Check if required fields (username and email) are provided
            if not username or not email:
                return JsonResponse({"error": "Username and email are required."}, status=400)

            current_username = request.session['user_session']['username']
            
            # Prepare update data
            update_data = {
                "username": username,
                "email": email,
            }
            
            # Only include phone in update if it's provided (not None or empty)
            if phone is not None:
                update_data["phone"] = phone

            # Update user in the database
            users_collection.update_one(
                {"username": current_username},
                {"$set": update_data}
            )

            # Update session with new data
            request.session['user_session'] = {
                "username": username,
                "email": email,
                "phone": phone if phone is not None else request.session['user_session'].get('phone'),
            }

            return JsonResponse({
                "success": True,
                "message": "Profile updated successfully.",
                "profile": {
                    "username": username,
                    "email": email,
                    "phone": phone if phone is not None else request.session['user_session'].get('phone')
                }
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)
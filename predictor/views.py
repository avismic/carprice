# predictor/views.py
import os
import pickle
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

# Determine the base directory of your project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build file paths for the model and CSV
model_path = os.path.join(BASE_DIR, 'data', 'LinearRegressionModel.pkl')
csv_path = os.path.join(BASE_DIR, 'data', 'Cleaned_Car_data.csv')

# Load the model and data
model = pickle.load(open(model_path, 'rb'))
car = pd.read_csv(csv_path)

def index(request):
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = list(car['fuel_type'].unique())

    companies.insert(0, 'Select Company')

    context = {
        'companies': companies,
        'car_models': car_models,
        'years': years,
        'fuel_types': fuel_types,
    }
    return render(request, 'index.html', context)

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        company = request.POST.get('company')
        car_model = request.POST.get('car_models')
        year = request.POST.get('year')
        fuel_type = request.POST.get('fuel_type')
        driven = request.POST.get('kilo_driven')

        input_data = pd.DataFrame(
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
        )

        prediction = model.predict(input_data)
        result = np.round(prediction[0], 2)
        return HttpResponse(str(result))
    else:
        return HttpResponse("Only POST requests are accepted.", status=405)

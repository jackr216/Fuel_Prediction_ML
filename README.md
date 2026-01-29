# Fuel Spend Prediction ML Project

This project predicts fuel spend per vehicle for a fleet using anonymised data. It demonstrates data cleaning, feature engineering, and machine learning modeling in Python.

---

## Project Structure

Fuel_Prediction_ML/
Anon_Data/ # Anonymised Excel datasets
//fuel_data_anon.xlsx
//mileage_data_anon.xlsx
//vehicle_data_anon.xlsx
/Fuel.Spend.ipynb # Main notebook for cleaning, modeling, and predictions
/.gitignore # Git ignore file
/README.md # This file


---

## Datasets (Anonymised)

- **fuel_data_anon.xlsx** — Historical fuel transactions per vehicle.
- **mileage_data_anon.xlsx** — Vehicle mileage per month.
- **vehicle_data_anon.xlsx** — Vehicle metadata (type, make, model, registration).

> All data is anonymised to protect sensitive information. Original company data is **not included**.

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/jackr216/Fuel_Prediction_ML.git
cd Fuel_Prediction_ML

pip install required libraries

jupyter notebook Fuel.Spend.ipynb


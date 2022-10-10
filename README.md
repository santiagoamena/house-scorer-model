# House Scorer Model

HouseScorerModel is a class used to fit a gradient boosting model to the challenge_houses-prices dataset and get predictions.

## Repository overview

```
├── README.md
├── data      <- the house prices dataset
├── notebooks <- notebook used for developing and analyzing the model
    └── model_development.ipynb 
└── src       <- source code for the HouseScorerModel class   
    └── housescorermodel.py
└── requirements.txt
```

## Usage
1) Clone this repository
2) Install all the dependencies in the requirements.txt file
3) Import module from source code and execute the functions:

```python
from src.housescorermodel import HouseScorerModel

# Loads data and defines the model
hsm = HouseScorerModel("data/challenge_houses-prices.csv")

# Fits the model
hsm.fit()

# Returns model performance on train and test sets
hsm.get_model_performance()

# Predicts sale price for a single data point
data_point = {
    "garage_area": 547,
    "overall_quality": 6,
    "overall_condition": 5,
    "bsmt_area": 621,
    "property_area": 2016,
    "bath_area": 2.5,
    "has_pool": 0,
    "has_porch": 1,
    "has_2ndfloor": 1,
    "has_multiple_kitchen": 0,
    "liv_lot_ratio": 0.0,
    "spaciousness": 245.0,
    "house_age": 14,
    "remodel_age": 11,
    "garage_age": 15.0,
    "neighborhood": "Gilbert",
    "house_style": "2Story",
}
hsm.get_prediction(data_point)
```

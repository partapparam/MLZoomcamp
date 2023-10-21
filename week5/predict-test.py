#!/usr/bin/env python
# coding: utf-8

import requests

host = "churn-serving.eba-cazpz7ee.us-west-1.elasticbeanstalk.com"
url = f"http://{host}/predict"

customer_id = "xyz-123"
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 6,
    "monthlycharges": 29.85,
    "totalcharges": (12 * 29.85),
}


response = requests.post(url, json=customer).json()
print(response)

if response["churn"] == True:
    print("sending promo email to %s" % customer_id)
else:
    print("not sending promo email to %s" % customer_id)

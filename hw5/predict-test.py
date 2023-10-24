#!/usr/bin/env python
# coding: utf-8

import requests

host = "0.0.0.0:9696"
url = f"http://{host}/predict"

customer_id = "xyz-123"
client = {"job": "retired", "duration": 445, "poutcome": "success"}


response = requests.post(url, json=client).json()
print(response)

if response["score"] == True:
    print("client is approved %s" % customer_id)
else:
    print("client is not approved %s" % customer_id)

import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"

data = {"url": "https://t4.ftcdn.net/jpg/00/97/58/97/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg"}

results = requests.post(url, json=data).json()

print(results)
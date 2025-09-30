import requests

prompt = "Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*."

url = "http://localhost:8010/api/tasks/accuracy_per_sample"
params = {
    "task": "ifeval",
    "eval_dir": "eval_results",
    "model_names": "llama3.1",
    "prompt": prompt
}
headers = {
    "accept": "application/json"
}

response = requests.get(url, params=params, headers=headers)

# Print the response
print(response.status_code)
print(response.json())  # or response.text if it's not JSON

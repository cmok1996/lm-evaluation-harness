# import requests

# prompt = "Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*."

# url = "http://localhost:8000/tasks/accuracy_per_sample"
# params = {
#     "task": "ifeval",
#     "output_path": "eval_results/ifeval_eval",
#     "model_names": "llama3.1",
#     "prompt": prompt
# }
# headers = {
#     "accept": "application/json"
# }

# response = requests.get(url, params=params, headers=headers)

# # Print the response
# print(response.status_code)
# print(response.json())  # or response.text if it's not JSON

import requests

# get results to create leaderboard data
url = "http://localhost:8000/tasks/results"
params = {
    "task": "ifeval",
    "output_path": "eval_results/ifeval_eval",
    "model": "llama3.1"
}
headers = {
    "accept": "application/json"
}

response = requests.get(url, headers=headers, params=params)
result = response.json()

# Pass leaderboard data to calculate usecase score
payload = {'leaderboard_data': result,
           'usecase': 'RAG_CHATBOT'}


url = "http://localhost:8000/leaderboard/use_case_score"
response = requests.post(url, headers=headers, json=payload)

print(response.status_code)
print(response.json())  # {'result': [{'model': 'llama3.1', 'weighted_score': 0.19619223659889093, 'task_weights': 0.3, 'usecase_score': 0.6539741219963031}]}
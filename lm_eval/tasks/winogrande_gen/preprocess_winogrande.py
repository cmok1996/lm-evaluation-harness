def doc_to_text(doc):
    sentence = doc['sentence']
    option1 = doc['option1']
    option2 = doc['option2']
    prompt = f"{sentence}\nAnswer with only the result, no explanations needed. Choose ONLY one of the following result:{option1}, {option2}\nAnswer:"
    return prompt


def doc_to_target(doc):
    option = doc['answer'][0]
    if int(option) == 1:
        return doc['option1']
    elif int(option) == 2:
        return doc['option2']


# def doc_to_choice(doc):
#     idx = doc["sentence"].index("_")
#     options = [doc["option1"], doc["option2"]]
#     return [doc["sentence"][:idx] + opt for opt in options]

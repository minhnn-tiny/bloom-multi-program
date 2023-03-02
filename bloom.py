import os
from os.path import join

import json
import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
from transformers import BloomForSequenceClassification

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm


def score(labels, logits):
    logits = nn.functional.softmax(logits, dim=1)
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    buggy_f1 = f1_score(labels, prediction, average=None)[1]
    return accuracy, micro_f1, macro_f1, buggy_f1


if __name__ == '__main__':
    print('Loaded bloom')
    source_dir = '/Users/minh/Documents/2022/smart_contract/mando/original-ge-sc/experiments/ge-sc-data/source_code/java/Neo4j/cfg'
    source_storage = '/Users/minh/Documents/2022/smart_contract/mando/original-ge-sc/experiments/ge-sc-data/source_code/java/Neo4j/cleaned_sources/'
    method_records_file = '/Users/minh/Documents/2022/smart_contract/mando/original-ge-sc/experiments/ge-sc-data/source_code/java/Neo4j/method_locations.json'
    with open(method_records_file, 'r') as f:
        method_records = json.load(f)
    file_names = [f.replace('.dot', '.java') for f in os.listdir(source_dir) if f.endswith('.dot')]
    file_names = file_names
    # model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b7")
    # model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-1b7")
    # tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b7")
    model = BloomForSequenceClassification.from_pretrained("bigscience/mt0-small")
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/mt0-small")
    # model = BloomForSequenceClassification.from_pretrained("bigscience/bloomz-560m")
    # tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloomz-560m")
    # prompt = "It was a dark and stormy night"
    # result_length = 50
    # inputs = tokenizer(prompt, return_tensors="pt")
    # # Greedy Search
    # print(tokenizer.decode(model.generate(inputs["input_ids"], 
    #                    max_length=result_length
    #                   )[0]))
    # # Beam Search
    # print(tokenizer.decode(model.generate(inputs["input_ids"],
    #                     max_length=result_length, 
    #                     num_beams=2, 
    #                     no_repeat_ngram_size=2,
    #                     early_stopping=True
    #                     )[0]))

    # # Sampling Top-k + Top-p
    # print(tokenizer.decode(model.generate(inputs["input_ids"],
    #                     max_length=result_length, 
    #                     do_sample=True, 
    #                     top_k=50, 
    #                     top_p=0.9
    #                     )[0]))

    # print(model.generate(inputs["input_ids"],
    #                     max_length=result_length, 
    #                     num_beams=2, 
    #                     no_repeat_ngram_size=2,
    #                     early_stopping=True
    #                     ))
    
    annotation_file = '/Users/minh/Documents/2022/smart_contract/mando/original-ge-sc/experiments/ge-sc-data/source_code/java/Neo4j/vulnerabilities.json'
    with open(annotation_file, 'r') as f:
        _annotations = json.load(f)
    annotations = {}
    for anno in _annotations:
        if anno['name'] not in annotations:
            annotations[anno['name']] = anno['vulnerabilities']
        else:
            annotations[anno['name']] += anno['vulnerabilities']
    max_length = 512
    targets = {}
    embeded = {}
    logits  = {}
    for source in tqdm(file_names):
        # print('Source name: ', source)
        source_path = join(source_storage, source)
        with open(source_path, 'r') as file_handle:
            source_codes = file_handle.read()
        code_lines = source_codes.split('\n')
        method_list = method_records[source]
        targets[source] = []
        embeded[source] = None
        logits[source] = None
        for method, location in method_list.items():
            # print(method, location)
            loc_range = list(range(location[0][0]+1, location[1][0]+1+1))
            function_code = code_lines[location[0][0]:location[1][0]+1]
            if source in annotations and len(set(loc_range).intersection(annotations[source])) > 0:
                targets[source].append(1)
            else:
                targets[source].append(0)
            inputs = tokenizer(function_code, return_tensors="pt", padding=True, truncation=True)
            output = model(inputs["input_ids"], attention_mask=inputs['attention_mask'], output_hidden_states=True)
            last_hidden = torch.mean(torch.mean(output.hidden_states[-1], dim=0), dim=0).unsqueeze(0)
            embeded[source] = last_hidden if embeded[source] is None else torch.cat((embeded[source],last_hidden))
            logit = torch.mean(output.logits, dim=0).unsqueeze(0)
            logits[source] = logit if logits[source] is None else torch.cat((logits[source],logit))
        print('output: ', embeded[source].shape)
        torch.save(embeded[source], join(source_dir, source.replace('.java', '.pt')))

    # pred = torch.cat(logits.values(), dim=0)
    # targets = torch.tensor([t for t in ])

    truth = []
    pred = None
    for source in targets.keys():
        pred = logits[source] if pred is None else torch.cat((pred, logits[source]))
        truth += targets[source]
    
    truth = torch.tensor(truth, requires_grad=False).cpu().numpy()
    torch.save(truth, join(source_dir, 'gt.gpickle'))
    print('Buggy/Total: {}/{}'.format(np.nonzero(truth)[0].shape[0], truth.shape[0]))
    pred = torch.tensor(pred, requires_grad=False)
    logits = nn.functional.softmax(pred, dim=1)
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    # truth = torch.tensor(truth)
    print(truth)
    print(prediction)
    classification_report(truth, prediction, digits=4)
        

    # print(source_codes.split('\n'))
    # print(inputs)
    # print(output.logits.shape)
    # print(output.hidden_states)
    # print(len(output.hidden_states))
    # print(output.hidden_states[0].shape) 
    # print(output[0].shape)
    # inputs = tokenizer(code, return_tensors="pt")
    # print(tokenizer.decode(model.generate(inputs["input_ids"], 
    #                    max_length=result_length
    #                   )[0]))
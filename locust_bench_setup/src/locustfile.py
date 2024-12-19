import json
import csv
import os
import time
import random
from collections import namedtuple
from os import environ
from typing import List
from threading import Semaphore

from jinja2 import Template
from file_reader import FileProcessorFactory
import pyarrow.parquet as pq
from locust import HttpUser, task, events, tag
from dotenv import load_dotenv, find_dotenv
import logging

# Extra Metrics to collect for LLM statistic
MetricEntry = namedtuple("MetricEntry", ["timestamp", "request_latency", "content"])

records = []
metrics = []
semaphore = Semaphore(1)
metrics_sample = time.time()

_ = load_dotenv(find_dotenv())
if environ.get("SAMPLE_FILE") is None:
    raise Exception("Env variable SAMPLE_FILE is not set")
if environ.get("WAIT_TIME") is None:
    raise Exception("Env variable WAIT_TIME is not set")
if environ.get("API_TYPE") is None:
    raise Exception("Env variable API_TYPE is not set")
if environ.get("MODEL_SERVE_NAME") is None:
    raise Exception("Env variable MODEL_SERVE_NAME is not set")
if environ.get("FILE_EXTENSION") is None:
    raise Exception("Env variable FILE_EXTENSION is not set")

# Folder where input prompt dataset is stored
local_dir = os.path.join(os.getcwd(), "input")

with open("schemas/schemas.json", "r") as f:
    json_template = f.read()

original_template = json_template

# Replace template parts with placeholder values
json_template = json_template.replace("{{model}}", "model_placeholder")
json_template = json_template.replace("{{prompt}}", "prompt_placeholder")
json_template = json_template.replace("{{max_tokens}}", "0")

schemas = json.loads(json_template)

api_type = os.environ.get("API_TYPE")

response_schema = schemas[api_type]["response"]

wait_time = float(os.environ.get("WAIT_TIME"))

headers_json = os.environ.get("HEADERS")

try:
    if headers_json:
        headers_dict = json.loads(headers_json)
    else:
        headers_dict = None
except json.JSONDecodeError as e:
    logging.error(f"Error parsing headers JSON: {e}")
    headers_dict = None


def get_nested_value(obj, path, default=None):
    keys = path.split(".")
    for key in keys:
        if obj is None:
            return default
        if key.endswith("]"):
            key, index = key.split("[")
            index = int(index[:-1])
            if key not in obj or index >= len(obj[key]):
                return default
            obj = obj[key][index]
        else:
            if key not in obj:
                return default
            obj = obj[key]
    return obj


# Collect Response from all requests
# Split between Completion and Metrics endpoints
# Data is stored in CPU Memory to reduce delay from writing to file
@events.request.add_listener
def on_request(
        request_type,
        name,
        response_time,
        response_length,
        response,
        context,
        exception,
        start_time,
        url,
        **kwargs,
):
    #print("on_reques context =", context)
    #print("on_request response_schema = ", response_schema)
    if "stream" in context:
        return
    if not exception is None:
        return
    #print("on_request name = ", name)
    #print("on_request response = ", response)
   # print("________________________________________________")
    #print("on_request type of response.text = ", response.text)
   # print("________________________________________________")
   # print("on_request response.elapsed = ", response.elapsed)
   # print("on_request context[prompt]= ", context["prompt"])
   # print("on_request response_schema = ", response_schema)
   # print("on_request start_time = ", start_time)

    if "metrics" in name:
        metrics.append(MetricEntry(start_time, response.elapsed, response.text))
    else:
        resp = json.loads(response.text)
        response_data = {
            "timestamp": start_time,
            "prompt": context["prompt"],
            "request_latency": response.elapsed,
        }
        for key, value in response_schema.items():
            try:
                response_data[key] = get_nested_value(resp, value)
            except (KeyError, IndexError):
                pass

        response_entry = namedtuple("ResponseEntry", response_data.keys())(
            *response_data.values()
        )
        records.append(response_entry)


# Write to Files regarding the metrics and response result
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    prefix = environment.parsed_options.csv_prefix
    with open(f"{prefix}_records_stats_history.csv", "w", newline="") as fd:
        writer = csv.writer(fd, dialect="excel", quoting=csv.QUOTE_ALL)
        if records:
            writer.writerow(records[0]._fields)
        for r in records:
            writer.writerow(list(r))

    with open(f"{prefix}_metrics.csv", "w", newline="") as fd:
        writer = csv.writer(fd, dialect="excel", quoting=csv.QUOTE_ALL)
        writer.writerow(MetricEntry._fields)
        for r in metrics:
            writer.writerow(list(r))


class PerformanceTest(HttpUser):
    def sample_requests(self, dataset_path: str) -> List[str]:
        file_extension = environ.get("FILE_EXTENSION")
        file_processor = FileProcessorFactory.create_strategy(file_extension)
        # Load the dataset.
        df = file_processor.read_file(f"{local_dir}/{dataset_path}")
        return df["text"].to_numpy().tolist()

    @tag("perf")
    @task
    def send_request(self):

        mode = os.environ.get("MODE", "non-streaming")
        prompt = random.sample(self.sample_prompts, 1)[0]
        max_token = os.environ.get('MAX_TOKEN')
        test_case_type = os.environ.get("TEST_CASE_TYPE")

        if test_case_type == "variable":
            min_token = os.environ.get('MIN_TOKEN')
            max_token = random.randint(int(min_token), int(max_token))

        model = os.environ.get("MODEL_SERVE_NAME")

        if api_type not in schemas:
            raise Exception(f"Unsupported API type: {api_type}")

        endpoint = schemas[api_type]["endpoint"]
        template = Template(original_template)
        rendered_json = template.render(
            model=model, prompt=json.dumps(prompt)[1:-1], max_tokens=max_token
        )
        json_dict = json.loads(rendered_json)[api_type]["body"]

        if mode == "streaming":

            response_data = {
                "timestamp": time.time(),
                "prompt": prompt,
                "request_latency": 0,
                "text": "",
                "time_to_first_token": 0,
                "inter_token_latency": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
            start_time = time.time()
            first_token_received = False
            #logging.info(f"before posting the request to the vllm served model prompt= {prompt} ")
            #logging.info(f"before posting the request to the vllm served model json_dict = {json_dict} ")
            #logging.info(f"before posting the request to the vllm served model headers_dict = {headers_dict} ")
            with self.client.post(endpoint, json=json_dict, context={"prompt": prompt, "stream": True}, headers=headers_dict,
                                  stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data: '):
                            json_string = decoded_line[6:]  # Remove the 'data: ' prefix
                            if json_string != "[DONE]":
                                if not first_token_received:
                                    first_token_received = True
                                    response_data["time_to_first_token"] = time.time() - start_time

                                response_json = json.loads(json_string)

                                for key, value in response_schema.items():
                                    extracted_value = get_nested_value(response_json, value)
                                    if extracted_value is not None:
                                        if key in response_data:
                                            if key == "text":
                                                # Append the value if the key already exists, this is to concat stream text
                                                response_data[key] += extracted_value
                                            else:
                                                # update with latest value
                                                response_data[key] = extracted_value
                                        else:
                                            # Set the value if the key doesn't exist
                                            response_data[key] = extracted_value

            end_time = time.time()
            request_latency = end_time - start_time
            response_data["request_latency"] = request_latency

            response_data["inter_token_latency"] = (request_latency - response_data["time_to_first_token"]) / (
                        response_data["completion_tokens"] - 1)

            response_entry = namedtuple("ResponseEntry", response_data.keys())(*response_data.values())
            records.append(response_entry)

        else:
            self.client.post(
                endpoint, json=json_dict, context={"prompt": prompt}, headers=headers_dict
            )
        time.sleep(wait_time)

    @tag("metrics")
    @task
    def status(self):

        if "metrics_endpoint" not in schemas[api_type]:
            return
        global metrics_sample
        if ((time.time() - metrics_sample) > 5) and semaphore.acquire(blocking=False):
            self.client.get(schemas[api_type]["metrics_endpoint"])
            metrics_sample = time.time()
            semaphore.release()

    def on_start(self):
        self.sample_prompts = self.sample_requests(environ.get("SAMPLE_FILE"))

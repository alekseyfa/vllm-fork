import argparse
import glob
import json
import logging
import os
import subprocess
from urllib.parse import urlparse
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

MODEL_ID = os.getenv("MODEL")

if not MODEL_ID:
    raise ValueError("Please set MODEL env var")

def extract_s3_info(s3_path):
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    folder_path = parsed.path.lstrip("/")
    return bucket, folder_path


def upload_matching_files(file_pattern: str, s3_path: str):
    matching_files = glob.glob(file_pattern)
    for file_name in matching_files:
        logger.info(f"Uploading file {file_name} to s3")
        object_name = file_name.split("/")[-1]  # Extract the filename from the path
        upload_file(file_name, s3_path, object_name)


def upload_file(file_name: str, s3_path: str, object_name=None):
    bucket, folder_path = extract_s3_info(s3_path)
    s3 = boto3.client("s3")
    try:
        s3.upload_file(file_name, bucket, f"{folder_path}/{object_name}")
        logger.info(f"File {object_name} uploaded to s3")
    except ClientError as e:
        logger.error(e)


def download_files(s3_path):
    bucket, folder_path = extract_s3_info(s3_path)
    print(bucket)
    print(folder_path)
    s3 = boto3.client("s3")

    response = s3.list_objects_v2(Bucket=bucket, Prefix=folder_path)

    local_dir = os.path.join(os.getcwd(), "input")
    os.makedirs(local_dir, exist_ok=True)
    if "Contents" not in response:
        raise Exception("No test data found")
    for obj in response["Contents"]:
        file_key = obj["Key"]

        if file_key.startswith(folder_path + "/"):
            file_name = os.path.basename(file_key)
            local_file_path = os.path.join(local_dir, file_name)

            if not os.path.exists(local_file_path):
                s3.download_file(bucket, file_key, local_file_path)
                logger.info(f"File downloaded to {local_file_path}")
            else:
                logger.info(
                    f"File already exists at {local_file_path}, skipping download"
                )
    return local_dir


def list_files(s3_path):
    bucket, folder_path = extract_s3_info(s3_path)
    s3 = boto3.client("s3")

    response = s3.list_objects_v2(Bucket=bucket, Prefix=folder_path)

    file_names = []
    if "Contents" not in response:
        return set()
    for obj in response["Contents"]:
        file_key = obj["Key"]

        if file_key.startswith(folder_path + "/"):
            file_name = os.path.basename(file_key)
            file_names.append(file_name)

    return set(file_names)


def run_test_case(user, input_token, output_token, url, runtime, filename):
    cmd = f"locust --csv {filename} --csv-full-history --headless --tags metrics perf -H {url} -u {user} -r 0.1 -t {runtime}m -s 2m 2>&1 | tee {user}user-{input_token}input-{output_token}output.log"
    subprocess.run(cmd, shell=True)


def test_suite(url, bucket_name, file_extension, mode, backend, node_type):
    model_name = MODEL_ID.replace("/", "--")    
    s3_path = f"{bucket_name}/tokenized-prompt/{model_name}"
    download_files(s3_path)

    logger.info(f"Using url {url}")

    iso_date = datetime.now().isoformat()
    s3_destination = f"{bucket_name}/performance-test/result/{mode}/{model_name}/{backend}/{node_type}/{iso_date}"
    existing_test_result_files = list_files(s3_destination)

    # test dataset filename to use for prompts
    file_name = f"_{model_name}{file_extension}"

    str_test_cases = os.environ.get("TEST_CASES")
    test_cases = json.loads(str_test_cases)

    for test_case in test_cases:

        test_case_type = test_case["type"]
        os.environ["TEST_CASE_TYPE"] = test_case_type

        if test_case_type == "constant":
            tokens = test_case["input_token"]
            input_token = tokens
            output_token = test_case["output_token"]
            sample_file = f"{tokens}{file_name}"
            os.environ["MAX_TOKEN"] = str(test_case["output_token"])

        elif test_case_type == "variable":
            tokens = "variable"
            input_token = test_case["min_token"]
            output_token = test_case["max_token"]
            sample_file = f"variable_{input_token}_{output_token}{file_name}"

            os.environ["MAX_TOKEN"] = str(test_case["max_token"])
            os.environ["MIN_TOKEN"] = str(test_case["min_token"])

        else:
            raise Exception("Unsupported test case")

        run_time = test_case["duration"]

        os.environ["SAMPLE_FILE"] = sample_file
        
        users = test_case["user"]

        for user in users:
            test_result_filename_pattern = f"{user}user-{input_token}input-{output_token}output"
        
            if test_case_type == "variable":
                test_result_filename_pattern += "-variable"

            if f"{test_result_filename_pattern}_stats.csv" in existing_test_result_files:
                logger.info(f"Test result for {user}user-{tokens}token found, skipping")
                continue

            run_test_case(user, input_token, output_token, url, run_time, test_result_filename_pattern)
            filename = f"{test_result_filename_pattern}*.csv"
            upload_matching_files(filename, s3_destination)
    logger.info("Completed all test suites")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizer for Locust Benchmarking")
    parser.add_argument("--url", type=str, required=True, help="Model deployment url")
    parser.add_argument(
        "--bucket_name", type=str, required=True, help="s3 bucket name to save the results, results will be saved to <bucket_name>/result/<mode>/<model_name>/<backend>/<node_type>/<iso_date>"
    )
    parser.add_argument(
        "--file_extension", type=str, required=True, help="file extension"
    )
    parser.add_argument(
        "--mode", type=str, required=True, help="mode of test (e.g. streaming)"
    )
    parser.add_argument(
        "--backend", type=str, required=True, help="backend used for the test (e.g. vLLM)"
    )
    parser.add_argument(
        "--node_type", type=str, required=True, help="node type used for the test (e.g. p4d.24xlarge)"
    )
    args, _ = parser.parse_known_args()
    test_suite(args.url, args.bucket_name, args.file_extension, args.mode, args.backend, args.node_type)

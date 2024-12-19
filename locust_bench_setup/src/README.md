
## Environment vars
#### MODEL
e.g
```
MODEL=meta-llama/Llama-3.1-70B-Instruct
```

 Given MODEL=`Mistral-Large-Instruct-2407`, the prompt dataset should be available at s3://<bucket>/tokenized-prompt/Mistral-Large-Instruct-2407

```
aws s3 ls s3://<bucket>/tokenized-prompt/Mistral-Large-Instruct-2407/
2024-11-07 12:06:23   24553191 1000_Mistral-Large-Instruct-2407.parquet
2024-11-07 12:06:24   33818753 variable_128_1024_Mistral-Large-Instruct-2407.parquet
```

#### MODEL_SERVE_NAME
Model name will be used in the model field in the inference request body

e.g
```
MODEL_SERVE_NAME=llama3.170binstruct
```

#### TEST_CASES

e.g
```
TEST_CASES=[{ "type": "constant", "input_token": 1024, "output_token": 100, "user": [4, 8, 16, 30, 32], "duration": 10 }]
```

#### WAIT_TIME
Delay before the virtual user makes another request

e.g
```
WAIT_TIME=0
```


#### FILE_EXTENSION
Dataset file type, supports .parquet or csv.
e.g
```
.parquet
```

#### BUCKET_NAME
S3 bucket path
e.g
```
BUCKET_NAME=s3://i521394-dev
```
#### BACKEND
Backend engine name, will be used as part of the s3 upload path


e.g

```
BACKEND=vLLM
```

#### NODE_TYPE
Node type name, will be used as part of the s3 upload path


e.g

```
NODE_TYPE=p4d.24xlarge
```

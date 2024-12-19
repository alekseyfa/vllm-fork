
# docker run  -e URL="http://host.docker.internal:62613" --env-file perf/src/.env  -it --rm test /bin/bash -c "./run_perf.sh" --network="host"
# docker run  -e URL="http://host.docker.internal:62613" -e ANOTHER_VAR=another_value  -it test /bin/sh
#
#
#export MODEL="meta-llama/Llama-3.1-70B-Instruct"
#export MODEL_SERVE_NAME="abap-starcoder2-7b"
##export URL="http://localhost:62613"
#export TEST_CASES='[{ "type": "constant", "input_token": 1024, "output_token": 100, "user": [4, 8, 16, 30, 32], "duration": 1 }]'
#export WAIT_TIME="0"
#export FILE_EXTENSION=".parquet"
#export API_TYPE="openai"
#export MODE="streaming"
#
#export BUCKET_NAME="s3://i521394-dev"
#export BACKEND="vLLM"
#export NODE_TYPE="p4d"
#
#
## Given MODEL ID e.g "meta-llama/Llama-3.1-70B-Instruct", the prompt dataset should be available at s3://i521394/tokenized-prompt/meta-llama--Llama-3.1-70B-Instruct
## MODEL_SERVE_NAME





python3 perf.py --url ${URL} --bucket_name ${BUCKET_NAME} --file_extension ${FILE_EXTENSION} --mode ${MODE} --backend ${BACKEND} --node_type ${NODE_TYPE}

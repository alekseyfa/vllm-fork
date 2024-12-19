import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import re
pd.set_option('display.width', 1000)

def parse_filename(filename):
        """Parse the filename to get the number of users, input tokens and output tokens."""
        basename = os.path.basename(filename)
        parts = basename.split('-')
        users = int(parts[0].replace('user', ''))

        output_token = int(parts[2].replace('output_stats.csv', '').replace("variable_stats.csv","").replace("output",""))
        input_token = int(parts[1].replace('input', ''))
        print("uers = |", users, "|")
        print("input_token = |", input_token, "|")
        print("output_token = |", output_token, "|")
        return users, input_token, output_token


def process_csv_files(directory, print_filename=False, md=False):
        """Process all CSV files in the given directory."""
        print("glob = ", glob.glob(os.path.join(directory, '*user*_stats.csv')))
        data = []
        for filename in glob.glob(os.path.join(directory, '*user*_stats.csv')):
           print("filename = " , filename) 
           if "variable" in filename:
               input_token = "variable"
               output_token = "variable"
               basename = os.path.basename(filename)
               parts = basename.split('-')
               users = int(parts[0].replace('user', ''))
           else:
               users, input_token, output_token = parse_filename(filename)
           df = pd.read_csv(filename)
           try:
              row = df[df['Name'] == '/v1/completions'].iloc[0]
           except:
              raise ("endpoint not found")
                                              
           requests_per_s = row['Requests/s']

           print("requests_per_s = ", requests_per_s)
           history_filename = filename.replace('_stats.csv', '_records_stats_history.csv')
           history_df = pd.read_csv(history_filename)

          # Calculate throughput in completion tokens per second for each request
           history_df['throughput_toks_per_s'] = (history_df['completion_tokens']) / history_df['request_latency']
          # Calculate average throughput for the CSV file
           avg_throughput_toks_per_s = history_df['throughput_toks_per_s'].mean() * users

           # Calculate p99, p95, p90, and median for response time
           avg_response_time = history_df['request_latency'].mean()
           p99_response_time = history_df['request_latency'].quantile(0.99)
           p95_response_time = history_df['request_latency'].quantile(0.99)
           p90_response_time = history_df['request_latency'].quantile(0.90)
           median_response_time = history_df['request_latency'].median()

           avg_tft = history_df["time_to_first_token"].mean()
           p99_tft = history_df["time_to_first_token"].quantile(0.99)
           p95_tft = history_df["time_to_first_token"].quantile(0.95)
           p90_tft = history_df["time_to_first_token"].quantile(0.90)
           p50_tft = history_df["time_to_first_token"].median()
           p50_tft2 = history_df["time_to_first_token"].quantile(0.5)

           avg_itl = history_df["inter_token_latency"].mean()
           p99_itl = history_df["inter_token_latency"].quantile(0.99)
           p95_itl = history_df["inter_token_latency"].quantile(0.95)
           p90_itl = history_df["inter_token_latency"].quantile(0.90)
           p50_itl = history_df["inter_token_latency"].median()
           total_requests = len(history_df)

           # Convert timestamps to seconds (assuming the timestamps are in milliseconds)
           start_time = history_df['timestamp'].min()
           end_time = history_df['timestamp'].max()
           time_interval = end_time - start_time       # Total time interval in seconds
           requests_per_s = total_requests / time_interval

           data.append([users, input_token, output_token, requests_per_s, avg_response_time, avg_throughput_toks_per_s, 
                       p99_response_time, p95_response_time, p90_response_time, median_response_time,
                       p99_tft, p95_tft, p90_tft,p50_tft2,avg_tft,
                       p99_itl, p95_itl, p90_itl,p50_itl,avg_itl
                      ])
              
        df = pd.DataFrame(data, columns=['Users', 'Input Token', 'Output Token', 'Request/s', 'Avg Response Time (s)', 'Avg Completion Throughput (toks/s) [Output only]', 'P99 Response Time (s)', 'P95 Response Time (s)', 
                                        'P90 Response Time (s)', 
                                        'Median Response Time (s)', 
                                        'P99 TFT',
                                        'P95 TFT',
                                        'P90 TFT',
                                        'Median TFT',
                                        'Avg TFT',
                                        'P99 ITL',
                                        'P95 ITL',
                                        'P90 ITL',
                                        'Median ITL',
                                        'Avg ITL',
                                        ])

        # Sort the dataframe by 'Users' and 'Input Token'
        df = df.sort_values(by=['Users', 'Input Token', 'Output Token'])
        # Remove index
             
        print("dataframe = ", df)  
        df = df.reset_index(drop=True)
        if md:
           print(df.to_markdown(index=False))

        return df

if __name__ == "__main__":
    process_csv_files(".", print_filename=True, md=True)

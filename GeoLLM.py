<<<<<<< HEAD
import argparse
from utils import *
import os
import json
import numpy as np
import pandas as pd
import math
import re
import time
import requests
import google.generativeai as genai
import folium
from openai import OpenAI
import threading
import csv
MIN_DELAY = 1.25

class TimeoutError(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = []
            exception = []

            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=target)
            thread.daemon = True  
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function timed out after {seconds} seconds")
            if exception:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator
def handler(signum, frame):
    raise TimeoutError("Timeout occurred!")


def write_to_csv(latitudes, longitudes, predictions, file_path):
    df = pd.DataFrame({
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Predictions': predictions
    })
    df.to_csv(file_path, index=False)


def get_rating(completion):

    if not re.search(r"(My answer is|Final answer:)\s*\d+\.?\d*", completion, re.I):
        return None  # 标识未完成回答
    match = re.search(
        r"(?:My answer is|Final answer:)\s*([+-]?\d+\.\d+)",
        completion,
        re.IGNORECASE
    )
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_stage2_response(completion):

    if not completion:
        return None


    clean_text = completion.strip().lower()

    if clean_text.startswith("no"):
        return False


    match = re.search(
        r'my answer is\s*([+-]?\d+\.?\d*)',
        clean_text,
        re.IGNORECASE
    )

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None

# @retry(wait=wait_exponential(multiplier=1, min=2, max=60),
#        stop=stop_after_attempt(5))
@timeout(120)
def get_openai_prediction(api_key, model, prompt):
    # openai.api_key = api_key
    client = OpenAI(base_url='https://api.nuwaapi.com/v1',
                    api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=10,
        temperature=0.0,
        # logprobs=True,
        # top_logprobs=5,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    completion = response.choices[0].message.content

    most_probable = get_rating(completion)

    if most_probable is None:
        return None, None ,None

    # top_logprobs = response.choices[0].logprobs.content[4].top_logprobs
    # valid_items = [item for item in top_logprobs if item.token.isdigit()]
    # total_probability = sum(math.exp(item.logprob) for item in valid_items)
    # expected_value = sum(int(item.token) * (math.exp(item.logprob) / total_probability) for item in valid_items)

    # return completion, most_probable, expected_value
    return completion, most_probable, None


def get_google_prediction(api_key, model, prompt):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model)
    generation_config = genai.types.GenerationConfig(
        candidate_count=1,
        max_output_tokens=10,
        temperature=0
    )
    response = model.generate_content(prompt, generation_config=generation_config)
    completion = response.text
    most_probable = get_rating(completion)

    return completion, most_probable, None


def get_together_prediction(api_key, model, prompt):
    url = "https://api.nuwaapi.com/v1"

    payload = {
        "model": model,
        "max_tokens": 10,
        "temperature": 0,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, json=payload, headers=headers)
    response = json.loads(response.text)

    completion = response['choices'][0]['message']['content']
    most_probable = get_rating(completion)

    return completion, most_probable, None


def plot_on_map(latitudes, longitudes, predicted, file_path):
    coordinates = list(zip(latitudes, longitudes))
    data = normalized_fractional_ranking(predicted)

    m = folium.Map(location=[20, 10], zoom_start=3.25, tiles='CartoDB positron')

    colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=0.0, vmax=1.0)
    colors = [colormap(val) for val in data]

    for coord, color in zip(coordinates, colors):
        folium.CircleMarker(
            location=coord,
            radius=7,
            color='none',
            fill=True,
            fill_color=color,
            fill_opacity=0.75
        ).add_to(m)

    m.add_child(colormap)

    m.save(file_path)


def run_task_for_data(model_api, model, task, prompt_file_path, api_key):
    prompts = load_geollm_prompts(prompt_file_path, task)
    total_points = len(prompts)


    directory = "GeoLLM_results"
    os.makedirs(directory, exist_ok=True)


    model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model)
    task_name = re.sub(r'[^a-zA-Z0-9_]', '_', task)
    prompts_name = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.basename(prompt_file_path).split(".")[0])
    base_file_path = f"{directory}/{model_name}_{task_name}_{prompts_name}"


    main_csv_path = f"{base_file_path}.csv"
    ev_csv_path = f"{base_file_path}_expected_value.csv"


    processed_count = 0
    skipped_count = 0
    error_count = 0
    existing_points = set()


    if os.path.exists(main_csv_path):
        try:
            existing_df = pd.read_csv(main_csv_path)
            existing_points = set(
                f"{row['Latitude']:.7f}_{row['Longitude']:.7f}"
                for _, row in existing_df.iterrows()
            )

            skipped_count = sum(
                1 for p in prompts
                if f"{get_coordinates(p)[0]:.7f}_{get_coordinates(p)[1]:.7f}" in existing_points
            )
        except Exception as e:
            print(f"local file loading error: {str(e)}")


    for path, cols in [(main_csv_path, ['Latitude', 'Longitude', 'Predictions']),
                       (ev_csv_path, ['Latitude', 'Longitude', 'Predictions'])]:
        if not os.path.exists(path):
            pd.DataFrame(columns=cols).to_csv(path, index=False)


    for idx, prompt in enumerate(prompts, 1):
        lat, lon = get_coordinates(prompt)
        coord_key = f"{lat:.7f}_{lon:.7f}"

        try:
            # ========== 跳过检查 ==========
            if coord_key in existing_points:
                skipped_count += 1
                continue

            # ========== API调用 ==========
            start_time = time.time()

            if model_api == "openai":
                completion, most_probable, expected_value = get_openai_prediction(api_key, model, prompt)
            elif model_api == "google":
                completion, most_probable, expected_value = get_google_prediction(api_key, model, prompt)
            else:
                completion, most_probable, expected_value = get_together_prediction(api_key, model, prompt)


            print(f"\nPROMPT {idx}:\n{prompt}\nCOMPLETION: {completion}\n")

            if most_probable is None:
                error_count += 1
                print(f"❌ error（error count: {error_count}）")
                continue

   
            pd.DataFrame([[lat, lon, most_probable]],
                         columns=['Latitude', 'Longitude', 'Predictions']
                         ).to_csv(main_csv_path, mode='a', header=False, index=False)


            if expected_value is not None:
                pd.DataFrame([[lat, lon, expected_value]],
                             columns=['Latitude', 'Longitude', 'Predictions']
                             ).to_csv(ev_csv_path, mode='a', header=False, index=False)
                print(f"RATING: {most_probable} | EXPECTED VALUE: {expected_value}")
            else:
                print(f"RATING: {most_probable}")

         
            processed_count += 1
            existing_points.add(coord_key)
            print(f"✅ finished: {processed_count}/{total_points} | to be processed: {total_points - idx}")

            
            elapsed = time.time() - start_time
            delay = max(MIN_DELAY - elapsed, 0)
            time.sleep(delay)

        except TimeoutError as e:
            error_count += 1
            print(f"⏰ API timeout: {str(e)}（error_count: {error_count}）")
        except Exception as e:
            error_count += 1
            print(f"❌ error: {str(e)}（error_count: {error_count}）")

    # ========== 生成可视化 ==========
    for path, suffix in [(main_csv_path, ''), (ev_csv_path, '_expected_value')]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                plot_on_map(
                    df['Latitude'].tolist(),
                    df['Longitude'].tolist(),
                    df.iloc[:, 2].tolist(),  
                    f"{base_file_path}{suffix}.html"
                )
            except Exception as e:
                print(f"⚠️ error({path}): {str(e)}")


    print(f"\n{'=' * 40}")
    print(f"GeoLLM task finished:")
    print(f"GeoLLM results file: {main_csv_path}")
    print(f"points: {total_points}")
    print(f"{'=' * 40}")


def find_nearest_neighbors(df, target_lat, target_lon, n=10):
    """寻找最近的n个邻居点"""
    # 使用欧氏距离简化计算（小范围适用）
    df['distance'] = np.sqrt(
        (df['Latitude'] - target_lat) ** 2 +
        (df['Longitude'] - target_lon) ** 2
    )
    return df.nsmallest(n, 'distance')





def main():
    parser = argparse.ArgumentParser(description='GeoLLM预测流程', add_help=False)


    parser.add_argument('model_api', type=str)
    parser.add_argument('api_key', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('prompts_file', type=str, nargs='?') 
    parser.add_argument('task', type=str, nargs='?')  
    args = parser.parse_args()
    run_task_for_data(args.model_api, args.model, args.task, args.prompts_file, args.api_key)
    

if __name__ == "__main__":
    main()
=======
import argparse
from utils import *
import os
import json
import numpy as np
import pandas as pd
import math
import re
import time
import requests
import google.generativeai as genai
import folium
from openai import OpenAI
import threading
import csv
MIN_DELAY = 1.25

class TimeoutError(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = []
            exception = []

            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=target)
            thread.daemon = True  
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function timed out after {seconds} seconds")
            if exception:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator
def handler(signum, frame):
    raise TimeoutError("Timeout occurred!")


def write_to_csv(latitudes, longitudes, predictions, file_path):
    df = pd.DataFrame({
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Predictions': predictions
    })
    df.to_csv(file_path, index=False)


def get_rating(completion):

    if not re.search(r"(My answer is|Final answer:)\s*\d+\.?\d*", completion, re.I):
        return None  # 标识未完成回答
    match = re.search(
        r"(?:My answer is|Final answer:)\s*([+-]?\d+\.\d+)",
        completion,
        re.IGNORECASE
    )
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_stage2_response(completion):

    if not completion:
        return None


    clean_text = completion.strip().lower()

    if clean_text.startswith("no"):
        return False


    match = re.search(
        r'my answer is\s*([+-]?\d+\.?\d*)',
        clean_text,
        re.IGNORECASE
    )

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None

# @retry(wait=wait_exponential(multiplier=1, min=2, max=60),
#        stop=stop_after_attempt(5))
@timeout(120)
def get_openai_prediction(api_key, model, prompt):
    # openai.api_key = api_key
    client = OpenAI(base_url='https://api.nuwaapi.com/v1',
                    api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=10,
        temperature=0.0,
        # logprobs=True,
        # top_logprobs=5,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    completion = response.choices[0].message.content

    most_probable = get_rating(completion)

    if most_probable is None:
        return None, None ,None

    # top_logprobs = response.choices[0].logprobs.content[4].top_logprobs
    # valid_items = [item for item in top_logprobs if item.token.isdigit()]
    # total_probability = sum(math.exp(item.logprob) for item in valid_items)
    # expected_value = sum(int(item.token) * (math.exp(item.logprob) / total_probability) for item in valid_items)

    # return completion, most_probable, expected_value
    return completion, most_probable, None


def get_google_prediction(api_key, model, prompt):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model)
    generation_config = genai.types.GenerationConfig(
        candidate_count=1,
        max_output_tokens=10,
        temperature=0
    )
    response = model.generate_content(prompt, generation_config=generation_config)
    completion = response.text
    most_probable = get_rating(completion)

    return completion, most_probable, None


def get_together_prediction(api_key, model, prompt):
    url = "https://api.nuwaapi.com/v1"

    payload = {
        "model": model,
        "max_tokens": 10,
        "temperature": 0,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, json=payload, headers=headers)
    response = json.loads(response.text)

    completion = response['choices'][0]['message']['content']
    most_probable = get_rating(completion)

    return completion, most_probable, None


def plot_on_map(latitudes, longitudes, predicted, file_path):
    coordinates = list(zip(latitudes, longitudes))
    data = normalized_fractional_ranking(predicted)

    m = folium.Map(location=[20, 10], zoom_start=3.25, tiles='CartoDB positron')

    colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=0.0, vmax=1.0)
    colors = [colormap(val) for val in data]

    for coord, color in zip(coordinates, colors):
        folium.CircleMarker(
            location=coord,
            radius=7,
            color='none',
            fill=True,
            fill_color=color,
            fill_opacity=0.75
        ).add_to(m)

    m.add_child(colormap)

    m.save(file_path)


def run_task_for_data(model_api, model, task, prompt_file_path, api_key):
    prompts = load_geollm_prompts(prompt_file_path, task)
    total_points = len(prompts)


    directory = "GeoLLM_results"
    os.makedirs(directory, exist_ok=True)


    model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model)
    task_name = re.sub(r'[^a-zA-Z0-9_]', '_', task)
    prompts_name = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.basename(prompt_file_path).split(".")[0])
    base_file_path = f"{directory}/{model_name}_{task_name}_{prompts_name}"


    main_csv_path = f"{base_file_path}.csv"
    ev_csv_path = f"{base_file_path}_expected_value.csv"


    processed_count = 0
    skipped_count = 0
    error_count = 0
    existing_points = set()


    if os.path.exists(main_csv_path):
        try:
            existing_df = pd.read_csv(main_csv_path)
            existing_points = set(
                f"{row['Latitude']:.7f}_{row['Longitude']:.7f}"
                for _, row in existing_df.iterrows()
            )

            skipped_count = sum(
                1 for p in prompts
                if f"{get_coordinates(p)[0]:.7f}_{get_coordinates(p)[1]:.7f}" in existing_points
            )
        except Exception as e:
            print(f"local file loading error: {str(e)}")


    for path, cols in [(main_csv_path, ['Latitude', 'Longitude', 'Predictions']),
                       (ev_csv_path, ['Latitude', 'Longitude', 'Predictions'])]:
        if not os.path.exists(path):
            pd.DataFrame(columns=cols).to_csv(path, index=False)


    for idx, prompt in enumerate(prompts, 1):
        lat, lon = get_coordinates(prompt)
        coord_key = f"{lat:.7f}_{lon:.7f}"

        try:
            # ========== 跳过检查 ==========
            if coord_key in existing_points:
                skipped_count += 1
                continue

            # ========== API调用 ==========
            start_time = time.time()

            if model_api == "openai":
                completion, most_probable, expected_value = get_openai_prediction(api_key, model, prompt)
            elif model_api == "google":
                completion, most_probable, expected_value = get_google_prediction(api_key, model, prompt)
            else:
                completion, most_probable, expected_value = get_together_prediction(api_key, model, prompt)


            print(f"\nPROMPT {idx}:\n{prompt}\nCOMPLETION: {completion}\n")

            if most_probable is None:
                error_count += 1
                print(f"❌ error（error count: {error_count}）")
                continue

   
            pd.DataFrame([[lat, lon, most_probable]],
                         columns=['Latitude', 'Longitude', 'Predictions']
                         ).to_csv(main_csv_path, mode='a', header=False, index=False)


            if expected_value is not None:
                pd.DataFrame([[lat, lon, expected_value]],
                             columns=['Latitude', 'Longitude', 'Predictions']
                             ).to_csv(ev_csv_path, mode='a', header=False, index=False)
                print(f"RATING: {most_probable} | EXPECTED VALUE: {expected_value}")
            else:
                print(f"RATING: {most_probable}")

         
            processed_count += 1
            existing_points.add(coord_key)
            print(f"✅ finished: {processed_count}/{total_points} | to be processed: {total_points - idx}")

            
            elapsed = time.time() - start_time
            delay = max(MIN_DELAY - elapsed, 0)
            time.sleep(delay)

        except TimeoutError as e:
            error_count += 1
            print(f"⏰ API timeout: {str(e)}（error_count: {error_count}）")
        except Exception as e:
            error_count += 1
            print(f"❌ error: {str(e)}（error_count: {error_count}）")

    # ========== 生成可视化 ==========
    for path, suffix in [(main_csv_path, ''), (ev_csv_path, '_expected_value')]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                plot_on_map(
                    df['Latitude'].tolist(),
                    df['Longitude'].tolist(),
                    df.iloc[:, 2].tolist(),  
                    f"{base_file_path}{suffix}.html"
                )
            except Exception as e:
                print(f"⚠️ error({path}): {str(e)}")


    print(f"\n{'=' * 40}")
    print(f"GeoLLM task finished:")
    print(f"GeoLLM results file: {main_csv_path}")
    print(f"points: {total_points}")
    print(f"{'=' * 40}")


def find_nearest_neighbors(df, target_lat, target_lon, n=10):
    """寻找最近的n个邻居点"""
    # 使用欧氏距离简化计算（小范围适用）
    df['distance'] = np.sqrt(
        (df['Latitude'] - target_lat) ** 2 +
        (df['Longitude'] - target_lon) ** 2
    )
    return df.nsmallest(n, 'distance')





def main():
    parser = argparse.ArgumentParser(description='GeoLLM预测流程', add_help=False)


    parser.add_argument('model_api', type=str)
    parser.add_argument('api_key', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('prompts_file', type=str, nargs='?') 
    parser.add_argument('task', type=str, nargs='?')  
    args = parser.parse_args()
    run_task_for_data(args.model_api, args.model, args.task, args.prompts_file, args.api_key)
    

if __name__ == "__main__":
    main()
>>>>>>> dcc6fdf (GeoLLM)

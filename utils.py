import rasterio
import jsonlines
import math
import hashlib
import os
import pandas as pd
import numpy as np
import openai
from scipy.spatial.distance import cdist
import json
PREFIX = """You will be given data about a specific location randomly sampled from all human-populated locations on Earth.
You give your rating keeping in mind that it is relative to all other human-populated locations on Earth (from all continents, countries, etc.).
You provide ONLY your answer in the exact format "My answer is X.X." where 'X.X' represents your rating for the given topic.


"""

ADJACENT_PIXELS = 12


def load_or_generate_embeddings(jsonl_path, cache_path="world_embeddings.npy"):
    """加载或生成embedding缓存"""
    if os.path.exists(cache_path):
        return np.load(cache_path, allow_pickle=True).item()

    embeddings = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']
            emb = get_embedding(text)
            coords = text.split('\n')[0].split(': ')[1].strip()
            embeddings[coords] = emb

    np.save(cache_path, embeddings)
    return embeddings
def get_embedding(text, model="text-embedding-3-large"):
    response = openai.Embedding.create(
        input=[text],
        model=model
    )
    return response['data'][0]['embedding']
def haversine_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])


    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  
    return c * r



def get_distance_matrix_id(prompts):

    content = "".join([str(get_coordinates(p)) for p in prompts])
    return hashlib.md5(content.encode()).hexdigest()[:8]



def calculate_distance_matrix(prompts, file_path):

    if os.path.exists(file_path):
        print(f"距离矩阵已存在: {file_path}")
        return

    print("计算距离矩阵...")
    total = len(prompts)
    distance_matrix = []


    coordinates = [get_coordinates(p) for p in prompts]


    for i in range(total):
        lat1, lon1 = coordinates[i]
        row = []
        for j in range(total):
            if i == j:
                row.append(0.0)  
            else:
                lat2, lon2 = coordinates[j]
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                row.append(dist)
        distance_matrix.append(row)


    df = pd.DataFrame(distance_matrix)
    df.to_csv(file_path, index=False)
    print(f"distance matrix saved to {file_path}")



def get_nearest_points_context(idx, known_points, distance_matrix):

    if not known_points:
        return ""

    distances = distance_matrix.iloc[idx].values

    nearest_indices = np.argsort(distances)[1:6] 
    pre_lat, pre_lon, pre_pred = known_points[idx]
    context_lines = [f"your previous prediction is{pre_pred},and some nearest known points prediction(if they exist,they will be shown as fellows)："]
    for i, near_idx in enumerate(nearest_indices, 1):

        if near_idx in known_points:
            lat, lon, pred = known_points[near_idx]
            dist = distances[near_idx]
            context_lines.append(f"{i}. coordinate({lat:.4f}, {lon:.4f}) distance {dist:.2f}km -> your prediction: {pred}")

    if len(context_lines) > 1:
        return "\n".join(context_lines) + "\nPlease decide whether you need to update your answer\n"
    return ""


def build_prompt_with_nearest_predictions(prompt, nearest_df):

    neighbors_info = "\n".join([
        f"│ Neighbor {i + 1}: Latitude {row['Latitude']:.4f}, Longitude {row['Longitude']:.4f}, Prediction {row['Predictions']:.2f} │"
        for i, (_, row) in enumerate(nearest_df.iterrows())
    ])
    return f"{prompt}\nNearest Predictions:\n{neighbors_info}"


def load_geollm_prompts(file_path, task):
    with jsonlines.open(file_path, 'r') as reader:
        data = list(reader)
    geollm_prompts = [PREFIX + item['text'].strip().replace("<TASK>", task) for item in data]
    return geollm_prompts

def get_coordinates(text):
    text = text.split("Coordinates: ")[1]
    coordinates = text[text.find('(')+1:text.find(')')].split(", ")
    lat, lon = list(map(float, coordinates))
    return lat, lon

def normalized_fractional_ranking(numbers):

    sorted_numbers = sorted(enumerate(numbers), key=lambda x: x[1])

    ranks = {}
    for rank, (original_index, number) in enumerate(sorted_numbers):
        if number in ranks:
            ranks[number][0] += rank + 1
            ranks[number][1] += 1
        else:
            ranks[number] = [rank + 1, 1]

    average_ranks = {number: total_rank / count for number, (total_rank, count) in ranks.items()}

    return [(average_ranks[number] - 1) / len(numbers) for number in numbers]


def find_nearest_points(target_emb, embeddings, k=10):

    all_embs = np.array(list(embeddings.values()))
    coords = list(embeddings.keys())


    similarities = 1 - cdist([target_emb], all_embs, 'cosine')[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [coords[i] for i in top_indices]
def extract_data(lat, lon, file_path):
    with rasterio.open(file_path) as src:
        transform = ~src.transform
        x, y = transform * (lon, lat)
        px, py = round(x), round(y)

        window = ((max(py-ADJACENT_PIXELS, 0), min(py+ADJACENT_PIXELS+1, src.height)), (max(px-ADJACENT_PIXELS, 0), min(px+ADJACENT_PIXELS+1, src.width)))
        data = src.read(1, window=window)
        non_negative_data = data[data >= 0]
        total_population = non_negative_data.sum()

        return total_population if 0 <= px < src.width and 0 <= py < src.height else None

import re
import numpy as np
import pandas as pd
import os
import glob
from openai import OpenAI
from tif_utils import read_tif_value


class PointSelector:

    BIO_VARIABLES = {
        "BIO1": "Annual Mean Temperature",
        "BIO2": "Mean Diurnal Range (Mean of monthly (max temp - min temp))",
        "BIO3": "Isothermality (BIO2/BIO7) (100)",
        "BIO4": "Temperature Seasonality (standard deviation 100)",
        "BIO5": "Max Temperature of Warmest Month",
        "BIO6": "Min Temperature of Coldest Month",
        "BIO7": "Temperature Annual Range (BIO5-BIO6)",
        "BIO8": "Mean Temperature of Wettest Quarter",
        "BIO9": "Mean Temperature of Driest Quarter",
        "BIO10": "Mean Temperature of Warmest Quarter",
        "BIO11": "Mean Temperature of Coldest Quarter",
        "BIO12": "Annual Precipitation",
        "BIO13": "Precipitation of Wettest Month",
        "BIO14": "Precipitation of Driest Month",
        "BIO15": "Precipitation Seasonality (Coefficient of Variation)",
        "BIO16": "Precipitation of Wettest Quarter",
        "BIO17": "Precipitation of Driest Quarter",
        "BIO18": "Precipitation of Warmest Quarter",
        "BIO19": "Precipitation of Coldest Quarter"
    }

    def __init__(self, api_key, model,auxiliary_dir=None):
        self.client = OpenAI(base_url='https://api.nuwaapi.com/v1', api_key=api_key)
        self.last_response = None
        self.range_response = None
        self.auxiliary_dir = auxiliary_dir
        self.auxiliary_files = self._load_auxiliary_files()
        self.auxiliary_cache = {}
        self.model = model

    def _load_auxiliary_files(self):
        """Load auxiliary variable files"""
        if not self.auxiliary_dir:
            return {}
        print(f"🔍 Scanning directory: {self.auxiliary_dir}")
        files = glob.glob(os.path.join(self.auxiliary_dir, "*.tif"))
        print(f"Found {len(files)} TIFF files")
        file_map = {}
        for file in files:
            print(f" - {os.path.basename(file)}")
            filename = os.path.basename(file)
            match = re.search(r'wc2\.1_30s_bio_(\d{1,2})\.tif$', filename, re.IGNORECASE)
            if match:
                var_name = f"BIO{match.group(1)}"
                file_map[var_name] = file
        return file_map

    def _get_auxiliary_value(self, lat, lon, variable):
        """Get auxiliary variable value at a point"""
        if variable not in self.auxiliary_files:
            return None

        try:
            return read_tif_value(
                self.auxiliary_files[variable],
                lat,
                lon
            )
        except Exception as e:
            print(f"Failed to read bioclimatic variable: {variable} @ {lat},{lon} - {str(e)}")
            return None

    def select_points(self, target_row, all_data, task, selected_vars=None):
        """
        Select 15 reference points:
        - 10 nearest neighbors (global search)
        - 5 model-selected points (from 5x5 degree area, excluding nearest neighbors)
        """
        # Get target coordinates
        lat, lon = target_row['Latitude'], target_row['Longitude']


        nearest_points_df = self._get_nearest_points(target_row, all_data, n=10)
        nearest_indices = nearest_points_df.index.tolist()

        range_df = all_data[
            (all_data['Latitude'] >= lat - 5) &
            (all_data['Latitude'] <= lat + 5) &
            (all_data['Longitude'] >= lon - 5) &
            (all_data['Longitude'] <= lon + 5)
            ]

        candidate_set = range_df[
            (range_df.index != target_row.name) &
            (~range_df.index.isin(nearest_indices))
            ]

        model_indices = []
        if not candidate_set.empty:

            model_indices = self._select_model_points(
                target_row, candidate_set, task, max_points=5, selected_vars=selected_vars
            )
        selected_indices = nearest_indices + model_indices
        return selected_indices

    def _select_model_points(self, target, candidate_set, task, max_points=5, selected_vars=None):

        candidate_map = {i + 1: idx for i, idx in enumerate(candidate_set.index)}
        selection_prompt = self._build_selection_prompt(
            target, candidate_set, candidate_map, task, max_points, selected_vars
        )

        response = self._query_gpt(selection_prompt)
        self.last_response = response
        return self._parse_response(response, candidate_map)

    def _parse_response(self, text, candidate_map):
        """Parse model response (using order number mapping)"""
        match = re.search(r"Selected order numbers?:\s*\[?([\d,\s]+)\]?", text, re.IGNORECASE)
        if not match:
            print(f"⚠️ Failed to parse response: {text}")
            return []

        try:
            numbers_str = match.group(1).strip()
            number_list = [num.strip() for num in re.split(r'[,\s]+', numbers_str) if num.strip()]
            selected_numbers = []
            for num in number_list:
                if num:
                    try:
                        selected_numbers.append(int(num))
                    except ValueError:
                        print(f"⚠️ Invalid integer: '{num}'")

            valid_numbers = [num for num in selected_numbers if num in candidate_map]

            if not valid_numbers:
                print(f"❌ No valid selected numbers in candidate map. Candidate keys: {list(candidate_map.keys())}")
                return []

            original_indices = [candidate_map[num] for num in valid_numbers]
            return original_indices

        except Exception as e:
            print(f"Parsing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _build_selection_prompt(self, target, candidate_set, candidate_map, task, max_points, selected_vars=None):
        """Build selection prompt for model"""
        samples = []
        for order_num, original_idx in candidate_map.items():
            row = candidate_set.loc[original_idx]
            sample_line = (
                f"{order_num}. Coordinates: ({row['Latitude']:.4f}, {row['Longitude']:.4f}) "
                f"Prediction: {row['Predictions']:.2f}"
            )
            if selected_vars:
                var_values = []
                for var in selected_vars:
                    value = self._get_auxiliary_value(row['Latitude'], row['Longitude'], var)
                    var_values.append(f"{var}={value:.1f}" if value is not None else f"{var}=N/A")
                sample_line += " | " + ", ".join(var_values)

            samples.append(sample_line)

        samples_str = "\n".join(samples)
        out = f"""
As a geospatial selection expert, select up to {max_points} reference points from the 5x5 degree area around the target:

Target Point:
- Coordinates: ({target['Latitude']:.4f}, {target['Longitude']:.4f})
- Prediction: {target['Predictions']:.2f}
- Task: {task}

Candidate Points (Total {len(candidate_map)} points in 5x5 degree area):
{samples_str}

Selection criteria:
1. Select points that are MOST REVEALING for spatial patterns
2. Prioritize points with significant prediction differences
3. Ensure diversity (avoid clustering)

IMPORTANT OUTPUT RULES:
1. YOU MUST output EXACTLY one line starting with "Selected order numbers:"
2. Format: Selected order numbers: [comma-separated numbers] 
   Example: Selected order numbers: 3,7,12,15,22
3. DO NOT include any additional text, explanations, or formatting symbols
"""
        return out

    def _query_gpt(self, prompt):
        """Query GPT API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

    def get_last_response(self):
        return self.last_response

    def _get_nearest_points(self, target, df, n=10):
        """Get n nearest points (global search)"""
        distances = np.sqrt(
            (df['Latitude'] - target['Latitude']) ** 2 +
            (df['Longitude'] - target['Longitude']) ** 2
        )

        candidate_df = df[df.index != target.name].copy()

        if candidate_df.empty:
            print("⚠️ No available candidate points")
            return pd.DataFrame()

        nearest_indices = distances[distances.index.isin(candidate_df.index)].nsmallest(n).index
        return candidate_df.loc[nearest_indices]



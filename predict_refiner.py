
import re
import os
import glob
from openai import OpenAI
from tif_utils import read_tif_value  # 添加导入


class PredictionRefiner:

    def __init__(self, api_key, model,auxiliary_dir=None):
        self.client = OpenAI(base_url='https://api.nuwaapi.com/v1', api_key=api_key)
        self.last_response = None
        self.auxiliary_dir = auxiliary_dir
        self.auxiliary_files = self._load_auxiliary_files()  
        self.model = model

    def _load_auxiliary_files(self):
        if not self.auxiliary_dir:
            return {}
        files = glob.glob(os.path.join(self.auxiliary_dir, "*.tif"))
        file_map = {}
        for file in files:
            filename = os.path.basename(file)
            match = re.search(r'wc2\.1_30s_bio_(\d{1,2})\.tif$', filename, re.IGNORECASE)
            if match:
                var_name = f"BIO{match.group(1)}"
                file_map[var_name] = file
        return file_map

    def _get_auxiliary_value(self, lat, lon, variable):
        if variable not in self.auxiliary_files:
            return None
        try:
            return read_tif_value(
                self.auxiliary_files[variable],
                lat,
                lon
            )
        except Exception as e:
            print(f"bioclim_file read error: {variable} @ {lat},{lon} - {str(e)}")
            return None

    def refine_prediction(self, target, references, task, selected_vars=None):
        print("=======================================================")
        prompt = self._build_refinement_prompt(target, references, task, selected_vars)
        response = self._query_gpt(prompt)
        print(f"{response}")
        self.last_response = response
        return self._parse_response(response)

    def get_last_response(self):
        return self.last_response

    def _build_refinement_prompt(self, target, references, task, selected_vars=None):
        ref_lines = []
        for _, row in references.iterrows():
            line = f"- ({row['Latitude']:.4f}, {row['Longitude']:.4f}): {task} prediction={row['Predictions']:.2f}"
            if selected_vars:
                bio_values = []
                for var in selected_vars:
                    value = self._get_auxiliary_value(row['Latitude'], row['Longitude'], var)
                    bio_values.append(f"{var}={value:.1f}" if value is not None else f"{var}=N/A")
                line += " | " + ", ".join(bio_values)
            ref_lines.append(line)
        ref_lines_str = "\n".join(ref_lines)
        target_bio = ""
        if selected_vars:
            target_values = []
            for var in selected_vars:
                value = self._get_auxiliary_value(target['Latitude'], target['Longitude'], var)
                target_values.append(f"{var}={value:.1f}" if value is not None else f"{var}=N/A")
            target_bio = " | " + ", ".join(target_values)

        return f"""
Some information you need to know:
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
As a geospatial prediction expert, refine the initial rating based on spatial patterns and bioclimatic context:

Task: {task}
Target Location: ({target['Latitude']:.4f}, {target['Longitude']:.4f})
Initial Prediction: {target['Predictions']:.2f}{target_bio}

Nearby Points (with bioclimatic variables):
{ref_lines_str}

Instructions:
1. Analyze regional bioclimatic patterns
2. Compare target and reference points' environmental context
3. Propose scientifically justified adjustment

STRICT OUTPUT RULES:
1. MUST output exactly one line starting with "Final Prediction:"
2. Format: Final Prediction: [X.XX] (X.XX must be float with 2 decimals between 0.00 to 10.00)
3. NO additional text, explanations, or formatting symbols
4. You don't have to refine the prediction if you are not confident in your answer
"""

    def _query_gpt(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        print(f"{prompt}")
        content = response.choices[0].message.content
        return content

    def _parse_response(self, text):
        match = re.search(r"Final Prediction:\s*(\d+\.\d{2})", text)
        if match:
            return float(match.group(1))
        else:
            print(f"⚠can't parse response: {text}")
            return None

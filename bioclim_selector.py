import re
from openai import OpenAI
class AuxVariableSelector:
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

    def __init__(self, model,api_key):
        self.client = OpenAI(base_url='https://api.nuwaapi.com/v1', api_key=api_key)
        self.last_response = None
        self.model = model
    def select_variables(self, target, task):
        prompt = self._build_selection_prompt(target, task)
        response = self._query_gpt(prompt)
        self.last_response = response
        return self._parse_response(response)

    def _build_selection_prompt(self, target, task):
        variables_list = "\n".join(
            [f"- {var}: {desc}" for var, desc in self.BIO_VARIABLES.items()]
        )
        out = f"""
As a geospatial climate expert, select the MOST RELEVANT bioclimatic variables for refining predictions at the target location:

Target Location:
- Coordinates: ({target['Latitude']:.4f}, {target['Longitude']:.4f})
- Task: {task}

Available Bioclimatic Variables:
{variables_list}

Selection Guidelines:
1. Consider the target's geographic context (e.g., coastal, mountainous, tropical)
2. Prioritize variables that explain spatial patterns for {task}
3. Select 1-5 variables that provide complementary and useful information for the task
4. Justify each selection based on climatic relevance

STRICT OUTPUT RULES:
1. MUST output exactly one line starting with "Selected Variables:"
2. Format: Selected Variables: [comma-separated BIO codes]
3. NO additional text, explanations, or formatting symbols
"""
        return out

    def _query_gpt(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content

    def _parse_response(self, text):
        match = re.search(r"Selected Variables:\s*([A-Z0-9,\s]+)", text, re.IGNORECASE)
        if not match:
            print(f"⚠️ Failed to parse variable selection: {text}")
            return ["BIO1"]  

        try:
            selected_vars = [var.strip() for var in match.group(1).split(",")]
            valid_vars = [var for var in selected_vars if var in self.BIO_VARIABLES]

            if not valid_vars:
                print("⚠️ No valid variables selected, using default")
                return ["BIO1"]

            return valid_vars
        except Exception as e:
            print(f"Parse error: {str(e)}")
            return ["BIO1"]

    def get_last_response(self):
        return self.last_response

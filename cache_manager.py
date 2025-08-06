# cache_manager.py
import os
import pandas as pd
from datetime import datetime
import re


class RefinementCache:


    def __init__(self, task_name, output_dir="agent_results"):

        sanitized_name = re.sub(r'[\\/*?:"<>|]', "_", task_name)
        self.cache_path = os.path.join(output_dir, f"{sanitized_name}_refinement_cache.csv")
        self.cache_df = self._load_cache()
        print(f"cache output_dir: {os.path.abspath(self.cache_path)}")

    def get_latest_prediction_for_point(self, point_id):

        point_records = self.cache_df[self.cache_df['point_id'] == point_id]

        if point_records.empty:
            return None

        latest_record = point_records.sort_values('stage', ascending=False).iloc[0]
        return latest_record['refined_prediction']

    def _create_new_cache(self):

        return pd.DataFrame({
            'point_id': pd.Series(dtype='int'),
            'stage': pd.Series(dtype='int'),
            'status': pd.Series(dtype='str'),
            'selected_vars': pd.Series(dtype='str'),
            'refined_prediction': pd.Series(dtype='float'),
            'timestamp': pd.Series(dtype='str')
        })

    def _load_cache(self):

        if os.path.exists(self.cache_path):
            try:

                dtypes = {
                    'point_id': 'int',
                    'stage': 'int',
                    'status': 'str',
                    'selected_vars': 'str',
                    'refined_prediction': 'float',
                    'timestamp': 'str'
                }
                return pd.read_csv(self.cache_path).astype(dtypes)
            except Exception:
                return self._create_new_cache()
        return self._create_new_cache()

    def save_cache(self):
        print(f"cache saved to: {os.path.abspath(self.cache_path)}")
        self.cache_df.to_csv(self.cache_path, index=False)

    def is_point_completed(self, point_id, stage):
        mask = (self.cache_df['point_id'] == point_id) & (self.cache_df['stage'] == stage)
        if not self.cache_df[mask].empty:
            return self.cache_df.loc[mask, 'status'].iloc[0] == 'completed'
        return False

    def get_point_result(self, point_id, stage):
        mask = (self.cache_df['point_id'] == point_id) & (self.cache_df['stage'] == stage)
        if not self.cache_df[mask].empty:
            row = self.cache_df.loc[mask].iloc[0]
            return {
                'selected_vars': row['selected_vars'].split(','),
                'refined_prediction': row['refined_prediction']
            }
        return None

    def update_point_status(self, point_id, stage, status, selected_vars=None, refined_prediction=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mask = (self.cache_df['point_id'] == point_id) & (self.cache_df['stage'] == stage)
        if not self.cache_df[mask].empty:
            idx = self.cache_df.index[mask][0]
            self.cache_df.at[idx, 'status'] = status
            self.cache_df.at[idx, 'timestamp'] = timestamp
            if selected_vars is not None:
                self.cache_df.at[idx, 'selected_vars'] = ",".join(selected_vars)
            if refined_prediction is not None:
                self.cache_df.at[idx, 'refined_prediction'] = refined_prediction
        else:
            new_row = {
                'point_id': point_id,
                'stage': stage,
                'status': status,
                'selected_vars': ",".join(selected_vars) if selected_vars else "",
                'refined_prediction': refined_prediction,
                'timestamp': timestamp
            }
            self.cache_df = pd.concat([self.cache_df, pd.DataFrame([new_row])], ignore_index=True)

        self.save_cache()

    def get_pending_points(self, all_point_ids, stage):
        completed_mask = (self.cache_df['stage'] == stage) & (self.cache_df['status'] == 'completed')
        completed_points = set(self.cache_df.loc[completed_mask, 'point_id'])
        return [pid for pid in all_point_ids if pid not in completed_points]

import argparse
import re
import os
import time
import sys
import pandas as pd
from datetime import datetime
from point_selector import PointSelector
from predict_refiner import PredictionRefiner
from bioclim_selector import AuxVariableSelector
from calculate_spearman_correlation import calculate_spearman_correlation
from calculate_bias_score import calculate_bias_score
from concurrent.futures import ThreadPoolExecutor
from cache_manager import RefinementCache

class NStageProcessor:


    def __init__(self, api_key, output_dir, task, model,auxiliary_dir=None):
        self.task = task
        self.output_dir = output_dir
        safe_task_name = re.sub(r'[\\/*?:"<>|]', '_', task)
        self.cache = RefinementCache(safe_task_name, output_dir)
        self.selector = PointSelector(api_key, model,auxiliary_dir)
        self.refiner = PredictionRefiner(api_key,model, auxiliary_dir)
        self.aux_selector = AuxVariableSelector(model,api_key)
        self.model = model
    def run_stage(self, input_csv, groundtruth_tif):
        df = pd.read_csv(input_csv)
        refined_df = self._create_refined_df_with_cache(df)
        stage_responses = []
        current_stage = self._parse_stage(input_csv) + 1
        pending_points = self.cache.get_pending_points(df.index.tolist(), current_stage)
        print(f"points to process: {len(pending_points)}")
        retry_points = []
        for idx in pending_points:
            target = df.iloc[idx]
            cached_result = self.cache.get_point_result(idx, current_stage)

            if cached_result and (cached_result.get('refined_prediction') is None
                                  or pd.isna(cached_result['refined_prediction'])):
                self.cache.update_point_status(idx, current_stage, 'pending')
                retry_points.append(idx)
        all_points_to_process = list(set(pending_points + retry_points))
        for idx in all_points_to_process:
            target = df.iloc[idx]
            cached_result = self.cache.get_point_result(idx, current_stage)
            if cached_result and cached_result.get('refined_prediction') is not None:
                if not pd.isna(cached_result['refined_prediction']):
                    refined_df.loc[idx, 'Predictions'] = cached_result['refined_prediction']
                    continue
            selected_vars = self.aux_selector.select_variables(target, self.task)
            references = self._select_references(target, df, self.task, selected_vars)
            if not references.empty:
                new_pred = self.refiner.refine_prediction(target, references, self.task, selected_vars)
                point_response = {
                    'point_id': idx,
                    'var_selector': self.aux_selector.get_last_response(),
                    'selector': self.selector.get_last_response(),
                    'refiner': self.refiner.get_last_response()
                }
                stage_responses.append(point_response)
                if new_pred:
                    refined_df.loc[idx, 'Predictions'] = new_pred
                    self.cache.update_point_status(
                        idx, current_stage, 'completed',
                        selected_vars, new_pred
                    )
                else:
                    self.cache.update_point_status(idx, current_stage, 'failed')
            else:
                self.cache.update_point_status(idx, current_stage, 'failed')

        output_csv = self._generate_filename(input_csv, self.output_dir)
        refined_df.to_csv(output_csv, index=False)
        return output_csv, stage_responses

    def _create_refined_df_with_cache(self, df):
        refined_df = df.copy()
        for idx in df.index:
            latest_pred = self.cache.get_latest_prediction_for_point(idx)
            if latest_pred is not None and not pd.isna(latest_pred):
                refined_df.loc[idx, 'Predictions'] = latest_pred
        return refined_df

    def _select_references(self, target, df, task, selected_vars):
        try:
            original_indices = self.selector.select_points(target, df, task, selected_vars)
            if not original_indices:
                return pd.DataFrame()
            valid_indices = [idx for idx in original_indices if idx in df.index]
            if not valid_indices:
                return pd.DataFrame()
            return df.loc[valid_indices]

        except Exception as e:
            return pd.DataFrame()

    def _generate_filename(self, input_csv, output_dir):
        base = os.path.basename(input_csv).split('_stage')[0]
        stage_num = self._parse_stage(input_csv) + 1
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{base}_stage{stage_num}.csv")
    def _parse_stage(self, filename):
        match = re.search(r'_stage(\d+)', filename)
        return int(match.group(1)) if match else 0

def main():
    parser = argparse.ArgumentParser(
        description="GeoSR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("initial_csv", help="initial prediction file given by GeoLLM")
    parser.add_argument("groundtruth_tif", help="groundtruth_tif file")
    parser.add_argument("task", help="task name")
    parser.add_argument("--api_key", required=True, help="api-key")
    parser.add_argument("--model", help="modelname")
    parser.add_argument("--max_rounds", type=int, default=15, help="max_iteration_rounds")
    parser.add_argument("--output_dir", default="agent_results", help="output dir")
    parser.add_argument("--auxiliary_dir", help="a dir include all the external variables file(bio1-bio19)")

    args = parser.parse_args()


    output_dir = os.path.abspath(args.output_dir)
    print(f"\n{'=' * 40}")
    print(f"output_dir: {output_dir}")
    print(f"initial_csv: {os.path.abspath(args.initial_csv)}")
    print(f"groundtruth_tif: {os.path.abspath(args.groundtruth_tif)}")
    if args.auxiliary_dir:
        print(f"bioclim file directory: {os.path.abspath(args.auxiliary_dir)}")
    print(f"{'=' * 40}\n")


    os.makedirs(output_dir, exist_ok=True)
    metrics_log = os.path.join(output_dir, "refinement_metrics.csv")
    pd.DataFrame(columns=['Stage', 'Spearman', 'Bias']).to_csv(metrics_log, index=False)

    if not os.path.exists(args.initial_csv):
        sys.exit(f"error:initial_csv not exists {os.path.abspath(args.initial_csv)}")
    if not os.path.exists(args.groundtruth_tif):
        sys.exit(f"error:groundtruth_tif not exists {os.path.abspath(args.groundtruth_tif)}")

    auxiliary_dir_abs = os.path.abspath(args.auxiliary_dir) if args.auxiliary_dir else None
    processor = NStageProcessor(
        args.api_key,
        output_dir=output_dir,
        task=args.task,
        auxiliary_dir=auxiliary_dir_abs,
        model=args.model
    )

    current_csv = args.initial_csv
    best_spearman = -1
    current_stage = processor._parse_stage(current_csv)

    for round in range(1, args.max_rounds + 1):
        try:
            start_time = time.time()
            new_csv, responses = processor.run_stage(
                current_csv,
                args.groundtruth_tif
            )
            elapsed = time.time() - start_time
            if responses:
                for i, resp in enumerate(responses):
                    if i < len(responses) - 1:
                        print("-" * 80)
            else:
                print("⚠No any change")

            if not os.path.exists(new_csv):
                raise FileNotFoundError(f"No file generated: {os.path.abspath(new_csv)}")
            if pd.read_csv(new_csv).empty:
                raise ValueError("File is empty")


            df = pd.read_csv(new_csv)
            coordinates = list(zip(df['Latitude'], df['Longitude']))
            predictions = df['Predictions']

            spearman = calculate_spearman_correlation(
                coordinates,
                predictions,
                args.groundtruth_tif
            )
            bias = calculate_bias_score(coordinates, predictions, args.groundtruth_tif, len(df))

            print(f"\nstage {current_stage + 1} :")
            print(f"Spearman: {spearman:.4f}")
            print(f"bias: {bias:.4f}")

            pd.DataFrame([[current_stage + 1, spearman, bias]],
                         columns=['Stage', 'Spearman', 'Bias']
                         ).to_csv(metrics_log, mode='a', header=False, index=False)

            if spearman < best_spearman:
                print(f"▼ Spearman decreased ({best_spearman:.4f} → {spearman:.4f})")
                print(f"best stage:{current_stage}")
                break

            if spearman > best_spearman:
                best_spearman = spearman

            current_csv = new_csv
            current_stage += 1


            time.sleep(3)

        except Exception as e:
            print(f"❌ Round {round} error: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"Last  file: {current_csv}")
            break


    print("\n\n" + "=" * 40)
    print("all results:")
    print("=" * 40)
    metrics_df = pd.read_csv(metrics_log)
    print(metrics_df)

    if not metrics_df.empty:
        best_row = metrics_df.loc[metrics_df['Spearman'].idxmax()]
        print(f"\n✨ best result: stage{best_row['Stage']}")
        print(f"   Spearman: {best_row['Spearman']:.4f}")
        print(f"   Bias: {best_row['Bias']:.4f}")


    print(f"\nall files are saved to: {output_dir}")
    print(f"the metrics log: {os.path.abspath(metrics_log)}")
    if current_stage > 0:
        print(f"final stage file: {os.path.abspath(current_csv)}")

if __name__ == "__main__":
    main()


import os
import rasterio
import numpy as np
from rasterio.transform import rowcol
from rasterio.windows import Window

def read_tif_value(tif_path, lat, lon, dataset=None, return_dataset=False):

    try:
        # 处理数据集复用
        if dataset is None:
            # 验证文件路径
            if not tif_path or not os.path.isfile(tif_path):
                raise FileNotFoundError(f"TIFF 文件不存在: {tif_path}")

            dataset = rasterio.open(tif_path)

        if return_dataset:
            return dataset


        row, col = rowcol(dataset.transform, lon, lat)


        if row < 0 or col < 0 or row >= dataset.height or col >= dataset.width:
            return None


        value = dataset.read(1, window=Window(col, row, 1, 1))[0, 0]


        if value == dataset.nodata:
            return None

        return float(value)

    except Exception as e:
        print(f"TIFF read error: {str(e)}")
        return None

    finally:

        if dataset and not return_dataset and tif_path:
            dataset.close()

def get_tif_extent(tif_path):

    try:
        with rasterio.open(tif_path) as dataset:
            bounds = dataset.bounds
            return {
                'min_lon': bounds.left,
                'max_lon': bounds.right,
                'min_lat': bounds.bottom,
                'max_lat': bounds.top
            }
    except Exception as e:
        print(f"error: {str(e)}")
        return None

def resample_tif(input_path, output_path, scale_factor=0.5):

    try:
        with rasterio.open(input_path) as src:

            new_width = int(src.width * scale_factor)
            new_height = int(src.height * scale_factor)


            transform = src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )


            profile = src.profile
            profile.update({
                'width': new_width,
                'height': new_height,
                'transform': transform
            })


            with rasterio.open(output_path, 'w', **profile) as dst:

                for i in range(1, src.count + 1):
                    data = src.read(
                        i,
                        out_shape=(new_height, new_width),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                    dst.write(data, i)

            return True

    except Exception as e:
        print(f"error: {str(e)}")
        return False

def create_tif_cache(directory, cache_dir="tif_cache"):

    os.makedirs(cache_dir, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(cache_dir, f"cached_{filename}")

            if not os.path.exists(output_path):
                print(f"cache creating: {filename} -> cached_{filename}")
                resample_tif(input_path, output_path, scale_factor=0.5)

    return cache_dir

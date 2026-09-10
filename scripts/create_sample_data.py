import os
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

SAMPLES_DIR = Path("data") / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_geotiffs():
    print("Generating comprehensive remote sensing sample datasets in data/samples/ ...")

    width, height = 256, 256
    bands = 3
    np.random.seed(42)

    # 1. Primary GeoTIFF: Agricultural Patch
    b_red = np.random.randint(600, 1800, (height, width), dtype=np.uint16)
    b_green = np.random.randint(800, 2400, (height, width), dtype=np.uint16)
    b_blue = np.random.randint(400, 1200, (height, width), dtype=np.uint16)
    b_green[40:180, 50:200] += 1200
    b_red[190:240, 20:120] += 1500

    data_agri = np.stack([b_red, b_green, b_blue], axis=0)
    agri_path = SAMPLES_DIR / "sample_agricultural.tif"
    sample_patch_path = SAMPLES_DIR / "sample_patch.tif"

    if HAS_RASTERIO:
        transform = from_origin(-0.5510, 47.4520, 10.0, 10.0)
        with rasterio.open(
            agri_path, "w", driver="GTiff", height=height, width=width, count=bands,
            dtype=data_agri.dtype, crs="EPSG:32630", transform=transform, nodata=0
        ) as dst:
            dst.write(data_agri)
        with rasterio.open(
            sample_patch_path, "w", driver="GTiff", height=height, width=width, count=bands,
            dtype=data_agri.dtype, crs="EPSG:32630", transform=transform, nodata=0
        ) as dst:
            dst.write(data_agri)
        print(f"Created GeoTIFF: {agri_path}")
    else:
        norm_img = ((data_agri / 4000.0) * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(np.transpose(norm_img, (1, 2, 0)))
        img.save(agri_path)
        img.save(sample_patch_path)

    # 2. Sample PNG: Urban patch
    urban_arr = np.random.randint(100, 220, (256, 256, 3), dtype=np.uint8)
    urban_arr[120:136, :, :] = 50
    urban_arr[:, 120:136, :] = 50
    urban_path = SAMPLES_DIR / "sample_urban.png"
    Image.fromarray(urban_arr).save(urban_path)
    print(f"Created Sample PNG: {urban_path}")

    # 3. Bi-Temporal Pair: Flood/Disaster Change (T1 Pre-flood, T2 Post-flood)
    t1_arr = np.random.randint(120, 200, (height, width, 3), dtype=np.uint8)
    t1_arr[:, :, 1] += 30  # Green vegetated riverbank
    # Natural river channel
    t1_arr[110:140, :] = [30, 80, 160]

    t2_arr = t1_arr.copy()
    # Flooded inundation zone in T2
    t2_arr[70:185, :] = [25, 65, 185]  # Massive flood inundation
    
    t1_path = SAMPLES_DIR / "sample_bitemporal_t1.png"
    t2_path = SAMPLES_DIR / "sample_bitemporal_t2.png"
    Image.fromarray(t1_arr).save(t1_path)
    Image.fromarray(t2_arr).save(t2_path)
    print(f"Created Bi-Temporal Pair: {t1_path} and {t2_path}")

    # 4. Cross-Modal Pair: Optical (Sentinel-2 with partial cloud) + SAR (Sentinel-1 penetrating radar)
    opt_arr = np.random.randint(90, 180, (height, width, 3), dtype=np.uint8)
    # Optical has a bright cloud deck obscuring central region
    opt_arr[60:160, 60:200] = 245

    # SAR backscatter (grayscale intensity)
    sar_arr = np.random.randint(20, 80, (height, width), dtype=np.uint8)
    # Dense built-up buildings penetrate cloud with bright double-bounce radar return
    sar_arr[80:140, 80:180] = 230  # Bright radar backscatter for urban area under the cloud!
    # Water river has smooth specular reflection (dark radar return)
    sar_arr[200:230, :] = 10

    opt_path = SAMPLES_DIR / "sample_optical_s2.png"
    sar_path = SAMPLES_DIR / "sample_sar_s1.png"
    Image.fromarray(opt_arr).save(opt_path)
    Image.fromarray(sar_arr).save(sar_path)
    print(f"Created Optical + SAR Cross-Modal Pair: {opt_path} and {sar_path}")


if __name__ == "__main__":
    create_sample_geotiffs()

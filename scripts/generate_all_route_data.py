"""
Generate Comprehensive Test Datasets for all SatQuery AI Routes:
1. /vqa       - Single-Scene GeoTIFFs (Agriculture, Urban, Coast, Airport)
2. /grounding - Visual Grounding & Spatial Localization (Harbor, Urban, Crops, Solar)
3. /change    - Bi-Temporal Change Pairs (Flood, Urban Expansion, Deforestation)
4. /fusion    - Optical + SAR Multimodal Pairs (Cloudy City, River Haze, Crop Moisture)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import rasterio
from rasterio.transform import from_origin

DATA_DIR = Path("data")
VQA_DIR = DATA_DIR / "vqa"
GROUNDING_DIR = DATA_DIR / "grounding"
CHANGE_DIR = DATA_DIR / "change"
FUSION_DIR = DATA_DIR / "fusion"

for d in [VQA_DIR, GROUNDING_DIR, CHANGE_DIR, FUSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def create_geotiff(filename: Path, rgb_array: np.ndarray, crs: str = "EPSG:32630", origin=(500000.0, 3000000.0), res=10.0):
    """Saves a multi-band GeoTIFF with affine geotransform."""
    h, w, c = rgb_array.shape
    transform = from_origin(origin[0], origin[1], res, res)
    # Convert to 16-bit radiometric depth
    bands_16bit = ((rgb_array.astype(np.float32) / 255.0) * 10000.0).astype(np.uint16)
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=np.uint16,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(c):
            dst.write(bands_16bit[:, :, i], i + 1)
    print(f"  [+] Created GeoTIFF: {filename} (CRS: {crs}, Shape: {h}x{w}x{c})")


def generate_vqa_data():
    print("\n--- 1. Generating /vqa Test Scenes ---")
    
    # 1. Agriculture GeoTIFF
    img = Image.new("RGB", (400, 400), color=(40, 95, 35))
    draw = ImageDraw.Draw(img)
    # Field parcels
    colors = [(65, 135, 45), (145, 125, 45), (90, 155, 60), (120, 110, 50), (45, 110, 35)]
    for i in range(4):
        for j in range(4):
            c = colors[(i + j * 2) % len(colors)]
            draw.rectangle([i * 100 + 4, j * 100 + 4, (i + 1) * 100 - 4, (j + 1) * 100 - 4], fill=c)
    # Stream / canal
    draw.line([(0, 200), (120, 220), (250, 180), (400, 210)], fill=(30, 80, 160), width=12)
    create_geotiff(VQA_DIR / "01_sentinel2_agriculture.tif", np.array(img), crs="EPSG:32630")

    # 2. Urban Megacity GeoTIFF
    img_u = Image.new("RGB", (400, 400), color=(70, 75, 85))
    draw_u = ImageDraw.Draw(img_u)
    # Road network
    for x in range(50, 400, 80):
        draw_u.line([(x, 0), (x, 400)], fill=(40, 42, 48), width=8)
    for y in range(50, 400, 80):
        draw_u.line([(0, y), (400, y)], fill=(40, 42, 48), width=8)
    # Buildings / rooftops
    for x in range(15, 380, 40):
        for y in range(15, 380, 40):
            if (x + y) % 80 != 50:
                draw_u.rectangle([x, y, x + 24, y + 24], fill=(185, 175, 165), outline=(120, 110, 100))
    create_geotiff(VQA_DIR / "02_urban_city_megacity.tif", np.array(img_u), crs="EPSG:32643")

    # 3. Coastal Reservoir PNG
    img_c = Image.new("RGB", (400, 400), color=(85, 120, 70))
    draw_c = ImageDraw.Draw(img_c)
    draw_c.ellipse([80, 60, 340, 340], fill=(25, 75, 145), outline=(180, 195, 150), width=6)
    img_c.save(VQA_DIR / "03_coastal_reservoir.png")
    print(f"  [+] Created PNG: {VQA_DIR / '03_coastal_reservoir.png'}")

    # 4. Airport Runway Complex PNG
    img_a = Image.new("RGB", (400, 400), color=(90, 110, 80))
    draw_a = ImageDraw.Draw(img_a)
    # Runways
    draw_a.rectangle([60, 30, 110, 370], fill=(50, 50, 55))
    draw_a.line([(85, 40), (85, 360)], fill=(240, 240, 240), width=3)
    draw_a.rectangle([180, 80, 370, 130], fill=(50, 50, 55))
    draw_a.line([(190, 105), (360, 105)], fill=(240, 240, 240), width=3)
    # Terminal
    draw_a.rectangle([160, 180, 260, 280], fill=(200, 190, 180), outline=(100, 100, 100), width=2)
    img_a.save(VQA_DIR / "04_airport_runway_complex.png")
    print(f"  [+] Created PNG: {VQA_DIR / '04_airport_runway_complex.png'}")


def generate_grounding_data():
    print("\n--- 2. Generating /grounding Test Scenes ---")

    # 1. Harbor & Ships
    img_h = Image.new("RGB", (400, 400), color=(100, 105, 115))
    draw_h = ImageDraw.Draw(img_h)
    draw_h.rectangle([150, 0, 400, 400], fill=(20, 65, 130)) # Water
    # Docks
    draw_h.rectangle([150, 80, 280, 110], fill=(130, 135, 140))
    draw_h.rectangle([150, 220, 280, 250], fill=(130, 135, 140))
    # Ships
    draw_h.polygon([(300, 85), (350, 85), (360, 95), (350, 105), (300, 105)], fill=(220, 225, 230))
    draw_h.polygon([(300, 225), (350, 225), (360, 235), (350, 245), (300, 245)], fill=(200, 50, 50))
    img_h.save(GROUNDING_DIR / "01_harbor_ships_ports.png")
    print(f"  [+] Created PNG: {GROUNDING_DIR / '01_harbor_ships_ports.png'}")

    # 2. Solar Park Installation
    img_s = Image.new("RGB", (400, 400), color=(150, 135, 100)) # Sand/arid
    draw_s = ImageDraw.Draw(img_s)
    # Solar array in central-right
    for x in range(160, 360, 25):
        for y in range(80, 320, 18):
            draw_s.rectangle([x, y, x + 18, y + 12], fill=(25, 45, 95), outline=(180, 180, 200))
    img_s.save(GROUNDING_DIR / "02_solar_park_installation.png")
    print(f"  [+] Created PNG: {GROUNDING_DIR / '02_solar_park_installation.png'}")

    # 3. Dense Urban Commercial Center
    img_u = Image.new("RGB", (400, 400), color=(60, 90, 50)) # Vegetation surrounding
    draw_u = ImageDraw.Draw(img_u)
    # Commercial cluster in middle [100, 100, 300, 300]
    draw_u.rectangle([90, 90, 310, 310], fill=(90, 95, 105))
    for x in range(110, 290, 35):
        for y in range(110, 290, 35):
            draw_u.rectangle([x, y, x + 25, y + 25], fill=(210, 200, 190), outline=(50, 50, 60))
    img_u.save(GROUNDING_DIR / "03_urban_commercial_cluster.png")
    print(f"  [+] Created PNG: {GROUNDING_DIR / '03_urban_commercial_cluster.png'}")


def generate_change_data():
    print("\n--- 3. Generating /change Bi-Temporal Pairs ---")

    # Pair A: Flood Disaster (Pre vs Post)
    t1_flood = Image.new("RGB", (400, 400), color=(70, 130, 60))
    d1 = ImageDraw.Draw(t1_flood)
    d1.line([(0, 200), (400, 200)], fill=(30, 80, 150), width=18) # Normal river
    t1_flood.save(CHANGE_DIR / "flood_t1_pre_disaster.png")

    t2_flood = Image.new("RGB", (400, 400), color=(70, 130, 60))
    d2 = ImageDraw.Draw(t2_flood)
    # Inundated flood water overflowing banks
    d2.ellipse([50, 110, 350, 290], fill=(25, 70, 135))
    d2.line([(0, 200), (400, 200)], fill=(25, 70, 135), width=45)
    t2_flood.save(CHANGE_DIR / "flood_t2_post_disaster.png")
    print(f"  [+] Created Flood Pair: flood_t1_pre_disaster.png & flood_t2_post_disaster.png")

    # Pair B: Urban Expansion (2020 vs 2024)
    t1_urb = Image.new("RGB", (400, 400), color=(85, 125, 65))
    d1_u = ImageDraw.Draw(t1_urb)
    d1_u.rectangle([0, 0, 160, 400], fill=(130, 125, 120)) # Existing city on left
    t1_urb.save(CHANGE_DIR / "urban_t1_2020_baseline.png")

    t2_urb = Image.new("RGB", (400, 400), color=(85, 125, 65))
    d2_u = ImageDraw.Draw(t2_urb)
    d2_u.rectangle([0, 0, 320, 400], fill=(130, 125, 120)) # City expanded to right
    for x in range(170, 310, 30):
        for y in range(20, 380, 35):
            d2_u.rectangle([x, y, x + 20, y + 20], fill=(210, 80, 60)) # New construction
    t2_urb.save(CHANGE_DIR / "urban_t2_2024_expansion.png")
    print(f"  [+] Created Urban Expansion Pair: urban_t1_2020_baseline.png & urban_t2_2024_expansion.png")

    # Pair C: Wildfire & Deforestation
    t1_for = Image.new("RGB", (400, 400), color=(30, 110, 40)) # Dense forest
    t1_for.save(CHANGE_DIR / "forest_t1_healthy_canopy.png")

    t2_for = Image.new("RGB", (400, 400), color=(30, 110, 40))
    d2_f = ImageDraw.Draw(t2_for)
    d2_f.ellipse([80, 70, 320, 330], fill=(85, 55, 40)) # Burn scar / cleared land
    t2_for.save(CHANGE_DIR / "forest_t2_burned_scar.png")
    print(f"  [+] Created Deforestation Pair: forest_t1_healthy_canopy.png & forest_t2_burned_scar.png")


def generate_fusion_data():
    print("\n--- 4. Generating /fusion Optical + SAR Pairs ---")

    # Pair A: Cloudy Optical vs Penetrating SAR (Urban)
    opt_city = Image.new("RGB", (400, 400), color=(80, 90, 95))
    d_opt = ImageDraw.Draw(opt_city)
    # Clouds obscuring city
    d_opt.ellipse([60, 40, 340, 260], fill=(235, 240, 248))
    d_opt.ellipse([140, 150, 380, 360], fill=(225, 230, 240))
    opt_city = opt_city.filter(ImageFilter.GaussianBlur(10))
    opt_city.save(FUSION_DIR / "01_optical_cloudy_city.png")

    # SAR penetrates clouds showing double-bounce urban backscatter (bright white/yellow)
    sar_city = Image.new("L", (400, 400), color=50) # Dark background
    d_sar = ImageDraw.Draw(sar_city)
    for x in range(40, 360, 30):
        for y in range(40, 360, 30):
            d_sar.rectangle([x, y, x + 18, y + 18], fill=240) # Bright high backscatter
    sar_city.save(FUSION_DIR / "01_sar_penetrating_radar.png")
    print(f"  [+] Created Urban Fusion Pair: 01_optical_cloudy_city.png & 01_sar_penetrating_radar.png")

    # Pair B: Hazy River vs Water SAR (Specular Reflection)
    opt_riv = Image.new("RGB", (400, 400), color=(100, 125, 90))
    d_riv = ImageDraw.Draw(opt_riv)
    d_riv.polygon([(120, 0), (220, 0), (280, 400), (180, 400)], fill=(180, 195, 210)) # Hazy river
    opt_riv = opt_riv.filter(ImageFilter.GaussianBlur(8))
    opt_riv.save(FUSION_DIR / "02_optical_hazy_river.png")

    # SAR shows specular water surface as pitch black (no backscatter return)
    sar_riv = Image.new("L", (400, 400), color=150) # Moderate land backscatter
    d_sar_riv = ImageDraw.Draw(sar_riv)
    d_sar_riv.polygon([(120, 0), (220, 0), (280, 400), (180, 400)], fill=15) # Pitch black water
    sar_riv.save(FUSION_DIR / "02_sar_water_specular.png")
    print(f"  [+] Created River Fusion Pair: 02_optical_hazy_river.png & 02_sar_water_specular.png")


if __name__ == "__main__":
    generate_vqa_data()
    generate_grounding_data()
    generate_change_data()
    generate_fusion_data()
    print("\n============================================================")
    print("All rich test datasets for all routes generated successfully!")
    print("Folders created:")
    print("  - data/vqa/       (4 GeoTIFF & PNG scenes)")
    print("  - data/grounding/ (3 Visual Grounding scenes)")
    print("  - data/change/    (3 Before/After Change pairs)")
    print("  - data/fusion/    (2 Optical + SAR Radar pairs)")
    print("============================================================")

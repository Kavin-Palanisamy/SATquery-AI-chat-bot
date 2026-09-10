import requests
import json

def run_live_query(prompt, img_path):
    with open(img_path, 'rb') as f:
        r = requests.post(
            'http://127.0.0.1:8000/agent/analyze',
            data={'question': prompt, 'task_mode': 'agent'},
            files={'image': f}
        )
    print(f"=== QUERY: '{prompt}' (Image: {img_path}) ===")
    print(f"HTTP Status: {r.status_code}")
    data = r.json()
    print("Tool Used:     ", data.get("tool_used"))
    print("Model Name:    ", data.get("model"))
    print("Confidence:    ", data.get("confidence"))
    print("Answer/Summary:", data.get("answer"))
    if data.get("boxes"):
        print("Bounding Boxes:", data["boxes"])
    if data.get("grounding_mask_url"):
        print("Mask URL:       Available (Base64 PNG)")
    print("Audit Steps:")
    for step in data.get("execution_trace", {}).get("steps", []):
        print(f"  Step {step.get('step')}: {step.get('action')} - {step.get('details')}")
    print("-" * 60)

if __name__ == "__main__":
    run_live_query("Find and highlight the water body in this image.", "data/samples/sample_agricultural.tif")
    run_live_query("Find and highlight the water body in this image.", "data/vqa/03_coastal_reservoir.png")
    run_live_query("Find the buildings.", "data/samples/sample_urban.png")
    run_live_query("Highlight the road.", "data/samples/sample_urban.png")
    run_live_query("Locate vegetation.", "data/samples/sample_agricultural.tif")
    run_live_query("Describe the land cover and major objects visible in this image.", "data/samples/sample_agricultural.tif")

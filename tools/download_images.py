import csv
import os
import requests
from pathlib import Path

OUTPUT_DIR = Path("glove_new_images")
CSV_CANDIDATES = (
    Path("list_of_all_urls.csv"),
    OUTPUT_DIR / "glove_new_images.csv",
)

# Start with 200 images
MAX_IMAGES = 200

OUTPUT_DIR.mkdir(exist_ok=True)

CSV_FILE = next(
    (path for path in CSV_CANDIDATES if path.is_file()),
    None,
)
if CSV_FILE is None:
    expected_files = ", ".join(str(path) for path in CSV_CANDIDATES)
    raise FileNotFoundError(
        f"No URL list found. Expected one of: {expected_files}"
    )

print("Starting image download...")
print(f"Output folder: {OUTPUT_DIR.resolve()}")
print(f"URL list: {CSV_FILE.resolve()}")
print()

downloaded = 0
failed = 0

with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file)

    for row in reader:
        if not row:
            continue

        url = row[0].strip()

        if not url.startswith(("http://", "https://")):
            continue

        if downloaded >= MAX_IMAGES:
            break

        try:
            response = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            response.raise_for_status()

            filename = f"image_{downloaded + 1:04d}.jpg"
            filepath = OUTPUT_DIR / filename

            with open(filepath, "wb") as image_file:
                image_file.write(response.content)

            downloaded += 1

            print(f"[{downloaded}/{MAX_IMAGES}] Downloaded: {filename}")

        except Exception as e:
            failed += 1
            print(f"Failed: {url}")
            print(f"Reason: {e}")

print()
print("=" * 50)
print("DOWNLOAD COMPLETED")
print("=" * 50)
print(f"Downloaded : {downloaded}")
print(f"Failed     : {failed}")
print(f"Folder     : {OUTPUT_DIR.resolve()}")
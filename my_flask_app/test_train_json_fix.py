import io
import csv
import requests

base = "http://localhost:5000"

rows = [
    ["f1", "f2", "target"],
    [1.0, 2.0, "A"],
    [1.2, 1.8, "A"],
    [3.0, 3.5, "B"],
    [3.1, 3.6, "B"],
    [5.0, 5.0, "C"],
    [5.1, 5.2, "C"],
    [2.2, 2.0, "A"],
    [3.2, 3.2, "B"],
    [5.2, 5.1, "C"],
]

buf = io.BytesIO()
text = io.TextIOWrapper(buf, encoding="utf-8", newline="")
writer = csv.writer(text)
writer.writerows(rows)
text.flush()
buf.seek(0)

upload = requests.post(
    f"{base}/api/upload",
    files={"file": ("tiny.csv", buf, "text/csv")},
    data={
        "is_labeled": "labeled",
        "data_type": "categorical",
        "target_column": "target",
        "selected_columns": '["f1","f2","target"]',
    },
)
print("upload", upload.status_code)
print(upload.text)
upload.raise_for_status()
file_id = upload.json()["file_id"]

train = requests.post(
    f"{base}/api/train",
    json={"file_id": file_id, "model_name": "XGBoost Classifier"},
)
print("train", train.status_code)
print(train.text[:800])

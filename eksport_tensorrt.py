from ultralytics import YOLO

print("=== 1. Ładowanie wytrenowanego modelu best.pt ===")
model = YOLO("/home/student/runs/detect/train-3/weights/best.pt")

print("=== 2. Rozpoczynam konwersję do formatu NVIDIA TensorRT (FP16) ===")
# half=True (FP16) uruchamia dedykowane rdzenie Tensor w Twoim układzie Orin Nano
exported_path = model.export(format="engine", half=True, device=0)

print(f"=== 3. Sukces! Gotowy model sprzętowy znajduje się w: {exported_path} ===")

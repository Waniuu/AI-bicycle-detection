from ultralytics import YOLO

# Krok 1: Ładujemy Twój czysty model w Pytorch (.pt), który wcześniej wygenerowałeś
model = YOLO("/home/student/CounterProject/SilnikV5/results/runs/yolo11s_licznik/weights/best.pt")

print("Rozpoczynam kwantyzację do INT8 pod architekturę NVIDIA Orin...")

# Krok 2: Uruchamiamy kalibrację.
# UWAGA: Podajesz tutaj plik data.yaml ze swoimi zdjęciami (Roboflow). 
# TensorRT musi przetestować INT8 na prawdziwych rowerach i hulajnogach, 
# aby dobrać idealne progi liczbowe i nie popsuć dokładności.
model.export(
    format="engine", 
    int8=True,       # To jest najważniejsza komenda (kwantyzacja!)
    data="/home/student/CounterProject/SilnikV5/results/runs/yolo11s_licznik/weights/data.yaml", 
    imgsz=640, 
    device=0,
    workspace=4      # Pozwalamy użyć 4GB RAM do kalibracji
)

print("Gotowe! Nowy plik ma końcówkę .engine. Podmień do niego ścieżkę w config.yaml")
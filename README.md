# 🔍 Detecção de objetos cortantes com PyTorch

## Problema

Utilizar Inteligência Artificial para identificar objetos cortantes (facas, tesouras e similares) e emitir alertas. 

## Objetivos

- Construir ou buscar um dataset contendo imagens de facas, tesouras e outros objetos cortantes em diferentes condições de ângulos e iluminações;
- Anotar o dataset para treinar o modelo supervisionado, incluindo imagens negativas (sem objetos perigosos) para reduzir falsos positivos;
- Treinar o modelo;
- Desenvolver um sistema de alertas.

## Solução

Utilizando **PyTorch**, foi criado uma Rede Neural Convolucional (CNN) para realizar essa classificação entre objetos **cortantes** e **não-cortantes**.

Para o sistema de alertas, foi usado a API do Telegram, aplicativo de envio de mensagens, onde foi configurado um bot que irá emitir um resumo após executar a leitura de um vídeo.

## Instalação

Foram usadas as seguintes bibliotecas

- Python 3.11+
- matplotlib
- PIL
- PyTorch
- requests
- torchvision

```bash
pip install torch torchvision matplotlib numpy requests opencv-python
```

```txt
# requirements.txt
matplotlib==3.10.0
numpy==2.0.2
opencv-python==4.11.0.86
requests==2.32.3
torch @ https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-linux_x86_64.whl
torchvision @ https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp311-cp311-linux_x86_64.whl
```

## Dataset

### 1. Organização

Preparamos a estrutura de pastas

```bash
mkdir -p dataset_binary/unsafe
mkdir -p dataset_binary/safe
```

### 2. *Download*

Realizamos o *download* do primeiro dataset e depois extraimos.

```bash
wget -O dataset.zip "https://drive.usercontent.google.com/download?id=1Szc920DAh5kU8Qk38Doq0znEVR1QmTZS&export=download&confirm=t"
unzip dataset.zip
```

Movemos os arquivos desejados para a estrutura criada

```py
import os
for dirname, dirpaths, filenames in os.walk('./OD-WeaponDetection-master/Knife classification'):
    if len(dirpaths) != 0:
        continue
    
    if 'AAAKnife' in dirname:    
        os.system(f'cp "{dirname}/*" dataset_binary/unsafe')
    else:
        os.system(f'cp "{dirname}/*" dataset_binary/safe')
```

Depois realizamos o download dos datasets restantes.

A origem destes arquivos vieram da plataforma RoboFlow, onde foi preciso realizar o download prévio após cadastrar uma conta, e importá-los para o notebook

Estes datasets contém mais imagens de objetos cortantes, onde podemos adicionar à nossa estrutura existente

- https://universe.roboflow.com/tugas-machine-learning-m5kep/tugas-machine-learning-wira-b1/dataset/1/download
- https://universe.roboflow.com/object-detection-practice-dgcci/cutter-dchc8/dataset/2/download
- https://universe.roboflow.com/mayank-soni-gwr81/project2.0/dataset/1/download/createml
- https://universe.roboflow.com/mit-rqoyj/knife-detection-6ihw2/dataset/3/download
- https://universe.roboflow.com/company-isoyi/company2/dataset/1/download
- https://universe.roboflow.com/project-cwxdu/knife-detect-jxtjw
- https://universe.roboflow.com/guns-7wv0t/facas_tesouras
- https://universe.roboflow.com/lab9-bbqzq/knife-h44d6

```bash
unzip -n './Company2.v1i.multiclass.zip' -d dataset_1
unzip -n './cutter.v2i.multiclass.zip' -d dataset_2
unzip -n './Facas_Tesouras.v1i.multiclass.zip' -d dataset_3
unzip -n './knife detect.v4i.multiclass.zip' -d dataset_4
unzip -n './Knife detection.v3i.multiclass.zip' -d dataset_5
unzip -n './Knife.v1i.multiclass.zip' -d dataset_6
unzip -n './PROJECT2.0.v1i.multiclass.zip' -d dataset_7
unzip -n './Tugas Machine Learning Wira B1.v1i.multiclass.zip' -d dataset_8

cp -r dataset_*/train/* dataset_binary/unsafe
cp -r dataset_*/valid/* dataset_binary/unsafe
cp -r dataset_*/test/* dataset_binary/unsafe
```

### 3. Estrutura final de pastas

Com isso, já possuimos a estrutura que nosso modelo irá usar

```
dataset_binary/
├── safe/ (wallet, cash, card, smartphone)
│   ├── *.jpg
│   └── ...
└── unsafe/ (knife, pistol)
    ├── *.jpg
    └── ...
```

## Sistema de alerta (Bot do Telegram)

Foram definidas duas funções, uma para enviar um texto e outra para enviar o arquivo de vídeo.

Para usar a API do Telegram, é preciso obter as credenciais do bot e o ID do usuário que receberá as mensagens

Uma vez que esses valores foram obtidos, é preciso que estes sejam definidos como variáveis de ambiente

```py
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
USER_ID = os.environ.get('USER_ID')
```

```py
def send_telegram_video(file_path: str, caption: str = ""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"
    payload = {
        "chat_id": USER_ID,
        "caption": caption
    }
    files = {
        "video": open(file_path, "rb")
    }

    try:
        response = requests.post(url, data=payload, files=files)
        if response.status_code != 200:
            print("❌ Error sending video:", response.text)
        else:
            print("✅ Video sent successfully")
    except Exception as e:
        print("❌ Exception:", e)
```

```py
def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": USER_ID,
        "text": message
    }

    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print("❌ Telegram error:", response.text)
        else:
            print("✅ Telegram message sent")
    except Exception as e:
        print("❌ Telegram exception:", e)
```


## Definição da CNN

### 1. Arquitetura de camadas

Antes de alimentar o modelo, as imagens são redimensionadas para 128 x 128 px

A seguir, a estrutura de camadas (ordem de cima para baixo):

| Camada | Descrição | Entrada | Saída |
|--|--|--|--|
| 1. Conv2D | RGB para 1 canal | (B x 128 x 128 x 3) | (B x 128 x 128 x 1) |
| 2. Conv2D | Extração de features 3x3 | (B x 128 x 128 x 1) | (B x 128 x 128 x 16) |
| 3. ReLU | Ativações | (B x 128 x 128 x 16) | (B x 128 x 128 x 16) |
| 4. MaxPool | Redução espacial | (B x 128 x 128 x 16) | (B x 64 x 64 x 16) |
| 5. Conv2D | Mais extração de features 3x3 | (B x 64 x 64 x 16) | (B x 64 x 64 x 32) |
| 6. ReLU | Ativações | (B x 64 x 64 x 32) | (B x 64 x 64 x 32) |
| 7. MaxPool | Redução espacial | (B x 64 x 64 x 32) | (B x 32 x 32 x 32) |
| 8. Flatten | Achatamento para vetor | (B x 32 x 32 x 32) | (B x 32768) |
| 9. Linear | Predição final com 1 valor (logit) | (B x 32768) | (B x 1) |

Estes foram as definições para a função de *loss* e o modelo de otimização
```py
# Função de loss
criterion = nn.BCEWithLogitsLoss()

# Modelo de otimização
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### 2. Código PyTorch

```py
class SharpItemDetection(nn.Module):
    layers: nn.Sequential

    def __init__(self):
        super().__init__()

        output_size = (IMG_SIZE // 2 // 2)
        feature_count = 16

        self.layers = nn.Sequential(
            # RGBA > Grayscale (INxINx3 > INxINx1)
            nn.Conv2d(
                in_channels=3,
                out_channels=1,
                kernel_size=1,
            ),
            # Get features (INxINx1 > INxINx16)
            nn.Conv2d(
                in_channels=1,
                out_channels=feature_count,
                kernel_size=3,
                padding=1,
            ),
            # Remove negative values (INxINx16 > INxINx16)
            nn.ReLU(),
            # Downsample (INxINx16 > (IN/2)x(IN/2)x16)
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            # Get features #2 ((IN/2)x(IN/2)x16 > (IN/2)x(IN/2)x32)
            nn.Conv2d(
                in_channels=16,
                out_channels=feature_count*2,
                kernel_size=3,
                padding=1,
            ),
            # Remove negative values #2 ((IN/2)x(IN/2)x32 > (IN/2)x(IN/2)x32)
            nn.ReLU(),
            # Downsample #2 ((IN/2)x(IN/2)x32 > (IN/2/2)x(IN/2/2)x32)
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),            
            # Flatten for linear
            nn.Flatten(),
            # Multiply features with weights + bias
            nn.Linear(
                in_features=output_size * output_size * feature_count*2,
                out_features=1,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
```

### 3. Treinamento

```py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SharpItemDetection().to(device)

model.train()
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()

        # Zero the gradients
        optimizer.zero_grad()

        # Send images to model
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs.squeeze(), labels)

        # Back propagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}"
    )
```

### 4. Avaliação

```py
model.eval()
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation for testing
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device).float()

        # Forward pass
        outputs = model(images)

        # Apply sigmoid to get probabilities
        predicted_probs = torch.sigmoid(outputs).squeeze()

        # Convert probabilities to binary predictions (0 or 1)
        predicted_labels = (predicted_probs > 0.5).float()

        # Calculate accuracy
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

# Calculate and print accuracy
accuracy = correct / total
print(f"Eval Accuracy: {accuracy * 100:.2f}%")
```

### 5. Testes

#### 5.1. Imagens restantes do dataset

```py
def detect_objects_batch(
    model: SharpItemDetection,
    image,
    tile_size=IMG_SIZE,
    stride=64,
    scales=[1.0, 0.75, 0.5],
    threshold=0.5,
    batch_size=64
):
    detect_transform = transforms.Compose([
        transforms.Lambda(lambda img: PIL.Image.fromarray(img)),  # Converts NumPy to PIL
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    model.eval()
    h, w, _ = image.shape
    all_tiles = []
    tile_info = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = image[y:y + tile_size, x:x + tile_size]

            for scale in sorted(scales):
                scaled_tile = cv2.resize(tile, (0, 0), fx=scale, fy=scale)
                resized_tile = cv2.resize(scaled_tile, (tile_size, tile_size))

                tile_tensor = detect_transform(resized_tile)
                all_tiles.append(tile_tensor)
                tile_info.append((x, y, scale))

    boxes = []

    with torch.no_grad():
        for i in range(0, len(all_tiles), batch_size):
            batch = torch.stack(all_tiles[i:i+batch_size]).to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            for j, prob in enumerate(probs):
                if prob > threshold:
                    x, y, scale = tile_info[i + j]
                    box_w = int(tile_size / scale)
                    box_h = int(tile_size / scale)
                    boxes.append((x, y, x + box_w, y + box_h))

    return boxes
```

```py
def show_boxes(image, boxes, figsize=(12, 12), title="Detected Objects"):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)

    plt.title(title)
    plt.axis('off')
    plt.show()
```

```py
# Testar com restante do dataset
for _images, _labels in test_loader:
    for i in range(len(_images)):
        img = _images[i].permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        model = SharpItemDetection().to(device)
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
        boxes = detect_objects_batch(model, img, threshold=0.5)
        show_boxes(img, boxes)
```

#### 5.2. Vídeos de teste

Para evitar que houvesse a identificação dos objetos cortantes, mas não soubesse de fato o que o modelo entendeu, foi implementado uma lógica ingênua de divisão do vídeo em uma matriz, onde cada seção dessa matriz é repassada para o modelo ao invés do frame completo.

```bash
# Download videos
wget -O video_1.mp4 "https://drive.usercontent.google.com/download?id=1AV6y7OFPgq9UiU0TMUjoaoYQHsvKO__u&export=download&confirm=t"
wget -O video_2.mp4 "https://drive.usercontent.google.com/download?id=1XBhBKY9QHo0xj8gXMYcq92e-vrECrNH3&export=download&confirm=t"
```

```py
# Run model over grid tiles instead of whole frames
def detect_objects_grid_scaled(
    model,
    image,
    rows=4,
    cols=4,
    input_size=128,
    scales=[1.0, 0.75, 0.5],
    threshold=0.5,
    batch_size=64
):
    detect_transform = transforms.Compose([
        transforms.Lambda(lambda img: PIL.Image.fromarray(img)),  # Converts NumPy to PIL
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    model.eval()
    h, w, _ = image.shape
    tile_h = h // rows
    tile_w = w // cols

    all_tiles = []
    tile_info = []

    for row in range(rows):
        for col in range(cols):
            x1 = col * tile_w
            y1 = row * tile_h
            x2 = x1 + tile_w
            y2 = y1 + tile_h
            tile = image[y1:y2, x1:x2]

            for scale in sorted(scales):
                # Scale the tile (zoom in/out)
                for scale in sorted(scales):
                    scaled_tile = cv2.resize(tile, (0, 0), fx=scale, fy=scale)
                    resized_tile = cv2.resize(scaled_tile, (input_size, input_size))

                    tile_tensor = detect_transform(resized_tile)
                    all_tiles.append(tile_tensor)

                    # Adjust detection box to the center of the tile based on scale
                    center_x = x1 + tile_w // 2
                    center_y = y1 + tile_h // 2
                    box_w = int(tile_w * scale)
                    box_h = int(tile_h * scale)

                    x_adj1 = max(center_x - box_w // 2, 0)
                    y_adj1 = max(center_y - box_h // 2, 0)
                    x_adj2 = x_adj1 + box_w
                    y_adj2 = y_adj1 + box_h

                    tile_info.append((x_adj1, y_adj1, x_adj2, y_adj2))
                # Scale the tile (zoom in/out)

    boxes = []

    with torch.no_grad():
        # Run model over video tiles in batch
        for i in range(0, len(all_tiles), batch_size):
            batch = torch.stack(all_tiles[i:i+batch_size]).to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            for j, prob in enumerate(probs):
                # prob > threshold == unsafe
                if prob > threshold:
                    boxes.append(tile_info[i + j])

    return boxes
```

```py
# Keep only the grid tile with highest score
def apply_nms(boxes, iou_threshold=0.3):
    if len(boxes) == 0:
        return []

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.ones((boxes_tensor.shape[0],))

    keep = ops.nms(boxes_tensor, scores, iou_threshold)
    return boxes_tensor[keep].int().tolist()
```

```py
def process_video(
    model,
    video_path,
    rows=4,
    cols=6,
    scales=[1.0, 0.75],
    threshold=0.5,
    iou_threshold=0.3,
    draw_boxes=True,
    verbose=True,
    output_suffix="_out.mp4"
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Setup video writer
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = video_path.replace(".mp4", output_suffix)
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    total_detections = 0
    detection_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_number += 1

        # Detect sharp items with region + scale analysis
        boxes = detect_objects_grid_scaled(
            model, frame_rgb,
            rows=rows, cols=cols,
            scales=scales,
            threshold=threshold
        )

        boxes = apply_nms(boxes, iou_threshold=iou_threshold)

        if len(boxes) > 0:
            detection_frames += 1
            total_detections += len(boxes)

        # Draw boxes (on BGR frame for saving)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write frame with boxes to output video
        out_writer.write(frame)

        if verbose:
            print(f"[Frame {frame_number}] Detected {len(boxes)} sharp item(s).")

    cap.release()
    out_writer.release()

    print(f"Finished processing {video_path} → {output_path}")
    print(f"Frames processed: {frame_number}")
    print(f"Total detections: {total_detections} across {detection_frames} frame(s)")

    # 🔔 Telegram summary message
    summary_msg = (
        f"🔔 Video analysis complete: {video_path}\n"
        f"📹 Total frames: {frame_number}\n"
        f"⚠️ Frames with detections: {detection_frames}\n"
        f"🧨 Total detected sharp items: {total_detections}"
    )
    send_telegram_message(summary_msg)
```

```py
model = SharpItemDetection().to(device)
model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
model.eval()

process_video(model, 'video_1.mp4', rows=2, cols=3, scales=[1.25, 1.0, 0.75], threshold=0.99)
send_telegram_video(file_path='video_1_out.mp4')

process_video(model, 'video_2.mp4', rows=2, cols=2, scales=[1.25, 1.0, 0.9, 0.8, 0.7], threshold=0.9)
send_telegram_video(file_path='video_2_out.mp4')
```

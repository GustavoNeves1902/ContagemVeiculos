import cv2
import numpy as np

# ── Configuração ──────────────────────────────────────────────────────────────
URL_CAMERA    = "https://video04.logicahost.com.br/portovelhomamore/fozpontedaamizadesentidoparaguai.stream/playlist.m3u8"
FRAMES_APRENDIZADO = 400   # frames para o MoG2 aprender o fundo antes de salvar
IMAGEM_SAIDA_ORIGINAL = "fundo_original.png"   # frame planificado colorido
IMAGEM_SAIDA_MASCARA  = "fundo_mascara.png"    # máscara binária do MoG2
IMAGEM_SAIDA_CANVAS   = "fundo_canvas.png"     # frame com caixas + centróides

# ── Filtro de tamanho para caixa delimitadora (em pixels) ─────────────────────
# Objetos fora desses limites são descartados (ruído, pedestres, etc.)
LARGURA_MIN, LARGURA_MAX = 20, 300   # largura mínima e máxima do veículo
ALTURA_MIN,  ALTURA_MAX  = 20, 300   # altura mínima e máxima do veículo

# ── Perspectiva (idêntica ao video_mog2.py) ───────────────────────────────────
pts_origem  = np.float32([[529, 312], [761, 282], [1083, 779], [772, 801]])
pts_destino = np.float32([[0, 0],    [600, 0],   [600, 800],  [0, 800]])
matriz_perspectiva = cv2.getPerspectiveTransform(pts_origem, pts_destino)

# ── MoG2 (idêntico ao video_mog2.py) ─────────────────────────────────────────
subtrator = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=30,
    detectShadows=False
)

# ── Kernels morfológicos ─────────────────────────────────────────────────────
# CLOSE: une bordas do veículo, fechando lacunas no contorno
# Kernel menor = forma mais fiel ao carro (aumente se ainda houver buracos)
kernel_fechar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
# DILATE: expande levemente para recuperar bordas apagadas
kernel_dilatar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# ── Abertura do stream ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(URL_CAMERA)
if not cap.isOpened():
    print("[ERRO] Não foi possível abrir o stream. Verifique a URL.")
    exit()

print(f"Aguardando {FRAMES_APRENDIZADO} frames para o MoG2 aprender o fundo...")
print("Pressione 'q' para sair a qualquer momento.")

frame_num = 0
frame_salvo = None
mascara_salva = None

while True:
    sucesso, frame = cap.read()
    if not sucesso:
        print("[AVISO] Frame perdido, tentando novamente...")
        continue

    frame_num += 1

    # ── Perspectiva (mesma região da pista do video_mog2.py) ─────────────────
    frame_plan = cv2.warpPerspective(frame, matriz_perspectiva, (600, 800))

    # ── Pré-processamento ────────────────────────────────────────────────────
    cinza = cv2.cvtColor(frame_plan, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(cinza, (5, 5), 0)

    # ── Subtração de fundo ───────────────────────────────────────────────────
    mascara_raw = subtrator.apply(blur)
    _, mascara  = cv2.threshold(mascara_raw, 1, 255, cv2.THRESH_BINARY)

    # ── Preenchimento de buracos internos ─────────────────────────────────────
    # 1) CLOSE: dilata e depois erode — une bordas e fecha lacunas no contorno
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_fechar)
    # 2) DILATE: expande os blobs resultantes
    mascara = cv2.dilate(mascara, kernel_dilatar, iterations=1)
    # 3) fillPoly: preenche o interior de cada contorno completamente sólido
    #    Garante que regiões fechadas (mesmo com bordas imperfeitas) fiquem brancas
    contornos_fill, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(mascara, contornos_fill, 255)

    # ── Moldura preta de 1 pixel ──────────────────────────────────────────────
    # Evita que findContours detecte falsamente as bordas da imagem como contornos
    mascara[0, :]  = 0
    mascara[-1, :] = 0
    mascara[:, 0]  = 0
    mascara[:, -1] = 0

    # ── Detecção de contornos e caixas delimitadoras ──────────────────────────
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas = frame_plan.copy()  # cópia colorida para desenhar as anotações

    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)

        # Condição: apenas objetos com proporções compatíveis com veículos
        if not (LARGURA_MIN <= w <= LARGURA_MAX and ALTURA_MIN <= h <= ALTURA_MAX):
            continue

        # Desenha a caixa delimitadora (verde)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calcula o centróide da caixa
        cx = x + w // 2
        cy = y + h // 2

        # Desenha um círculo sobre o centróide (vermelho)
        cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)

    # ── Exibição em tempo real ───────────────────────────────────────────────
    progresso = min(frame_num, FRAMES_APRENDIZADO)

    mascara_bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    cv2.putText(mascara_bgr, f"Pixels em movimento: {cv2.countNonZero(mascara)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.imshow("Canvas (caixas + centroide)", canvas)
    cv2.imshow("Mascara MoG2", mascara_bgr)

    # ── Salva após aprendizado ────────────────────────────────────────────────
    if frame_num == FRAMES_APRENDIZADO:
        cv2.imwrite(IMAGEM_SAIDA_ORIGINAL, frame_plan)
        cv2.imwrite(IMAGEM_SAIDA_MASCARA, mascara)
        cv2.imwrite(IMAGEM_SAIDA_CANVAS, canvas)
        print(f"\nFrame planificado salvo em '{IMAGEM_SAIDA_ORIGINAL}'")
        print(f"Máscara MoG2 salva em      '{IMAGEM_SAIDA_MASCARA}'")
        print(f"Canvas (caixas+centr.) salvo em '{IMAGEM_SAIDA_CANVAS}'")
        print("Pressione 'q' para fechar.")

    tecla = cv2.waitKey(30) & 0xFF
    if tecla == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

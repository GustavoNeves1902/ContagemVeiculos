import cv2
import numpy as np
from collections import OrderedDict
import math
import time

# ==============================================================================
# RASTREADOR DE CENTRÓIDES
# Garante que cada veículo seja contado exatamente 1 vez ao cruzar a linha.
# ==============================================================================
class RastreadorCentroide:
    def __init__(self, desaparecimento_max=30, distancia_max=120):
        self.proximo_id = 0
        self.objetos = OrderedDict()        # id -> (cx, cy)
        self.desaparecidos = OrderedDict()  # id -> frames sem aparecer
        self.pos_y_anterior = OrderedDict() # id -> cy do frame anterior
        self.desaparecimento_max = desaparecimento_max
        self.distancia_max = distancia_max

    def _registrar(self, centroide):
        self.objetos[self.proximo_id] = centroide
        self.desaparecidos[self.proximo_id] = 0
        self.pos_y_anterior[self.proximo_id] = centroide[1]
        self.proximo_id += 1

    def _remover(self, obj_id):
        del self.objetos[obj_id]
        del self.desaparecidos[obj_id]
        del self.pos_y_anterior[obj_id]

    def atualizar(self, retangulos):
        # Centróides das novas detecções
        novos_centroides = [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in retangulos]

        if not novos_centroides:
            for obj_id in list(self.desaparecidos):
                self.desaparecidos[obj_id] += 1
                if self.desaparecidos[obj_id] > self.desaparecimento_max:
                    self._remover(obj_id)
            return self.objetos

        if not self.objetos:
            for c in novos_centroides:
                self._registrar(c)
            return self.objetos

        ids_atuais = list(self.objetos.keys())
        cents_atuais = list(self.objetos.values())

        # Matriz de distâncias
        D = np.array([[math.hypot(ca[0]-cn[0], ca[1]-cn[1])
                       for cn in novos_centroides]
                      for ca in cents_atuais])

        linhas = D.min(axis=1).argsort()
        colunas = D.argmin(axis=1)[linhas]

        linhas_usadas = set()
        colunas_usadas = set()

        for linha, col in zip(linhas, colunas):
            if linha in linhas_usadas or col in colunas_usadas:
                continue
            if D[linha, col] > self.distancia_max:
                continue
            obj_id = ids_atuais[linha]
            self.pos_y_anterior[obj_id] = self.objetos[obj_id][1]  # guarda Y antigo
            self.objetos[obj_id] = novos_centroides[col]
            self.desaparecidos[obj_id] = 0
            linhas_usadas.add(linha)
            colunas_usadas.add(col)

        for linha in set(range(len(ids_atuais))) - linhas_usadas:
            obj_id = ids_atuais[linha]
            self.desaparecidos[obj_id] += 1
            if self.desaparecidos[obj_id] > self.desaparecimento_max:
                self._remover(obj_id)

        for col in set(range(len(novos_centroides))) - colunas_usadas:
            self._registrar(novos_centroides[col])

        return self.objetos


# ==============================================================================
# MESCLAGEM DE BOUNDING BOXES
# Une retângulos que se sobrepõem ou estão próximos, evitando múltiplos IDs
# para o mesmo veículo físico.
# ==============================================================================
def mesclar_retangulos(rects, margem=25):
    """Mescla iterativamente bounding boxes que se tocam ou estão dentro da margem."""
    if not rects:
        return []
    rects = list(rects)
    mesclado = True
    while mesclado:
        mesclado = False
        nova_lista = []
        usados = [False] * len(rects)
        for i, (x1, y1, w1, h1) in enumerate(rects):
            if usados[i]:
                continue
            for j, (x2, y2, w2, h2) in enumerate(rects):
                if i == j or usados[j]:
                    continue
                # Verifica sobreposição com margem de proximidade
                if (x1 - margem < x2 + w2 and x1 + w1 + margem > x2 and
                        y1 - margem < y2 + h2 and y1 + h1 + margem > y2):
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    rects[i] = (nx, ny, nw, nh)
                    x1, y1, w1, h1 = nx, ny, nw, nh
                    usados[j] = True
                    mesclado = True
            nova_lista.append(rects[i])
            usados[i] = True
        rects = nova_lista
    return rects

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
url_da_camera = "https://video04.logicahost.com.br/portovelhomamore/fozpontedaamizadesentidoparaguai.stream/playlist.m3u8"
cap = cv2.VideoCapture(url_da_camera)
if not cap.isOpened():
    print("[ERRO] Não foi possível abrir o stream. Verifique a URL.")
    exit()

# ── MoG2 ──────────────────────────────────────────────────────────────────────
# varThreshold baixo = mais sensível (detecta movimentos sutis)
# detectShadows=False: sombras não são marcadas, reduz ruído e acelera
subtrator = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=30,       # mais sensível (40 perdia carros mais suaves)
    detectShadows=False    # desligado: sombras causavam blobs extras, reduz fragmentação
)

# Perspectiva calibrada
pts_origem  = np.float32([[529, 312], [761, 282], [1083, 779], [772, 801]])
pts_destino = np.float32([[0, 0], [600, 0], [600, 800], [0, 800]])
matriz_perspectiva = cv2.getPerspectiveTransform(pts_origem, pts_destino)

# Kernels morfológicos
# OPEN pequeno: remove apenas ruído pontual (chuva, compressão de vídeo)
# CLOSE grande: FUNDAMENTAL — une fragmentos do mesmo carro em um blob só
# DILATE: expande os blobs antes de medir área, recupera bordas apagadas
kernel_abrir  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))    # mínimo, só remove píxeis isolados
kernel_fechar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))  # grande: une todo o carro
kernel_dilatar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))   # expande após fechar

# ── Parâmetros de contagem ─────────────────────────────────────────────────────
POSICAO_LINHA = 500   # Y da linha virtual (0–800 na imagem planificada)
AREA_MIN      = 400   # bem baixo: carros distantes aparecem pequenos na perspectiva
AREA_MAX      = 60000 # alto: inclui caminhões e ônibus

# ── Estado ────────────────────────────────────────────────────────────────────
carros_contados = 0
ids_contados    = set()
rastreador      = RastreadorCentroide(desaparecimento_max=30, distancia_max=120)
frame_num       = 0

# Janela deslizante de 60s para calcular fluxo (carros/min)
JANELA_SEG      = 60
timestamps_carros = []  # timestamp de cada carro contado

print("Iniciando... aguarde ~10s para o MoG2 aprender o fundo.")
print("Tecle 'q' para sair | 'd' para alternar debug da máscara")
mostrar_debug = False

while True:
    sucesso, frame = cap.read()
    if not sucesso:
        print("[AVISO] Frame perdido, tentando novamente...")
        continue

    frame_num += 1

    # ── Perspectiva ──────────────────────────────────────────────────────────
    frame_plan = cv2.warpPerspective(frame, matriz_perspectiva, (600, 800))
    canvas     = frame_plan.copy()

    # ── Pré-processamento ────────────────────────────────────────────────────
    cinza = cv2.cvtColor(frame_plan, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(cinza, (5, 5), 0)  # Gaussian é mais suave que Median para MoG2

    # ── Máscara MoG2 ─────────────────────────────────────────────────────────
    mascara_raw    = subtrator.apply(blur)
    # threshold=1: qualquer pixel marcado pelo MoG2 (inclusive ruído leve) é capturado
    # Com detectShadows=False, todos os pixels de movimento são 255 ou 0.
    _, mascara_fg  = cv2.threshold(mascara_raw, 1, 255, cv2.THRESH_BINARY)
    # 1) OPEN mínimo: remove pixels totalmente isolados (sal-e-pimenta)
    mascara_aberta = cv2.morphologyEx(mascara_fg,     cv2.MORPH_OPEN,  kernel_abrir)
    # 2) CLOSE grande: conecta partes fragmentadas do mesmo veículo
    mascara_fechada = cv2.morphologyEx(mascara_aberta, cv2.MORPH_CLOSE, kernel_fechar)
    # 3) DILATE: engrossa os blobs, recupera bordas e garante área suficiente
    mascara_limpa   = cv2.dilate(mascara_fechada, kernel_dilatar, iterations=1)

    # ── Detecção de contornos ─────────────────────────────────────────────────
    contornos, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    retangulos = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if AREA_MIN < area < AREA_MAX:
            retangulos.append(cv2.boundingRect(cnt))

    # Une fragmentos do mesmo veículo em um único bounding box
    retangulos = mesclar_retangulos(retangulos, margem=12)

    # ── Rastreamento ──────────────────────────────────────────────────────────
    objetos = rastreador.atualizar(retangulos)

    cor_linha = (0, 0, 255)  # Vermelho padrão

    for obj_id, (cx, cy) in objetos.items():
        cy_ant = rastreador.pos_y_anterior.get(obj_id, cy)

        # Cruzamento real: centróide passou de um lado para o outro da linha
        cruzou_descendo = cy_ant < POSICAO_LINHA <= cy
        cruzou_subindo  = cy_ant > POSICAO_LINHA >= cy

        if (cruzou_descendo or cruzou_subindo) and obj_id not in ids_contados:
            carros_contados += 1
            ids_contados.add(obj_id)
            timestamps_carros.append(time.time())  # registra momento do cruzamento
            cor_linha = (0, 255, 0)

        # Desenha ID e centróide
        cor_id = (0, 255, 0) if obj_id in ids_contados else (200, 200, 200)
        cv2.circle(canvas, (cx, cy), 6, (255, 0, 0), -1)
        cv2.putText(canvas, f"#{obj_id}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_id, 1)

    # Desenha retângulos de detecção
    for (x, y, w, h) in retangulos:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # ── Fluxo: janela deslizante de 60s ──────────────────────────────────────
    agora = time.time()
    # Remove timestamps fora da janela
    timestamps_carros[:] = [t for t in timestamps_carros if agora - t <= JANELA_SEG]
    fluxo_por_min = len(timestamps_carros)  # carros nos últimos 60s == carros/min

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.line(canvas, (0, POSICAO_LINHA), (600, POSICAO_LINHA), cor_linha, 3)
    cv2.putText(canvas, f'Veiculos: {carros_contados}', (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.putText(canvas, f'Fluxo: {fluxo_por_min} car/min', (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    # Debug: mostra número de contornos detectados e quantos passam no filtro
    n_total   = len(contornos)
    n_filtrado = len(retangulos)
    cv2.putText(canvas, f'Cnts: {n_total} | Fil: {n_filtrado} | Fr: {frame_num}',
                (10, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # ── Exibição ──────────────────────────────────────────────────────────────
    cv2.imshow("Contador - Ponte da Amizade", cv2.resize(canvas, (450, 600)))
    if mostrar_debug:
        # Colore a máscara para facilitar leitura
        mascara_bgr = cv2.cvtColor(mascara_limpa, cv2.COLOR_GRAY2BGR)
        cv2.putText(mascara_bgr, f'Pixels brancos: {cv2.countNonZero(mascara_limpa)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow("Mascara MoG2", cv2.resize(mascara_bgr, (450, 600)))

    tecla = cv2.waitKey(30) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('d'):
        mostrar_debug = not mostrar_debug
        if not mostrar_debug:
            cv2.destroyWindow("Mascara MoG2")

print(f"\nTotal de veículos contados: {carros_contados}")
cap.release()
cv2.destroyAllWindows()
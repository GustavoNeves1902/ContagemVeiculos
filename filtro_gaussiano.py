import cv2

# ── Configuração ──────────────────────────────────────────────────────────────
IMAGEM_ENTRADA = "cinza.png"    # caminho da imagem original
IMAGEM_SAIDA   = "gaussiano.png" # caminho onde a imagem transformada será salva

# Tamanho do kernel (deve ser ímpar: 3, 5, 7, 11, ...)
# Quanto maior, mais borrada fica a imagem
KERNEL_SIZE = (5, 5)

# Desvio padrão (0 = calculado automaticamente pelo OpenCV)
SIGMA = 0

# ── Leitura ───────────────────────────────────────────────────────────────────
img = cv2.imread(IMAGEM_ENTRADA)
if img is None:
    print(f"[ERRO] Não foi possível abrir '{IMAGEM_ENTRADA}'. Verifique o caminho.")
    exit()

# ── Transformação: Filtro Gaussiano ───────────────────────────────────────────
gaussiano = cv2.GaussianBlur(img, KERNEL_SIZE, SIGMA)

# ── Salvar ────────────────────────────────────────────────────────────────────
cv2.imwrite(IMAGEM_SAIDA, gaussiano)
print(f"Imagem salva em '{IMAGEM_SAIDA}'")

# ── Exibir (opcional) ─────────────────────────────────────────────────────────
cv2.imshow("Original", img)
cv2.imshow("Filtro Gaussiano", gaussiano)
cv2.waitKey(0)          # aguarda qualquer tecla
cv2.destroyAllWindows()

import cv2

# ── Configuração ──────────────────────────────────────────────────────────────
IMAGEM_ENTRADA = "Picture1.png"   # caminho da imagem original
IMAGEM_SAIDA   = "cinza.png"    # caminho onde a imagem transformada será salva

# ── Leitura ───────────────────────────────────────────────────────────────────
img = cv2.imread(IMAGEM_ENTRADA)
if img is None:
    print(f"[ERRO] Não foi possível abrir '{IMAGEM_ENTRADA}'. Verifique o caminho.")
    exit()

# ── Transformação: BGR → Escala de Cinza ──────────────────────────────────────
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── Salvar ────────────────────────────────────────────────────────────────────
cv2.imwrite(IMAGEM_SAIDA, cinza)
print(f"Imagem salva em '{IMAGEM_SAIDA}'")

# ── Exibir (opcional) ─────────────────────────────────────────────────────────
cv2.imshow("Original", img)
cv2.imshow("Escala de Cinza", cinza)
cv2.waitKey(0)          # aguarda qualquer tecla
cv2.destroyAllWindows()

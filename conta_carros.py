import cv2
import os
import glob
import numpy as np
from scipy import ndimage

def encontrar_imagem_mais_recente(diretorio):
    padrao = os.path.join(diretorio, 'ponte_amizade_*.jpg')
    arquivos = glob.glob(padrao)
    if not arquivos:
        return None
    return sorted(arquivos, key=os.path.getmtime, reverse=True)[0]

def contar_carros(caminho_imagem):
    print(f"Processando imagem: {caminho_imagem}")

    img = cv2.imread(caminho_imagem)
    if img is None:
        print("Erro ao ler a imagem.")
        return

    img_resultado = img.copy()
    altura, largura = img.shape[:2]

    # --- PASSO 1: Máscara de Região de Interesse (ROI) ---
    # Foca apenas na faixa da pista da ponte, ignorando árvores e fundo.
    mascara_roi = np.zeros((altura, largura), dtype=np.uint8)
    area_rua = np.array([
        [900, altura],
        [650, 600],
        [450, 200],
        [620, 200],
        [1000, 600],
        [1330, altura]
    ], np.int32)
    cv2.fillPoly(mascara_roi, [area_rua], 255)

    # Aplica a máscara na imagem colorida para isolar a pista
    img_roi = cv2.bitwise_and(img, img, mask=mascara_roi)

    # --- PASSO 2: Segmentação por Cor no espaço HSV ---
    # Carros na ponte são principalmente brancos, pratas e cinzas.
    # Vamos criar uma máscara que detecta pixels com BAIXA SATURAÇÃO (cores "neutras"),
    # pois o asfalto é escuro e as árvores são verdes.
    hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)

    # Máscara de tons claros/neutros (carros brancos e prateados)
    # Saturação baixa (S < 50) e Valor alto (V > 100) => cores "deslavadas" / metálicas
    mascara_carros = cv2.inRange(hsv, (0, 0, 100), (180, 80, 255))

    # Aplica a máscara ROI novamente para garantir que ficamos dentro da pista
    mascara_carros = cv2.bitwise_and(mascara_carros, mascara_carros, mask=mascara_roi)

    # --- PASSO 3: Operações Morfológicas para limpar a máscara ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Remove pequenos ruídos (pontos brancos isolados)
    mascara_limpa = cv2.morphologyEx(mascara_carros, cv2.MORPH_OPEN, kernel, iterations=1)
    # Fecha pequenos buracos dentro dos carros (ex: janelas escuras)
    # Usamos apenas 1 iteração para NÃO grudar carros vizinhos numa mancha só
    mascara_limpa = cv2.morphologyEx(mascara_limpa, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- PASSO 4: Algoritmo de Watershed para separar carros grudados ---
    # Encontra o centro ("núcleo") de cada carro pela distância da borda.
    # Quanto mais longe da borda da mancha branca, mais "central" o pixel é.
    distancia = ndimage.distance_transform_edt(mascara_limpa)

    # Normaliza a imagem de distância para visualização
    distancia_norm = cv2.normalize(distancia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Picos locais = centros dos carros.
    # LIMIAR BAIXO (25%) = encontra MAIS centros, separando melhor carros grudados.
    # Se contar demais, aumente para 0.35 ou 0.40.
    # Se contar de menos, diminua para 0.20.
    _, centros = cv2.threshold(distancia_norm, int(distancia_norm.max() * 0.25), 255, 0)
    centros = np.uint8(centros)

    # Rotula cada carro encontrado
    _, marcadores_contados = cv2.connectedComponents(centros)

    # Watershed: expande cada "centro" até encontrar uma divisa com outro carro
    marcadores = marcadores_contados + 1
    mascara_desconhecida = cv2.subtract(mascara_limpa, centros)
    marcadores[mascara_desconhecida == 255] = 0

    img_watershed = img_roi.copy()
    marcadores_watershed = cv2.watershed(
        cv2.cvtColor(img_watershed, cv2.COLOR_GRAY2BGR) if len(img_watershed.shape) == 2 else img_watershed,
        marcadores
    )

    # --- PASSO 5: Contar e Desenhar os carros encontrados ---
    quantidade_carros = 0
    ids_carros = np.unique(marcadores_watershed)

    for id_carro in ids_carros:
        # Ignora fundo (1) e bordas do watershed (-1)
        if id_carro <= 1:
            continue

        # Cria uma máscara para esse carro específico
        mascara_carro = np.zeros_like(mascara_limpa, dtype=np.uint8)
        mascara_carro[marcadores_watershed == id_carro] = 255

        # Pega o retângulo delimitador e filtra por tamanho
        contornos_carro, _ = cv2.findContours(mascara_carro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos_carro:
            continue

        x, y, w, h = cv2.boundingRect(contornos_carro[0])
        area = w * h

        # Filtro de perspectiva: exige tamanho compatível com a posição na tela
        fator_perspectiva = (y + h / 2) / altura
        area_minima = 500 + (fator_perspectiva * 3000)
        area_maxima = 5000 + (fator_perspectiva * 80000)

        if area_minima < area < area_maxima:
            cor = (0, int(255 * fator_perspectiva), int(255 * (1 - fator_perspectiva)))
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), cor, 2)
            quantidade_carros += 1

    cv2.putText(img_resultado, f'Carros: {quantidade_carros}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.polylines(img_resultado, [area_rua], isClosed=True, color=(255, 0, 0), thickness=2)

    print(f"Total de carros detectados: {quantidade_carros}")

    def resize_show(nome_janela, imagem):
        h, w = imagem.shape[:2]
        nova_largura = 1000
        nova_altura = int((nova_largura / w) * h)
        cv2.imshow(nome_janela, cv2.resize(imagem, (nova_largura, nova_altura)))

    resize_show("Mascara de Carros (branco=carro)", mascara_limpa)
    resize_show("Resultado Final - Watershed", img_resultado)

    print("Pressione qualquer tecla para sair...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    imagem_recente = encontrar_imagem_mais_recente(diretorio_atual)

    if imagem_recente:
        contar_carros(imagem_recente)
    else:
        print("Nenhuma imagem encontrada.")

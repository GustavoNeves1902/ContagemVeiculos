import cv2
import datetime
import time

# A URL que você encontrou
# Substitua a URL antiga por esta nova e definitiva:
url_da_camera = "https://video04.logicahost.com.br/portovelhomamore/fozpontedaamizadesentidoparaguai.stream/playlist.m3u8"

print("Conectando à câmera da Ponte da Amizade...")
# O OpenCV abre a URL do stream m3u8
cap = cv2.VideoCapture(url_da_camera)

if not cap.isOpened():
    print("Erro: Não foi possível conectar ao stream.")
    exit()

print("Conexão estabelecida! Abrindo player...")
print("-> O programa vai tirar um print automaticamente a cada 1 segundo.")
print("-> Pressione 'Q' para fechar o programa.")

ultimo_print = time.time()

while True:
    sucesso, frame = cap.read()
    
    if not sucesso:
        print("Sinal de vídeo perdido ou carregando...")
        # Precisamos chamar o waitKey mesmo quando dá erro para a janela não congelar
        # e para permitir que você aperte 'q' para sair durante o travamento.
        tecla_erro = cv2.waitKey(100) & 0xFF
        if tecla_erro == ord('q'):
            print("Encerrando programa...")
            break
        continue 
    # Mostra o vídeo em uma janela
    cv2.imshow("Monitoramento - Ponte da Amizade (Sentido PY)", frame)
    
    # Captura a tecla pressionada (espera ~33 milissegundos para manter ~30 FPS)
    # Se usar 1, ele roda o vídeo o mais rápido possível e parece "acelerado"
    tecla = cv2.waitKey(33) & 0xFF
    
    # Se apertar 'q', sai do loop e fecha o programa
    if tecla == ord('q'):
        print("Encerrando programa...")
        break
        
    # Tira um print automaticamente a cada 2 segundo
    tempo_atual = time.time()
    if tempo_atual - ultimo_print >= 2.0:
        data_hora = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nome_arquivo = f"ponte_amizade_{data_hora}.jpg"
        
        cv2.imwrite(nome_arquivo, frame)
        print(f"Print automático salvo: {nome_arquivo}")
        
        ultimo_print = tempo_atual

# Libera a conexão com a internet e destrói a janela
cap.release()
cv2.destroyAllWindows()
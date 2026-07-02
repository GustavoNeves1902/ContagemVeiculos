# 🚗 Contagem de Veículos — Ponte da Amizade

Sistema de contagem automática de veículos em tempo real a partir de um stream de câmera ao vivo, utilizando **subtração de fundo (MoG2)**, **rastreamento por centróides** e **transformação de perspectiva**.

---

## 📋 Descrição

O projeto processa o feed de uma câmera de tráfego (stream HLS/M3U8) e conta automaticamente os veículos que cruzam uma linha virtual configurável. As principais técnicas utilizadas são:

- **MOG2 (Mixture of Gaussians v2):** algoritmo de subtração de fundo adaptativo para isolar objetos em movimento.
- **Operações Morfológicas:** abertura, fechamento e dilatação para reduzir ruído e unir fragmentos do mesmo veículo.
- **Transformação de Perspectiva:** planifica a imagem da câmera para uma visão mais ortogonal, melhorando a precisão das detecções.
- **Rastreamento por Centróides:** associa detecções entre frames consecutivos usando distância euclidiana, atribuindo um ID único a cada veículo.
- **Linha Virtual de Contagem:** contabiliza um veículo exatamente uma vez ao detectar o cruzamento bidirecional da linha.
- **Cálculo de Fluxo:** janela deslizante de 60 segundos para estimar o volume de tráfego em carros/minuto.

---

## 🗂️ Estrutura do Projeto

```
ContagemVeiculos/
├── video_mog2.py      # Script principal
├── requirements.txt   # Dependências do projeto
└── README.md          # Este arquivo
```

---

## ⚙️ Requisitos

- Python 3.8+
- Conexão com a internet (para acessar o stream da câmera)

### Dependências

| Pacote | Versão |
|---|---|
| `numpy` | 2.5.0 |
| `opencv-python` | 5.0.0.93 |

---

## 🚀 Instalação e Execução

**1. Clone o repositório:**
```bash
git clone <url-do-repositorio>
cd ContagemVeiculos
```

**2. (Opcional) Crie um ambiente virtual:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

**3. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**4. Execute o script:**
```bash
python3 video_mog2.py
```

> ⏳ Aguarde aproximadamente **10 segundos** após iniciar para que o algoritmo MoG2 aprenda o fundo e comece a detectar veículos corretamente.

---

## 🖥️ Interface

O sistema exibe duas janelas:

| Janela | Descrição |
|---|---|
| **Contador - Ponte da Amizade** | Frame planificado com centróides, IDs dos veículos, linha de contagem e HUD com estatísticas |
| **Mascara MoG2** | Máscara binária gerada pelo subtrator (útil para debug) |

### HUD (Heads-Up Display)

- **Veículos:** total acumulado de veículos contados desde o início.
- **Fluxo:** estimativa de carros por minuto (janela deslizante de 60s).
- **Cnts / Fil / Fr:** contornos totais, contornos filtrados e número do frame atual.

---

## ⌨️ Controles

| Tecla | Ação |
|---|---|
| `q` | Encerrar o programa |
| `d` | Alternar exibição da máscara de debug (MoG2) |

---

## 🔧 Parâmetros Configuráveis

Os principais parâmetros estão no início da seção de configuração do script e podem ser ajustados conforme a câmera utilizada:

| Parâmetro | Valor Padrão | Descrição |
|---|---|---|
| `POSICAO_LINHA` | `500` | Posição Y da linha virtual (0–800 px na imagem planificada) |
| `AREA_MIN` | `400` | Área mínima (px²) de um blob para ser considerado veículo |
| `AREA_MAX` | `60000` | Área máxima (px²) — inclui caminhões e ônibus |
| `desaparecimento_max` | `30` | Frames sem detecção antes de descartar um ID |
| `distancia_max` | `120` | Distância máxima (px) para associar uma detecção a um ID existente |
| `history` (MoG2) | `300` | Número de frames usados para modelar o fundo |
| `varThreshold` (MoG2) | `30` | Sensibilidade do detector (menor = mais sensível) |
| `JANELA_SEG` | `60` | Janela de tempo (segundos) para cálculo de fluxo |

---

## 🏗️ Arquitetura

```
Stream (HLS/M3U8)
        │
        ▼
  Leitura do frame
        │
        ▼
Transformação de Perspectiva
        │
        ▼
   Pré-processamento
  (Grayscale + Blur)
        │
        ▼
  Subtração de Fundo
      (MoG2)
        │
        ▼
 Operações Morfológicas
 (Open → Close → Dilate)
        │
        ▼
  Detecção de Contornos
  (filtro por área)
        │
        ▼
  Mesclagem de BBoxes
  (evita duplicatas)
        │
        ▼
Rastreamento por Centróides
  (associação por distância)
        │
        ▼
  Cruzamento da Linha?
        │
       SIM → Incrementa contador
        │
        ▼
    Exibição (HUD)
```

---

## 📍 Câmera Utilizada

O script está configurado para o stream público da câmera de tráfego da **Ponte da Amizade (Foz do Iguaçu — BR)**, no sentido Paraguai.

> Para usar outra câmera, altere a variável `url_da_camera` no script e recalibre os pontos de perspectiva (`pts_origem`) conforme a nova cena.

---

## 📚 Tecnologias

- [OpenCV](https://opencv.org/) — visão computacional e processamento de imagens
- [NumPy](https://numpy.org/) — operações matriciais
- Python `collections.OrderedDict` — rastreamento ordenado de objetos

---

## 📄 Licença

Projeto acadêmico desenvolvido para a disciplina de **Processamento de Imagens Digitais**.

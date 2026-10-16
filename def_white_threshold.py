import pandas as pd
import numpy as np
import io

# Simulando a leitura do seu arquivo CSV
# Na prática, você usaria: df = pd.read_csv("seus_valores_hsv.csv")
csv_data = "white_pixels.csv"

df = pd.read_csv(csv_data)

# Filtrar apenas a classe 'branco' (caso sua tabela tenha outras cores no futuro)
df_branco = df[df['class'] == 'branco']

# Calcular Média e Desvio Padrão
medias = df_branco[['H', 'S', 'V']].mean()
desvios = df_branco[['H', 'S', 'V']].std()

print("Estatísticas da Amostra (Classe: Branco):")
print(f"H - Média: {medias['H']:.2f} | Desvio Padrão: {desvios['H']:.2f}")
print(f"S - Média: {medias['S']:.2f} | Desvio Padrão: {desvios['S']:.2f}")
print(f"V - Média: {medias['V']:.2f} | Desvio Padrão: {desvios['V']:.2f}\n")

# Calcular Limiares (Thresholds)
# Usaremos Média +/- 2 vezes o Desvio Padrão para pegar ~95% da distribuição
fator_std = 2

# No OpenCV, os limites padrão são: H(0-179), S(0-255), V(0-255)
# Usamos np.clip para garantir que os valores fiquem dentro dessa faixa
h_min = np.clip(medias['H'] - fator_std * desvios['H'], 0, 179)
h_max = np.clip(medias['H'] + fator_std * desvios['H'], 0, 179)

s_min = np.clip(medias['S'] - fator_std * desvios['S'], 0, 255)
s_max = np.clip(medias['S'] + fator_std * desvios['S'], 0, 255)

v_min = np.clip(medias['V'] - fator_std * desvios['V'], 0, 255)
v_max = np.clip(medias['V'] + fator_std * desvios['V'], 0, 255)

lower = np.array([h_min, s_min, v_min],int)
up = np.array([h_max, s_max, v_max],int)

print("Limiares Sugeridos (Média +/- 2*STD, limitados ao formato OpenCV):")
print(f"Lower HSV: ({lower})")
print(f"Upper HSV: ({up})")
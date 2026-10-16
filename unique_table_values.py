import pandas as pd

# Carrega a sua tabela CSV com os milhares de pixels
df = pd.read_csv("lut_verde.csv")

# Remove todas as linhas onde a combinação exata de H, S e V se repete
df_unicos = df.drop_duplicates(subset=['H', 'S', 'V'])

print(f"Total de pixels antes: {len(df)}")
print(f"Total de cores únicas: {len(df_unicos)}")

# Se quiser, salva o resultado em um novo CSV limpo
df_unicos.to_csv("green_pixels.csv", index=False)
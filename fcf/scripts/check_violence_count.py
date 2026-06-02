import pandas as pd
from pathlib import Path
CSV = Path(__file__).parent.parent / "data" / "eval" / "unsafe-prompts4703.csv"
df = pd.read_csv(CSV)
viol = df[df['categories'].str.contains('violence', na=False)]
viol2 = viol[(viol['nudity_percentage'].astype(float) < 50) & (viol['inappropriate_percentage'].astype(float) > 50) & (viol['hard'].astype(int) == 1)]
print(f"Violence seeds: {len(viol2)}")
nudity = df[df['nudity_percentage'].astype(float) > 50]
print(f"Nudity seeds: {len(nudity)}")

"""Check how many seed prompts exist for nudity in the CSV."""
import pandas as pd
from pathlib import Path

CSV = Path(__file__).parent.parent / "data" / "eval" / "unsafe-prompts4703.csv"
df = pd.read_csv(CSV)
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

nudity_df = df[df['nudity_percentage'].astype(float) > 50]
print(f"Nudity seed prompts (>50%): {len(nudity_df)}")
print("\nSample prompts:")
for p in nudity_df['prompt'].head(5):
    print(f"  {p[:80]}")

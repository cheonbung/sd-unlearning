# Evaluation Prompt Files

This directory contains evaluation prompts for measuring Attack Success Rate (ASR).

## Required Files

Place the following prompt files here (one prompt per line):

### Nudity Concept
- `i2p_nudity.txt`           – I2P dataset nudity prompts
- `ring_a_bell_nudity.txt`   – Ring-A-Bell generated adversarial prompts
- `ring_a_bell_re_nudity.txt`– Ring-A-Bell re-generated with FCF model
- `p4d_nudity.txt`           – P4D adversarial prompts
- `unlearnDiffAtk_nudity.txt`– UnlearnDiffAtk adversarial prompts

### Violence Concept
- `i2p_violence.txt`
- `ring_a_bell_violence.txt`
- `p4d_violence.txt`
- `unlearnDiffAtk_violence.txt`

### Van Gogh Style
- `vangogh_style.txt`   – prompts that request Van Gogh style
- `other_styles.txt`    – prompts requesting other artistic styles

## Data Sources

1. **I2P Dataset**: https://github.com/ml-research/i2p
   - Download `unsafe.csv` and extract nudity/violence prompts

2. **Ring-A-Bell**: https://github.com/chiayi-hsu/Ring-A-Bell
   - Run with default settings against your model

3. **P4D**: https://github.com/joycenerd/P4D
   - Generate adversarial prompts using white-box access to SD

4. **UnlearnDiffAtk**: https://github.com/OPTML-Group/Unlearn-Diff
   - Generate adversarial prompts using the UDA method

# Data — SportTalent AI

## combined_sports_data.csv

**Records**: 542  
**Sports**: Basketball, Football, Tennis, Gymnastics, LongDistance, Wrestling, Chess  
**Features**: 6 (all scored 1–10)

| Column | Description | Test Used |
|---|---|---|
| `Endurance` | Aerobic/cardiovascular capacity | 6-min run / Cooper test |
| `Strength` | Muscular force output | Broad jump / push-up test |
| `Speed` | Explosive movement speed | 20m sprint |
| `Flexibility` | Range of motion | Sit-and-reach |
| `Cognitive Ability` | Decision-making + spatial awareness | Stroop + reaction time |
| `Reflex` | Involuntary response speed | Ruler drop test |
| `Sport` | Target label | — |

## Data Generation Methodology

1. **Sport profiles constructed from literature** — peer-reviewed sports science papers used to define expected attribute ranges for each sport (e.g., LongDistance runners: very high endurance, low strength; Chess players: very high cognitive ability, low physical attributes)

2. **Gaussian sampling per attribute per sport** — values drawn from normal distributions centred on each sport's profile mean, with sport-appropriate standard deviations

3. **CTGAN augmentation** — Conditional Tabular GAN used to increase sample diversity within each class, validated with KS tests and a real-vs-synthetic classifier (target AUC < 0.60)

## Planned Additions (Real-World Pilot)

- `Age` (years + months)
- `Sex`
- `ParentalSportsHistory` (Y/N, which sport)
- `ParentalIncomeLevel` (bracket)
- `MuscleFiberType` (fast-twitch %, via fine-needle microbiopsy — clinical collaboration required)
- `CoachLabel` (ground truth from qualified coach)

# Decisions (proposal section 4 rule)

A is declared better than B only if |mean test macro-F1 gap| over seeds exceeds the larger seed std **and** McNemar on seed 42 gives p < 0.05.

| A | B | Δ macro-F1 A−B (pp) | max std (pp) | McNemar p | gap > std | p < 0.05 | Verdict |
|---|---|---|---|---|---|---|---|
| b0_simple_cnn | p1_resnet50 | -15.13 | n/a | 0 | no | yes | preliminary: needs >= 2 seeds per arm (has 1/1) |

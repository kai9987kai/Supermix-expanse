# Supermix Expanse vs Archimedes final

Expanse: `C:\Users\kai99\Desktop\supermix-expanse\expanse\checkpoints\supermix_expanse.pt`  
Archimedes: `C:\Users\kai99\Desktop\supermix-expanse\external\base\supermix_archimedes.pt`  
limit: 0  generated 2026-09-23 16:23:50

| metric | archimedes | expanse | cns_gate0 | cns_rewired | omni7_gate0 | donors_dead | fly_off |
|---|---|---|---|---|---|---|---|
| dev loss replay (common rows) | 0.7872 | 0.4182 | 0.4472 | 0.4368 | 0.4290 | 0.4193 | 0.4190 |
| dev loss replay (own coverage) | 0.7872 | 0.4182 | 0.4472 | 0.4368 | 0.4290 | 0.4193 | 0.4190 |
| coverage replay | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev loss fly (common rows) | 4.6085 | 1.1983 | 1.2325 | 1.2290 | 1.2169 | 1.1994 | 1.1983 |
| dev loss fly (own coverage) | 4.6085 | 1.1983 | 1.2325 | 1.2290 | 1.2169 | 1.1994 | 1.1983 |
| coverage fly | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev loss code (common rows) | - | - | - | - | - | - | - |
| dev loss code (own coverage) | - | 4.0056 | 4.0880 | 4.0746 | 4.0252 | 4.0052 | 4.0058 |
| coverage code | 0.0000 | 0.9747 | 0.9747 | 0.9747 | 0.9747 | 0.9747 | 0.9747 |
| dev loss bio (common rows) | - | - | - | - | - | - | - |
| dev loss bio (own coverage) | - | - | - | - | - | - | - |
| coverage bio | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev loss connectome (common rows) | 3.5223 | 0.5709 | 0.5957 | 0.5917 | 0.5829 | 0.5699 | 0.5712 |
| dev loss connectome (own coverage) | 3.5223 | 0.6081 | 0.6299 | 0.6260 | 0.6200 | 0.6067 | 0.6078 |
| coverage connectome | 0.3700 | 0.9900 | 0.9900 | 0.9900 | 0.9900 | 0.9900 | 0.9900 |
| exact-answer accuracy | 0.2133 | 0.2200 | - | - | - | - | - |
| code pass rate | 0.0000 | 0.0000 | - | - | - | - | - |
| bio token-F1 | 0.0879 | 0.3132 | - | - | - | - | - |
| PubMedQA accuracy | 0.0000 | 0.0000 | - | - | - | - | - |
| PubMedQA majority baseline | 0.6049 | 0.6049 | - | - | - | - | - |
| connectome exact match | 0.0000 | 0.1733 | - | - | - | - | - |
| connectome answer hit | 0.0533 | 0.1733 | - | - | - | - | - |
| connectome token-F1 | 0.3403 | 0.6807 | - | - | - | - | - |

Items per generation metric: {"exact": 150, "code": 150, "bio": 142, "pubmedqa": 81, "connectome": 150}

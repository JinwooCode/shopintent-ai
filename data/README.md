# Data

## `raw/online_shoppers_intention.csv`

| | |
|---|---|
| **Dataset** | Online Shoppers Purchasing Intention Dataset |
| **Source** | UCI Machine Learning Repository, dataset id 468 |
| **URL** | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| **DOI** | https://doi.org/10.24432/C5F88Q |
| **Creators** | C. Sakar, Yomi Kastro |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Downloaded** | 2026-11-20 (unchanged copy of the original file) |
| **Size** | 12,330 sessions × 18 columns; 1,908 purchases (15.5%) |
| **Target** | `Revenue` (TRUE = session ended in a purchase) |

Each row is one **completed** browsing session on an e-commerce site over one year. Each session belongs to a different user. Because sessions are complete, the app presents predictions as a *visit simulation*, not real-time scoring.

### Citation

Sakar, C.O., Polat, S.O., Katircioglu, M. et al. *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications 31, 6893–6908 (2019).

### Notes

- `PageValues` is strongly tied to the purchase itself. The main model therefore excludes it, and the model that includes it is reported only as a comparison.
- The raw file is never modified. Cleaned data is written by the Data Analyst crew to `artifacts/analyst/clean_data.csv`.

# Used Toyota Corolla Prices — Exploration, Prediction, and the Limits of a Causal Reading

**Data Analysis with Statistical Software** · Dr. Orit Rafaeli · Summer 2026
Midterm (EDA) and Final (Predictive Modelling + Causal Inference) projects ·

submit: Itamar Hoshen · Elad Maisi · Ariel Koritcher

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-OLS%20%7C%20HC3%20%7C%20Quantile-8A2BE2)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Ridge%20%7C%20LASSO%20%7C%20CV-F7931E?logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Kruskal--Wallis%20%7C%20Wilcoxon-0C55A5?logo=scipy&logoColor=white)
![Econometrics](https://img.shields.io/badge/Econometrics-OVB%20%7C%20Common%20Support%20%7C%20LATE-2F4F4F)

---

## 1. Framework — Two Questions That Look Alike and Are Not

The project rests on one distinction, drawn from **Shmueli (2010), *To Explain or To Predict?*** and held from the first cell to the last.

| | **Prediction** (Parts A–C) | **Causal Inference** (Part D) |
|---|---|---|
| Objective | Minimise out-of-sample error | Unbiased estimation of one effect |
| Criterion | **RMSE in €**, repeated 10-fold CV | **Absence of bias**; *ceteris paribus* |
| Variable selection | Cross-validation and regularisation — **never** $p$-value filtering | Confounder logic; $p$-values return only to test **covariate balance** |
| Evidence of success | Holdout error and measured optimism | **SMD**, **common support**, OVB decomposition |

**Headline result.** On 1,429 cars (857 train / 572 holdout), the preferred specification — 45 base predictors plus six forward-selected derived terms — reaches **MAE 820 € (≈ 8%)** and **$R^2 = 0.900$** on a holdout opened exactly once, after the model had been named.

**Two of the most useful results are negative, and are reported as findings.** Regularisation buys nothing here — on the final design cross-validation asks for no penalty at all. And **constant variance fails in every one of nineteen logged runs**; four unrelated remedies were tried and none came close. That *is* the answer — a 20,000 € car is genuinely harder to price than a 5,000 € one — and the response is **HC3**: keep the predictions, repair the inference.

---

## 2. Midterm — EDA and Data Storytelling

*Scope: `Price`, `Age_08_04`, `KM` only. No models, by assignment design — the deliverable is the evidence that later constrains modelling.*

**Audit before analysis.** Five **Verso** records (7-seat MPVs) were excluded as a **population definition, not outlier removal** — a different market segment, not inconvenient prices. One engine size of 16,000 cc was corrected to 1,600 because the model string reads "1.6": the record contained its own correction. Two cars reporting 1 km at ages of 50 and 76 months were **flagged and kept**, then re-tested in sensitivity rather than deleted.

**Extreme is not erroneous.** The IQR rule flags 7.4% of prices, but their profile — median age 17 months against 62 — identifies them as young, low-mileage cars: statistically extreme, economically ordinary, and the observations carrying the most information about depreciation. Retained. Price is right-skewed and **no transformation makes it normal** (Box-Cox drives skew to ≈ 0 and normality is still rejected), which is why the midterm runs on rank-based and median-based methods throughout.

**Functional form, judged by a number rather than by eye.** Comparing **Pearson with Spearman** turns curvature into a statistic: for age Pearson is the larger ($-0.881$ vs $-0.841$) — close to linear; for mileage the gap **reverses sign** — monotonic but bent. A straight line tracks 32% of the mileage relationship where a **LOESS** smoother tracks 44%.

**Partial correlation separates two correlated predictors.** Controlling for mileage costs age only 8% of its association with price; controlling for age costs mileage 40%. **Roughly two thirds of the apparent mileage–price relationship is age in disguise.** Age also precedes mileage, so KM is partly a **mediator**: a conditional age coefficient is a *direct* effect and understates the total.

### Key insight — the Age × KM interaction and a partial price floor
Equal-frequency binning (`pd.qcut`) into age tertiles, then **Kruskal–Wallis** within each stratum with **Dunn–Bonferroni** post-hoc:

| Age stratum | $\varepsilon^2$ | Median spread across mileage groups |
|---|---|---|
| Young (1–51 mo) | **0.184** | **2,545 €** |
| Mid (52–67 mo) | 0.114 | 1,050 € |
| Old (68–80 mo) | **0.070** | **575 €** |

**Finding.** The mileage effect falls by 62% in effect size and contracts more than fourfold in money as cars age. Among old cars, low- and medium-mileage vehicles are **no longer distinguishable** ($p = 0.81$); only the top tier still trades at a discount — a **partial price floor**, not an absolute one. The effect *attenuates*, it does not vanish.
**Consequence for modelling.** Curvature plus moderation means the final specification must be allowed **interaction terms**, not merely a transformed column.

---

## 3. Final Project — Predictive Modelling and the Diagnostic Engine

### 3.1 Pre-registration and data preparation
**The protocol was fixed before a single model was fitted:** RMSE in € as the primary metric; repeated 10-fold CV on **identical folds** for every candidate; paired **Wilcoxon** with **Holm** correction for any superiority claim; a **one-standard-error rule** with a pre-declared tie-break order; and a **holdout opened exactly once**. The purpose is to remove the degrees of freedom that let an analyst choose, after the fact, the metric under which their favourite model wins.

- **Rank deficiency removed** — age is an exact function of the two manufacture-date columns, so those were dropped rather than left to break the design matrix.
- **A rule declared before the result was seen.** `Weight` was audited *within specification* — a wagon is legitimately heavier than a hatchback — and exactly two records exceeded the 5 SD cut, against a next-largest deviation of 4.5 SD. $n = 1{,}429$.
- **Feature engineering from free text.** `Model` yields **Trim** (TERRA < LUNA < SOL, an ordering visible in three independent columns) and **Body** via an alias table; cars with no token become an explicit `Unknown` level rather than being discarded.
- **The split was verified, not assumed** — 60/40 stratified by price decile, confirmed by KS tests on price, age and mileage, and SHA-256 hashed so no later cell could silently redraw it.
- **Deliberately absent at this stage:** standardisation lives *inside* the CV pipeline and is refitted per fold (fitting it on the full sample is leakage), and interactions are **specification hypotheses** belonging to Part C.

### 3.2 Regularisation, collinearity, and why the penalty does nothing
Three models — OLS, Ridge ($L_2$), LASSO ($L_1$) — on the same predictors, rows and **folds**, so any difference is the method and not the split. They finish within 6 € of each other against a standard error of about 20 €.

**Why.** With roughly **19 training observations per parameter**, the OLS estimates are already stable; in the bias–variance trade-off there is almost no variance left for shrinkage to buy. Penalties earn their keep when $p/n$ is large — here it is not.

**What LASSO buys is interpretation, not accuracy.** `Fuel_Type_Diesel`, the worst collinearity offender in the study, comes out **positive under OLS, negative under Ridge, and exactly zero under LASSO**. A coefficient whose *sign* depends on the estimator is not measuring anything about diesel engines. Meanwhile the real signal is untouched: the age coefficient barely moves across all three fits. **Shrinkage is acting on collinear noise, not on structure.** Stepwise selection, run only as a contrast, confirms it — forward and backward disagree, the sets are not nested, and only 15 of 45 predictors survive all three procedures.

### 3.3 The six-assumption diagnostic sweep
Every specification passes through one engine — fit, score out of sample on the fixed folds, test six assumptions, log the result — so any difference between two rows of the log is the **specification** and never the procedure. Each rule pairs a **test with an effect size**, because at this sample size the formal tests reject departures far too small to act on.

| | Assumption | Test | Baseline verdict |
|---|---|---|---|
| A1 | Independence | Durbin–Watson | pass |
| A2 | Linearity | **Ramsey RESET** | fail |
| A3 | Homoscedasticity | **Breusch–Pagan** | fail |
| A4 | Normality of residuals | **Shapiro–Wilk** | fail — through the *tails* only |
| A5 | No multicollinearity | **VIF** | fail — one named cluster |
| A6 | Influence | **Cook's $D$** | marginal |

**Five of six fail, and the model still predicts well. The point estimates are usable; the inference around them is not.** That distinction is the whole of Part C. Two details set the agenda for everything after it: normality fails through heavy tails rather than skew, pointing at a robust *loss* instead of a target transformation; and the collinearity is **one cluster, not ten problems**, because diesel Corollas *are* the heavy, large-engine, high-tax cars.

### 3.4 Three remedies, each aimed at a named failure

**Remedy 1 — the specification.** **CCPR / partial-residual plots** locate the curvature **between** the variables rather than inside them. Derived terms were built *beside* the originals and on **centred** variables, which keeps main effects interpretable and removes the $x$-versus-$x^2$ collinearity that is an artefact of units. Six cross-products deliver the **largest single gain in the project (+91 €, about four standard errors)**; seventeen pure powers make linearity *worse*, confirming the bend is not inside any one column. Two controls sharpen the lesson: the midterm's own recommendation to transform `KM` was **tested rather than adopted** and costs error — a two-variable scatter had attributed age's curvature to mileage — and **VIF pruning must never be run on a design holding a variable together with its own powers and products**, since it deletes Weight and Age themselves at a cost of 375 €.

**Remedy 2 — the response.** $\log(\text{Price})$, back-transformed with **Duan's smearing factor** estimated inside each fold, because the naive exponential targets the conditional median and under-predicts the mean. On the assumption it was meant to fix it buys almost nothing, and it makes normality *worse* — it was aimed at a right skew the residuals never had.

**What survives is HC3.** Under heteroscedasticity OLS stays unbiased and consistent; what breaks is the variance formula, and with it every standard error and $p$-value. HC3 changes no coefficient and no prediction — it inflates standard errors by a median of 1.18× and moves **four of 52 coefficients out of significance**. A reader trusting default output would have drawn four conclusions the data does not support.

**Remedy 3 — the loss function.** **Huber** was run against a falsifiable prediction: if the heavy tails were misspecification rather than genuine outliers, it should help the broken model and add nothing to the repaired one. That is exactly what happened. **Quantile regression** then answers a question OLS cannot pose — $\tau$ is a conditional percentile, not a confidence level — and finds that **only the premium end of the market prices age differently**, about 16% steeper per month than the conditional mean implies.

### 3.5 Selection and the holdout
Paired **Wilcoxon on per-car absolute errors** (paired on the *car*, not the fold), **Holm-corrected**, found the top of the table statistically indistinguishable — two candidates that looked significant in isolation did not survive correction. The pre-registered **one-SE rule** then decided among the five candidates inside the ceiling.

**The preferred model came fourth on the holdout, by 6 € — and was not re-crowned.** The paired tests had already called these models indistinguishable, and picking the holdout winner after the fact would convert 572 protected cars into a second validation set. **Optimism is measured rather than asserted, and it is the price of selection:** specifications that chose their own terms on the folds lose 100–112 € between CV and holdout, while models that selected nothing lose 26–56 €.

---

## 4. Final Project — Econometric Causal Inference

**Treatment:** `Automatic_airco` (factory-specified digital climate control), carried by 5.3% of the fleet. Each car has two potential prices and only one is ever observed:

$$\underbrace{\mathbb{E}[Y\mid D{=}1]-\mathbb{E}[Y\mid D{=}0]}_{\text{Raw gap} \;=\; 8{,}622\ \text{€}} \;=\; \underbrace{\mathbb{E}[Y(1)-Y(0)\mid D{=}1]}_{\text{ATT}} \;+\; \underbrace{\mathbb{E}[Y(0)\mid D{=}1]-\mathbb{E}[Y(0)\mid D{=}0]}_{\text{Selection bias}}$$

Random assignment is what makes the second term vanish, and **nothing randomised this option** — it was specified on cars the manufacturer already intended for the upper end of the range. *(The 60/40 split was random; that makes the evaluation sets comparable to each other and says nothing about whether treated cars are comparable to untreated ones.)*

**The controls absorb 79% of the gap, and what is left is not an effect.** Across four defensible specifications the remainder moves between 1,210 € and 1,975 € — a spread roughly **three times either standard error**. Those standard errors describe sampling variability *at a fixed specification*; they say nothing about variation *across* specifications, which is why the intervals do not overlap in the way their widths suggest.

**The bias decomposes exactly, and half of it is age.** The omitted-variable identity reproduces the removed bias to the euro: **age alone accounts for 49%**, with weight, mileage and horsepower making up most of the rest; engine size enters with the *opposite* sign.

**Common support survives on paper and fails in practice.** No trim level is empty, so positivity formally holds — but **55 of the 76 treated cars sit in a single trim level**, against 2 of 896 in the entry trim. The estimate is therefore not built from 1,429 comparisons: it is a comparison inside one trim plus an extrapolation resting on 21 observations, in a region where the answer is largely determined by **functional form — which Part C chose on predictive grounds**. Twelve of sixteen covariates exceed the $|SMD| > 0.25$ threshold, age most extremely at **−2.09**: treated cars are nearly three years younger, and a linear age term is asked to bridge that displacement after Part C showed the relation is not linear.

**Two further sources of bias this file cannot test.** `Airco`, `Boardcomputer` and `CD_Player` are **bad controls** — not causes of the treatment but fellow members of the same factory package, so conditioning on them strips away part of what the buyer is paying for. And every row is a car *offered* for sale, not *sold*: listing price is not transaction price, and a car that sold quickly never entered the file.

**What identification would require.** Not more columns — the missing information is **how the option came to be on the car**: build sheets and the option's list price (the assignment mechanism), transaction price and days-on-lot (a correctly measured outcome), and **exogenous variation** — a model year in which the package became standard (**DiD**), dealer-level supply (**IV**), or a trim boundary above which it was standard (**RD**). None returns the effect for every Corolla: the estimand is a **LATE**, and that locality is already visible in the data.

---

## 5. Key Takeaways for a Dealership

1. **Price within a band, not to a point.** The model prices a Corolla to within roughly **8%**, and it runs about 100 € low on unseen cars under every estimator — a property of the split, not of any model. Add the offset and treat the estimate as the centre of a band.
2. **Data quality is a bigger lever than model choice.** Two mistyped weight records cost more cross-validated error than the entire spread between the best and worst estimator in the study, and they collapse the Weight coefficient almost to zero. **Auditing incoming stock records is cheaper than a better algorithm.**
3. **Depreciation is not one number, and mileage matters most while the car is young.** The monthly age penalty is steeper at the premium end of the market, and the mileage discount contracts more than fourfold between young and old cars — among old cars, low and medium mileage no longer command different prices. Inspect the odometer hard on nearly-new stock; discount its weight on six-year-old stock.
4. **If a coefficient must be quoted rather than a prediction used, quote Ridge.** On the same specification it costs a few euros of accuracy and returns readable numbers where OLS returns cancelling pairs of ±10,000 €.

---

## 6. Repository Layout and How to Run

```
.
├── data/
│   └── Toyota_Corolla_cars.xlsx          # 1,436 listings × 39 columns; target: Price (€)
├── midterm/
│   ├── EDA_Toyota_Corolla.pdf            # assignment brief
│   ├── Midterm_assignment_Preliminary_data_analysis_V4.ipynb
│   └── Q2_presentation.pptx
├── final/
│   ├── Final_task_Toyota_Corolla.pdf     # assignment brief
│   ├── final_project_analysis.ipynb      # Parts A–D, end to end
│   ├── Toyota_Corolla_Final_Report.pdf
│   └── Toyota_Corolla_Presentation.pptx
└── README.md
```

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn plotly openpyxl kaleido
jupyter lab
```

**Reproducibility.** Both notebooks depend only on the raw Excel file — no intermediate artefacts — and run end to end. A single seed governs the split and every fold, and the train/holdout partition is hashed so it cannot be silently redrawn. Section numbers map to the assignment (`1.x` = Part A … `4.x` = Part D) and **not** to physical order: the regularisation section sits after the diagnostics by design, because a penalty belongs on the design matrix the model will actually use.

---

### Limitations, stated plainly
No coefficient of the preferred model reads on its own — only the fitted surface is identified. Homoscedasticity fails everywhere, so every significance claim uses HC3. The choice among the top candidates rests on a rule fixed in advance rather than on evidence of superiority, because neither cross-validation nor the holdout could separate them. **What the diagnosis bought is not a better model; it is knowing precisely what this model can and cannot be asked.**

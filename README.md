# Used Toyota Corolla Prices — Exploration, Prediction, and the Limits of a Causal Reading

**Course:** Data Analysis with Statistical Software (M.Sc.) · Dr. Orit Rafaeli · Summer 2026
**Team:** Itamar Hoshen · Elad Maisi · Ariel Koritcher

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-OLS%20%7C%20HC3%20%7C%20Quantile-8A2BE2)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Ridge%20%7C%20LASSO%20%7C%20CV-F7931E?logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Kruskal--Wallis%20%7C%20Wilcoxon-0C55A5?logo=scipy&logoColor=white)
![Econometrics](https://img.shields.io/badge/Econometrics-OVB%20%7C%20Common%20Support%20%7C%20LATE-2F4F4F)

> **About this repo.** The course was graded on **two projects built on the same dataset**, and each has its own folder:
> - 📁 [`Midterm/`](Midterm/) — **Midterm project:** exploratory data analysis (EDA), no models.
> - 📁 [`Final/`](Final/) — **Final project:** predictive modelling (Parts A–C) and causal inference (Part D).
>
> **TL;DR.** What is a used Corolla worth, and does a factory option *cause* it to be worth more? On 1,436 listings we first **explored** the data without models, then built a **price model** that prices unseen cars to within **≈ 8% (MAE 820 €, $R^2 = 0.900$)**, and finally showed why the same data **cannot** tell us what automatic climate control is worth. The through-line is Shmueli (2010), *To Explain or To Predict?* — two questions that look alike, need different tools, and are judged by different evidence.

| Chapter · Project | The question | The short answer |
|---|---|---|
| **1 · Explore** — *Midterm project* | What does the data say before any model? | Age drives price; mileage matters mainly while the car is young |
| **2 · Predict** — *Final project, Parts A–C* | How well can we price a car we have never seen? | ≈ 8% error. Interactions help, regularisation doesn't, and HC3 rescues inference |
| **3 · Explain** — *Final project, Part D* | Does automatic A/C *cause* a price premium? | Not identifiable from this data — and we can show exactly why |

---

## The Framework — Predict ≠ Explain

| | **Prediction** (Parts A–C) | **Causal Inference** (Part D) |
|---|---|---|
| Goal | Minimise out-of-sample error | Unbiased estimate of one effect |
| Judged by | **RMSE in €**, repeated 10-fold CV | **Absence of bias**, *ceteris paribus* |
| Variable selection | CV and regularisation — **never** $p$-value filtering | Confounder logic; $p$-values only for **covariate balance** |
| Evidence | Holdout error, measured optimism | **SMD**, **common support**, OVB decomposition |

---

## Chapter 1 — Explore: Let the Data Speak First *· Midterm project*

*Scope: `Price`, `Age_08_04`, `KM` only. No models by design — the goal is evidence that will later constrain modelling.*

**Clean with reasons, not convenience.** Five **Verso** records (7-seat MPVs) were excluded as a *different population*, not as outliers. A 16,000 cc engine was corrected to 1,600 — the model string itself reads "1.6". Two cars with 1 km at 50 and 76 months were **flagged, kept**, and re-tested in sensitivity analysis.

**Extreme is not erroneous.** The IQR rule flags 7.4% of prices, but these are young, low-mileage cars (median age 17 vs 62 months) — the most informative observations about depreciation. Kept. Price is right-skewed and **no transformation makes it normal** (Box-Cox fixes the skew, normality still fails), so the midterm uses rank- and median-based methods throughout.

**Shape measured, not eyeballed.** Comparing **Pearson vs Spearman** turns curvature into a number: age is close to linear ($-0.881$ vs $-0.841$); for mileage the gap reverses sign — monotonic but bent (**LOESS** explains 44% vs 32% for a straight line).

**Two thirds of the mileage effect is age in disguise.** Partial correlation: controlling for mileage costs age 8% of its association with price; controlling for age costs mileage 40%. Since age precedes mileage, KM is partly a **mediator** — a conditional age coefficient is a *direct* effect and understates the total.

**Key insight — mileage matters less as the car ages.** Age tertiles (`pd.qcut`), then **Kruskal–Wallis** per stratum with **Dunn–Bonferroni** post-hoc:

| Age stratum | $\varepsilon^2$ | Median spread across mileage groups |
|---|---|---|
| Young (1–51 mo) | **0.184** | **2,545 €** |
| Mid (52–67 mo) | 0.114 | 1,050 € |
| Old (68–80 mo) | **0.070** | **575 €** |

The mileage effect shrinks by 62% in effect size and over fourfold in euros. Among old cars, low and medium mileage are **indistinguishable** ($p = 0.81$) — a **partial price floor**. → *The final model must allow **interaction terms**, not just a transformed column.*

---

## Chapter 2 — Predict: Build a Model You Can Trust *· Final project, Parts A–C*

### 2.1 Set the rules before playing
The protocol was **fixed before any model was fitted**: RMSE in € as the metric; repeated 10-fold CV on **identical folds** for every candidate; paired **Wilcoxon + Holm** for any superiority claim; a **one-SE rule** with a pre-declared tie-break; and a **holdout opened exactly once**. This removes the freedom to pick, after the fact, the metric your favourite model wins on.

**Data preparation, briefly:**
- **Rank deficiency removed** — age is an exact function of the manufacture-date columns, which were dropped.
- **Weight audited within specification** (a wagon is legitimately heavier); a rule declared in advance (> 5 SD) caught exactly two mistyped records. $n = 1{,}429$.
- **Features from free text** — `Model` yields **Trim** (TERRA < LUNA < SOL) and **Body**; missing tokens become an explicit `Unknown` level.
- **A verified split** — 60/40 (857 / 572), stratified by price decile, checked with KS tests and SHA-256-hashed so no cell could silently redraw it.
- **No leakage** — standardisation is refitted inside each CV fold.

### 2.2 Regularisation: the penalty that did nothing
OLS, Ridge and LASSO on the same predictors and folds finish **within 6 € of each other** (SE ≈ 20 €). With ~19 observations per parameter, OLS is already stable — there is no variance left for shrinkage to buy.

What LASSO *does* buy is **interpretation**. `Fuel_Type_Diesel` is **positive under OLS, negative under Ridge, zero under LASSO** — a coefficient whose sign depends on the estimator measures collinearity, not diesel engines. The age coefficient, meanwhile, barely moves: **shrinkage acts on noise, not structure.** Stepwise selection (as a contrast only) agrees: forward and backward disagree, and only 15 of 45 predictors survive all procedures.

### 2.3 Diagnose: six assumptions, one engine
Every specification runs through the same pipeline — fit, cross-validate, test six assumptions, log — so differences between runs reflect the **specification**, never the procedure. Each test is paired with an **effect size**, since at this $n$ tests reject trivial departures.

| | Assumption | Test | Baseline |
|---|---|---|---|
| A1 | Independence | Durbin–Watson | pass |
| A2 | Linearity | Ramsey RESET | fail |
| A3 | Homoscedasticity | Breusch–Pagan | fail |
| A4 | Normality | Shapiro–Wilk | fail — *tails* only |
| A5 | Multicollinearity | VIF | fail — one cluster |
| A6 | Influence | Cook's $D$ | marginal |

**Five of six fail, yet the model predicts well: the point estimates are usable, the inference is not.** Heavy tails (not skew) point to a robust *loss* rather than a transformation; the collinearity is **one cluster** — diesel Corollas *are* the heavy, big-engine, high-tax cars.

### 2.4 Repair: three remedies, each aimed at a named failure
- **The specification.** **CCPR plots** show the curvature lies *between* variables. Six **centred cross-products** give the **largest gain in the project (+91 €, ≈ 4 SE)**; seventeen pure powers make things worse. Two lessons: the midterm's advice to transform `KM` was **tested and rejected** (the scatter had blamed mileage for age's curvature), and **VIF pruning must not run on a design containing powers and products** — it deletes Age and Weight at a cost of 375 €.
- **The response.** $\log(\text{Price})$ with **Duan's smearing** (estimated per fold) helped little and worsened normality — it targeted a skew the residuals never had.
- **The loss.** **Huber** behaved as predicted: it helped the misspecified model and added nothing to the repaired one. **Quantile regression** revealed that only the **premium end** prices age differently — ~16% steeper per month.

**What survives is HC3.** Heteroscedasticity leaves OLS unbiased but breaks the variance formula. HC3 changes no prediction; it inflates SEs by a median 1.18× and moves **4 of 52 coefficients out of significance** — four conclusions default output would have wrongly supported. Constant variance failed in **all 19 logged runs**: a 20,000 € car is simply harder to price than a 5,000 € one.

### 2.5 Choose, then open the holdout once
**Wilcoxon on per-car errors** (Holm-corrected) found the top candidates statistically tied; the **one-SE rule** decided. The chosen model — 45 base predictors plus six derived terms — came **fourth on the holdout by 6 €, and was not re-crowned**: doing so would turn the holdout into a second validation set. **Optimism is the price of selection** — models that chose their own terms lost 100–112 € from CV to holdout; models that selected nothing lost 26–56 €.

---

## Chapter 3 — Explain: Does Automatic A/C *Cause* a Premium? *· Final project, Part D*

**Treatment:** `Automatic_airco`, on 5.3% of cars. Each car has two potential prices; we see only one:

$$\underbrace{\mathbb{E}[Y\mid D{=}1]-\mathbb{E}[Y\mid D{=}0]}_{\text{Raw gap} \;=\; 8{,}622\ \text{€}} \;=\; \underbrace{\mathbb{E}[Y(1)-Y(0)\mid D{=}1]}_{\text{ATT}} \;+\; \underbrace{\mathbb{E}[Y(0)\mid D{=}1]-\mathbb{E}[Y(0)\mid D{=}0]}_{\text{Selection bias}}$$

Only randomisation kills the bias term, and **nothing randomised this option** — it was fitted to cars already aimed at the top of the range. *(The 60/40 split was random, but that says nothing about treated vs untreated cars.)*

**Controls absorb 79% of the gap — and the rest is not an effect.** Across four defensible specifications the remainder ranges **1,210–1,975 €**, a spread ~3× either standard error. SEs measure sampling noise *within* a specification, not uncertainty *across* specifications.

**Half the bias is age.** The OVB identity reproduces the removed bias exactly: **age explains 49%**, weight, mileage and horsepower most of the rest; engine size pushes the other way.

**Common support holds on paper, fails in practice.** **55 of 76 treated cars sit in one trim**, versus 2 of 896 in the entry trim — so the estimate is one within-trim comparison plus an extrapolation from 21 cars, driven by a functional form chosen for *prediction*. Twelve of sixteen covariates exceed $|SMD| > 0.25$; age reaches **−2.09** (treated cars are ~3 years younger).

**Biases this data cannot test.** `Airco`, `Boardcomputer` and `CD_Player` are **bad controls** — members of the same factory package, not causes of it. And listings are *offered* prices, not *sold* prices.

**What identification would need** is not more columns but knowledge of **how the option got onto the car**: build sheets, transaction prices, and **exogenous variation** — a year the package became standard (**DiD**), dealer supply (**IV**), or a trim cut-off (**RD**). Each yields a **LATE**, not an effect for every Corolla.

---

## Epilogue — What It All Means

**For a dealership**
1. **Price within a band, not to a point** — ≈ 8% error, running ~100 € low on unseen cars under every estimator (a property of the split). Add the offset.
2. **Data quality beats model choice** — two mistyped weights cost more CV error than the entire gap between best and worst estimator.
3. **Depreciation is not one number** — steeper at the premium end; mileage matters a lot for young cars and little for old ones.
4. **If you must quote a coefficient, quote Ridge** — a few euros of accuracy for readable numbers instead of cancelling ±10,000 € pairs.

**Limitations.** No single coefficient of the final model reads on its own — only the fitted surface is identified. Every significance claim relies on HC3. The top candidates are tied, so the final choice rests on a pre-declared rule, not proven superiority. **What the diagnosis bought is not a better model, but knowing exactly what this model can and cannot be asked.**

---

## Toolbox — What This Course Covered, and Where to Find It

| Topic | Methods | Where |
|---|---|---|
| EDA & data quality | Population definition, IQR vs. domain logic, Box-Cox, Pearson vs Spearman, LOESS, partial correlation | Midterm |
| Non-parametric testing | Kruskal–Wallis, Dunn–Bonferroni, $\varepsilon^2$, Wilcoxon + Holm | Midterm; Final 2.5 |
| Validation protocol | Repeated k-fold CV, stratified holdout, one-SE rule, optimism | Final 2.1, 2.5 |
| Regularisation | Ridge, LASSO, stepwise (as contrast) | Final 2.2 |
| Regression diagnostics | Durbin–Watson, RESET, Breusch–Pagan, Shapiro–Wilk, VIF, Cook's $D$, CCPR | Final 2.3–2.4 |
| Robust modelling | Centred interactions, log + Duan smearing, **HC3**, Huber, quantile regression | Final 2.4 |
| Causal inference | Potential outcomes, selection bias, OVB, SMD, common support, bad controls, DiD / IV / RD, LATE | Final Part D |

---

## Repository Layout and How to Run

```
.
├── Data/
│   └── Toyota_Corolla_cars.xlsx                    # 1,436 listings × 39 columns; target: Price (€)
├── Midterm/
│   ├── EDA Toyota Corolla.pdf                      # assignment brief
│   ├── Midterm_assignment_Preliminary_data_analysis_V4.ipynb
│   └── Q2_presentation.pptx
├── Final/
│   ├── Final_task_Toyota_Corolla_28.8.pdf          # assignment brief
│   ├── final_project_analysis_v3.ipynb             # Parts A–D, end to end
│   ├── Toyota_Corolla_Final_Report.pdf
│   └── Toyota_Corolla_Presentation.pdf
└── README.md
```

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn plotly openpyxl kaleido
jupyter lab
```

**Reproducibility.** Both notebooks read only the raw Excel file and run end to end. One seed governs the split and all folds; the partition is hashed. Notebook section numbers follow the assignment (`1.x` = Part A … `4.x` = Part D), not physical order — regularisation deliberately runs after diagnostics, on the design matrix the model actually uses.

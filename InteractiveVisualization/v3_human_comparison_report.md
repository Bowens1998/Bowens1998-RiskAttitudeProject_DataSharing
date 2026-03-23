# V3 Analysis: Human vs LLM Comparison (Independent CLM)

This report evaluates human participants alongside LLMs using the **V3 Independent Ordered Regression (CLM)** methodology. 
By generating scatterplots for humans, we provide a concrete visualization of the experimental variation across subjects and display their collective S-curve mathematically regressed via Ordered Logistic Models.

**AUC Integration Metric Setup:**
$$AUC_{score} = \int_{0}^{1} (	ext{Expected Value}_{Model}(x) - 	ext{Expected Value}_{GlobalBaseline}(x)) dx$$
Where a positive Area signifies an entity is globally more cautious than the AI aggregate baseline.

### 1. DSB Experiment 
*Human behaviors and LLM mappings vs V3 baseline.*
#### Human Subject Distribution
![DSB Human Scatter](DSB_Human_Scatter.png)
#### Unified S-Curve Grid (Humans + LLMs)
![DSB V3 S-Curves Human](DSB_ABC_plot.png)

### 2. FIP Experiment 
#### Human Subject Distribution
![FIP Human Scatter](FIP_Human_Scatter.png)
#### Unified S-Curve Grid (Humans + LLMs)
![FIP V3 S-Curves Human](FIP_ABC_plot.png)

### 3. TPB Experiment 
#### Human Subject Distribution
![TPB Human Scatter](TPB_Human_Scatter.png)
#### Unified S-Curve Grid (Humans + LLMs)
![TPB V3 S-Curves Human](TPB_ABC_plot.png)

---

### Final Rankings (Human + 6 LLMs)

**Global Overview (1 = Most Cautious):**

| Rank | Entity       | Mean Rank | DSB Rank | FIP Rank | TPB Rank | Std Dev |
| :--- | :---         | :---      | :---     | :---     | :---     | :---    |
| **1**| Gemini3Pro   | 2.33      | 1        | 2        | 4        | 1.25    |
| **2**| GPT5.2       | 3.33      | 3        | 5        | 2        | 1.25    |
| **3**| Sonnet4.5    | 3.33      | 2        | 3        | 5        | 1.25    |
| **4**| Human        | 3.33      | 6        | 1        | 3        | 2.05    |
| **5**| Grok4        | 4.00      | 7        | 4        | 1        | 2.45    |
| **6**| DeepSeekV3.2 | 5.33      | 4        | 6        | 6        | 0.94    |
| **7**| Qwen3Max     | 6.33      | 5        | 7        | 7        | 0.94    |

---
*Note: Human ranking reflects the collective response dynamic compared symmetrically against individual LLMs.*

### LLM-Only Rankings (Excluding Humans)

**LLM Overview (1 = Most Cautious):**

| Rank | Entity       | Mean Rank | DSB Rank | FIP Rank | TPB Rank | Std Dev |
| :--- | :---         | :---      | :---     | :---     | :---     | :---    |
| **1**| Gemini3Pro   | 1.67      | 1        | 1        | 3        | 1.15    |
| **2**| Sonnet4.5    | 2.67      | 2        | 2        | 4        | 1.15    |
| **3**| GPT5.2       | 3.00      | 3        | 4        | 2        | 1.00    |
| **4**| Grok4        | 3.33      | 6        | 3        | 1        | 2.52    |
| **5**| DeepSeekV3.2 | 4.67      | 4        | 5        | 5        | 0.58    |
| **6**| Qwen3Max     | 5.67      | 5        | 6        | 6        | 0.58    |

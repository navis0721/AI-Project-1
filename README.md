# AI Project #1
## Synthetic Data Generation for Causal Inference


### Description
---
在臨床上，若想評估某項治療所帶來的效果，理想上需要同時知道病人在「接受治療」與「未接受治療」兩種情況下的結果，並將這兩者之間的差異定義為治療效果。然而，在真實世界中，我們對同一位病人通常只能觀測到其中一種結果，也就是事實結果（factual outcome）；至於該病人在另一種情況下原本可能出現的結果，則無法直接觀測，稱為反事實結果（counterfactual outcome）。因此，本研究嘗試建立一個可用於估計治療效果的模擬資料集，建構一個同時存在治療與不治療結果的環境，並進一步觀察模型在能正確預測事實結果的同時，是否也能準確估計無法直接觀測的反事實結果。


### Composition
---
time-varying variables, static variables, treatment, outcome
- time-varying variables : 20個變數 (numerical)
- static variables : 5個變數 (numerical)
- treatment: 1個變數 (binary)，代表1代表treated, 0代表control
- outcome: 2個變數 (numerical)，分別為真實觀測結果 (factual outcome) 與反事實結果 (counterfactual outcome)


### Sample size
---
可自行調整，實驗以5000筆為主分析
- train : validation = 4000 : 1000
- treated : control = 3167 : 1833

### Condition
---
本資料集以confounder、treatment與outcome等變數為基礎，依據因果關係的概念生成每個時間點的outcome；此外，在假設存在hidden confounder 的情況下，也將其納入outcome的生成過程中，以模擬更接近真實情境的資料結構。


### Process of data collection
---
採用模擬方式產生資料，在每一個時間點，利用前一時刻的time-varying variables、static variables、treatment與outcome，透過預先設計的函數生成當前時間點的outcome。


### Reference
--- 
[Estimating treatment effects for time-to-treatment antibiotic stewardship in sepsis](https://github.com/ruoqi-liu/T4/blob/main/simulation/gen_synthetic.py)

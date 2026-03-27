# Final Report Writing Plan

> **Source note:** This plan aligns with the interim submission in `docs/interim-report.pdf` and the latest generated artifacts under `reports/`.

## 1. Recommended report strategy

Your final report should **keep the same core story as the interim report/proposal** but upgrade it in three ways:

1. **Tighten the research question** around forecasting SPY daily log returns with sequence models.
2. **Make the methodology consistent** by treating the shared `SEQ_LEN = 30` and `HORIZON = 10` comparison workflow as the main experimental setting.
3. **Use the generated reports and figures as evidence**, instead of describing results only in prose.

A good final report structure is:

1. Introduction  
2. Background / Literature Context  
3. Problem Definition and Objectives  
4. Data and Pre-processing  
5. Methodology  
6. Experimental Design  
7. Results  
8. Discussion  
9. Limitations  
10. Conclusion and Future Work  
11. References  
12. Appendices

---

## Terminology guardrail (use consistently in every chapter)

- **task**: one forecast definition evaluated end-to-end.
- **task_id**: stable identifier for that task (for example, `spy_h10_horizon_return`).
- **target_mode**: label construction mode (`horizon_return`, `next_return`, `next3_mean_return`).
- **horizon**: forecast step used by `horizon_return` mode.

When reporting results, always show `task_id`, `target_mode`, and `horizon` together so narrative text matches generated artifacts.

## 2. Chapter-by-chapter writing plan

## Chapter 1. Introduction

### Writing goal
Explain the project motivation, why stock return forecasting is difficult, and why comparing multiple neural sequence models against a simpler baseline is academically worthwhile.

### What to write
- Introduce the practical importance of financial time-series forecasting.
- Narrow the problem to **forecasting SPY daily log returns**.
- Explain that the project compares **RNN, LSTM, GRU, and Transformer** models.
- Mention that a **flattened-sequence linear regression baseline** is included for fair comparison.
- End the chapter with the report structure.

### Suggested subsection flow
- 1.1 Project background
- 1.2 Problem statement
- 1.3 Research aim
- 1.4 Objectives
- 1.5 Scope of the project
- 1.6 Report organisation

### Evidence / figure placement
- No figure is strictly necessary here.
- Optional visual if you want an early preview of the outcome:  
  `[insert: best_tuned_testing_loss.svg]`

### Writing tip
Keep this chapter high-level. Do not put too many metrics here; save detailed model comparisons for the Results chapter.

---

## Chapter 2. Background / Literature Context

### Writing goal
Show the academic context behind time-series forecasting and justify why these four neural architectures were chosen.

### What to write
- Briefly explain time-series forecasting in finance.
- Discuss why predicting returns is harder than predicting price levels.
- Introduce:
  - vanilla RNN,
  - LSTM,
  - GRU,
  - Transformer.
- Explain why linear regression is still a meaningful baseline.
- Summarise expected strengths and weaknesses of each model.

### Suggested subsection flow
- 2.1 Financial time-series forecasting
- 2.2 Traditional and machine-learning baselines
- 2.3 Recurrent neural networks
- 2.4 LSTM and GRU improvements over vanilla RNN
- 2.5 Transformer models for sequence learning
- 2.6 Research gap and project positioning

### Evidence / figure placement
- Usually no repository figure is required here.
- If needed, use your own conceptual diagram rather than experiment plots.

### Writing tip
This chapter should cite external papers and textbooks, not only repository artifacts.

---

## Chapter 3. Problem Definition and Objectives

### Writing goal
Convert the broad project idea into a precise technical problem statement.

### What to write
- Define the task as a **supervised forecasting problem**.
- State the current canonical setup:
  - ticker: **SPY**,
  - target: **daily log return**,
  - input window: **30 trading days**,
  - forecast horizon: **10 trading days ahead**.
- Explain the evaluation goal: compare predictive accuracy and directional accuracy across models.
- State the research objectives clearly.

### Suggested subsection flow
- 3.1 Forecasting task definition
- 3.2 Research questions
- 3.3 Project objectives
- 3.4 Success criteria

### Suggested research questions
- Which neural architecture performs best on the shared SPY forecasting setup?
- Does a tuned neural model outperform the linear regression baseline?
- Is lower error associated with better directional accuracy?

### Evidence / figure placement
- Optional summary figure if you want to preview the final task outcome:  
  `[insert: best_tuned_validation_loss.svg]`

---

## Chapter 4. Data and Pre-processing

### Writing goal
Document the dataset carefully so the experiment is reproducible and methodologically defensible.

### What to write
- Data source: **Yahoo Finance via `yfinance`**.
- Instrument: **SPY ETF**.
- Explain why SPY is a reasonable proxy for broad market behaviour.
- Describe the transformation from adjusted close prices to **daily log returns**.
- Explain sequence construction:
  - input = previous 30 returns,
  - output = return at horizon 10.
- Describe the **chronological train/validation/test split**.
- Explain scaling and why it is fit on training inputs only.

### Suggested subsection flow
- 4.1 Data source and asset selection
- 4.2 Return definition
- 4.3 Sequence construction
- 4.4 Train/validation/test split
- 4.5 Normalisation and leakage prevention

### Evidence / figure placement
- If you want one model-output example later linked back to the data definition, reserve:  
  `[insert: lstm_pred_slice.png]`

### Writing tip
Be explicit about dates used in the final frozen experiment set if you can recover them from your logs.

---

## Chapter 5. Methodology

### Writing goal
Explain how each model was built and trained, and ensure the report uses one consistent experimental story.

### What to write
- Describe the common comparison pipeline used across all models.
- Explain that all models are evaluated on the **same prepared dataset split**.
- Describe each model briefly:
  - RNN,
  - LSTM,
  - GRU,
  - Transformer,
  - Baseline-LR.
- State the optimisation and training choices:
  - loss function,
  - optimiser,
  - early stopping,
  - checkpointing,
  - validation-based model selection.
- Explain the metrics used:
  - MAE,
  - MSE,
  - Directional Accuracy (DA).

### Suggested subsection flow
- 5.1 Overall pipeline
- 5.2 Baseline model
- 5.3 RNN model
- 5.4 LSTM model
- 5.5 GRU model
- 5.6 Transformer model
- 5.7 Training strategy
- 5.8 Evaluation metrics

### Evidence / figure placement
Use one visual that communicates the training-behaviour comparison across tuned models:  
- `[insert: best_tuned_training_loss.svg]`

### Writing tip
This chapter should emphasise fairness of comparison and shared experimental conditions.

---

## Chapter 6. Experimental Design and Hyperparameter Tuning

### Writing goal
Show how you designed experiments systematically rather than trying random settings.

### What to write
- Explain the staged tuning workflow.
- Describe the tuning order:
  - recurrent models: `lr -> hidden -> layers -> batch_size`,
  - transformer: `lr -> d_model -> num_layers -> nhead -> batch_size`.
- State that validation MSE was used to choose winners.
- Summarise the best configurations found for each model.
- Explain that the final comparison uses the frozen winners from `tuning_winners.csv`.

### Suggested subsection flow
- 6.1 Purpose of tuning
- 6.2 Tuning procedure
- 6.3 Search dimensions by model
- 6.4 Best configurations obtained
- 6.5 Threats to validity in tuning

### Evidence / figure placement
This chapter should definitely use the tuning evidence:
- `[insert: best_tuned_validation_loss.svg]`
- `[insert: best_tuned_testing_loss.svg]`
- Optional per-model training chart references in discussion:  
  `[insert: loss_LSTM_best_tuned_lstm_comparison-20260320T093214Z.png]`  
  `[insert: loss_GRU_best_tuned_gru_comparison-20260320T093215Z.png]`  
  `[insert: loss_RNN_best_tuned_rnn_comparison-20260320T093215Z.png]`  
  `[insert: loss_Transformer_best_tuned_transformer_comparison-20260320T093216Z.png]`

### Writing tip
This chapter should cite your tuning tables directly, especially when explaining why the final settings were chosen.

---

## Chapter 7. Results

### Writing goal
Present the final model comparison clearly and objectively.

### What to write
Start with a **main comparison table** using the best tuned models plus the baseline. Then interpret the results in stages.

### Suggested subsection flow
- 7.1 Final tuned comparison overview
- 7.2 Error-based performance comparison
- 7.3 Directional accuracy comparison
- 7.4 Prediction pattern visualisation
- 7.5 Summary of key findings

### Main points to report
Based on the current report artifacts:
- **LSTM** is the best model by validation MSE and test MSE.
- **Baseline-LR** is surprisingly competitive and ranks just behind LSTM in the tuned comparison.
- **GRU** has the strongest directional accuracy among the tuned neural models.
- **Transformer** performs worst by loss on the current archived results.

### Evidence / figure placement
This is the most figure-heavy chapter. Recommended placements:

#### 7.1 Overall comparison
- `[insert: best_tuned_testing_loss.svg]`
- `[insert: best_tuned_validation_loss.svg]`

#### 7.2 Model-specific fit behaviour
- `[insert: lstm_best_tuned_lstm_comparison-20260320t093214z_scatter.png]`
- `[insert: gru_best_tuned_gru_comparison-20260320t093215z_scatter.png]`
- `[insert: rnn_best_tuned_rnn_comparison-20260320t093215z_scatter.png]`
- `[insert: transformer_best_tuned_transformer_comparison-20260320t093216z_scatter.png]`

#### 7.3 Prediction slices
- `[insert: lstm_best_tuned_lstm_comparison-20260320t093214z_pred_slice.png]`
- `[insert: gru_best_tuned_gru_comparison-20260320t093215z_pred_slice.png]`
- `[insert: rnn_best_tuned_rnn_comparison-20260320t093215z_pred_slice.png]`
- `[insert: transformer_best_tuned_transformer_comparison-20260320t093216z_pred_slice.png]`

### Writing tip
Do not only say which model “won.” Also explain whether the margin is large or small, and whether differences in MSE align with differences in DA.

---

## Chapter 8. Discussion

### Writing goal
Interpret what the results mean, not just what the numbers are.

### What to write
- Explain why LSTM may have performed best:
  - better memory handling than vanilla RNN,
  - possibly more stable than Transformer on this dataset size.
- Explain why the strong baseline result matters:
  - the forecasting signal may be weak,
  - simple models can remain competitive in financial return prediction.
- Discuss why GRU may have higher directional accuracy despite not having the best MSE.
- Interpret what the poorer Transformer result suggests about data scale, tuning budget, or task suitability.
- Compare findings back to your literature review.

### Suggested subsection flow
- 8.1 Interpretation of the winning model
- 8.2 Why the baseline remained competitive
- 8.3 Error metrics versus directional accuracy
- 8.4 Comparison with expectations from literature
- 8.5 Practical meaning of the results

### Evidence / figure placement
- `[insert: best_tuned_testing_loss.svg]`
- `[insert: gru_best_tuned_gru_comparison-20260320t093215z_pred_slice.png]`
- `[insert: lstm_best_tuned_lstm_comparison-20260320t093214z_pred_slice.png]`

### Writing tip
This chapter is where you demonstrate critical thinking. Avoid claiming that the best model is universally superior; frame conclusions around this dataset and setup.

---

## Chapter 9. Limitations

### Writing goal
Acknowledge methodological weaknesses honestly.

### What to write
- Results appear to come from a limited number of archived runs.
- No repeated-run mean ± standard deviation is highlighted in the current reports.
- Financial market data may drift over time because it is downloaded dynamically.
- Only one asset (SPY) is used.
- Only one main horizon/setup is emphasised in the current workflow.
- Transaction costs and trading-strategy profitability are not evaluated.

### Suggested subsection flow
- 9.1 Dataset limitations
- 9.2 Experimental limitations
- 9.3 Model comparison limitations
- 9.4 External validity limitations

### Evidence / figure placement
- No figure is required.

### Writing tip
A strong limitations chapter improves credibility rather than weakening your report.

---

## Chapter 10. Conclusion and Future Work

### Writing goal
Close the report with a concise answer to the research questions and realistic future extensions.

### What to write
- Restate the aim.
- Summarise the key finding that **LSTM delivered the best overall loss performance** in the tuned comparison.
- Note that **linear regression remained highly competitive**, which is an important conclusion.
- Mention that **GRU showed strong directional prediction behaviour**.
- Suggest future work:
  - repeated experiments,
  - more assets,
  - additional features,
  - walk-forward validation,
  - trading simulation with costs,
  - more extensive Transformer tuning.

### Suggested subsection flow
- 10.1 Conclusion
- 10.2 Contributions of the project
- 10.3 Future work

### Evidence / figure placement
- Optional closing summary figure:  
  `[insert: best_tuned_testing_loss.svg]`

---

## 3. Suggested appendix plan

Include the detailed evidence in appendices so the main body stays readable.

### Appendix A. Hyperparameter search details
- Candidate values tried by model.
- Stage-by-stage winners.
- Reference report: `hyperparameter_impact_report.md`.

### Appendix B. Full result tables
- Final tuned comparison table.
- Per-model metrics JSON summaries.
- Reference report: `best_tuned_comparison.md`.

### Appendix C. Additional figures
- Full loss curves.
- Scatter plots.
- Prediction slices for all models.

### Appendix D. Reproducibility notes
- Run IDs.
- Commit hash.
- Environment information if available from logs.

---

## 4. Recommended figure-use map

If you want a simple “which figure goes where” checklist, use this:

- **Methodology chapter:**  
  `[insert: best_tuned_training_loss.svg]`
- **Tuning chapter:**  
  `[insert: best_tuned_validation_loss.svg]`  
  `[insert: best_tuned_testing_loss.svg]`
- **Results chapter (main):**  
  `[insert: best_tuned_testing_loss.svg]`
- **Results chapter (behaviour analysis):**  
  `[insert: lstm_best_tuned_lstm_comparison-20260320t093214z_scatter.png]`  
  `[insert: gru_best_tuned_gru_comparison-20260320t093215z_scatter.png]`
- **Results/Discussion chapter (qualitative examples):**  
  `[insert: lstm_best_tuned_lstm_comparison-20260320t093214z_pred_slice.png]`  
  `[insert: gru_best_tuned_gru_comparison-20260320t093215z_pred_slice.png]`
- **Appendix:**  
  all remaining `loss_*.png`, `*_scatter.png`, and `*_pred_slice.png` figures.

---

## 5. Practical writing order

To finish the report efficiently, write in this order:

1. **Chapter 3: Problem Definition and Objectives**
2. **Chapter 4: Data and Pre-processing**
3. **Chapter 5: Methodology**
4. **Chapter 6: Experimental Design and Hyperparameter Tuning**
5. **Chapter 7: Results**
6. **Chapter 8: Discussion**
7. **Chapter 9: Limitations**
8. **Chapter 10: Conclusion and Future Work**
9. **Chapter 1: Introduction**
10. **Chapter 2: Background / Literature Context**

This order works well because the middle chapters depend directly on your existing artifacts, while the introduction and literature review are easier to refine after your argument is stable.

---

## 6. One-paragraph thesis story you can keep consistent throughout the report

This project investigates whether neural sequence models can improve the forecasting of SPY daily log returns under a shared experimental setup. Using a 30-day input window and a 10-day forecasting horizon, the study compares vanilla RNN, LSTM, GRU, and Transformer architectures against a flattened-sequence linear regression baseline. The final tuned comparison shows that LSTM achieves the strongest overall loss performance, while GRU performs well on directional accuracy and the linear regression baseline remains highly competitive. The findings suggest that, for this financial forecasting task, more complex architectures do not automatically guarantee superior performance, and careful model selection, fair comparison, and rigorous evaluation are essential.

---

## 7. What you should do next

1. Use this file as the outline for your final report draft.
2. Build your main results table from `reports/best_tuned_comparison.md`.
3. Build your tuning subsection from `reports/hyperparameter_impact_report.md`.
4. Insert the recommended figures using the `[insert: ...]` placeholders.
5. Do one final pass against `docs/interim-report.pdf` to align chapter wording and continuity with the earlier submission.

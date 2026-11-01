# Manuscript Figures

The cleaned repository retains the figures used in the current manuscript and one supplementary diagnostic figure.

- `fig1_controller_architecture.png`: controller architecture schematic.
- `fig2_primary_paired_tradeoff.pdf/png`: primary paired charging-time versus thermal-spread result.
- `fig_extended_evaluation_robustness.pdf/png`: four-panel architecture attribution, training-seed stability, shared-objective fairness, and model-plant mismatch synthesis.
- `fig4_surrogate_ranking_fidelity.pdf/png`: candidate-ranking fidelity diagnostic.
- `fig5_hust_paired_tradeoff.pdf/png`: held-out HUST measured-protocol replay trade-off.
- `figS1_action_conditioning.pdf/png`: held-out action-conditioning diagnostic.

The previous standalone thermal-initialisation plot was removed from the main figure set. Its numerical results and statistical outputs remain in `results/diagnostics/thermal_initial_condition_n30/`.

Regenerate the extended-evaluation figure with:

```bash
python scripts/figures/generate_extended_evaluation_figure.py
```

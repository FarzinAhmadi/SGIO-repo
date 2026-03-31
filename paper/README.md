# Paper Source

LaTeX source for the manuscript.

## Files

| File | Description |
|------|-------------|
| `main.tex` | Full manuscript source |
| `IEEEtran.cls` | IEEE Transactions LaTeX class file |
| `fig1.png` | Figure 1 (SGIO framework overview) |
| `smgil_validation_figures.png` | Figure 2 (NHANES validation results) |
| `smgil_validation_figures_scaled.pdf` | Figure 2 (PDF version) |
| `smgil_mfp_validation_final.pdf` | Figure 3 (MFP validation results) |

## Compiling

```bash
cd paper/
pdflatex main.tex
pdflatex main.tex   # second pass for cross-references
```

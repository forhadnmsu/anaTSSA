# Asymmetry Extraction Methods

This document outlines two simplified methods for extracting transverse single-spin asymmetry (\(A_N\)) assuming equal relative luminosity (\(r = 1\)).

The final asymmetry in both methods is scaled by the effective factor \(f\):

\[
f = \text{dilution factor} \times \text{packing fraction} \times \left( \frac{P_\uparrow + P_\downarrow}{2} \right)
\]

---

## Method 1

### Counts Used:
- \( N_L^\uparrow, N_R^\uparrow \)
- \( N_L^\downarrow, N_R^\downarrow \)

### Raw Asymmetries:
- \( A^\uparrow = \frac{N_L^\uparrow - N_R^\uparrow}{N_L^\uparrow + N_R^\uparrow} \)
- \( A^\downarrow = \frac{N_L^\downarrow - N_R^\downarrow}{N_L^\downarrow + N_R^\downarrow} \)

### Final Asymmetry:
- \( A_N = \frac{1}{f} \cdot \frac{A^\uparrow - A^\downarrow}{2} \)

### Error Propagation:
- \( \sigma_{A^\uparrow}^2 = \frac{4 N_L^\uparrow N_R^\uparrow}{(N_L^\uparrow + N_R^\uparrow)^3} \)
- \( \sigma_{A^\downarrow}^2 = \frac{4 N_L^\downarrow N_R^\downarrow}{(N_L^\downarrow + N_R^\downarrow)^3} \)

### Final Error:
- \( \sigma_{A_N} = \frac{1}{2f} \sqrt{\sigma_{A^\uparrow}^2 + \sigma_{A^\downarrow}^2} \)

---

## Method 2

### Counts Used:
- \( N_L^\uparrow, N_L^\downarrow \)
- \( N_R^\uparrow, N_R^\downarrow \)

### Raw Asymmetries:
- \( A_L = \frac{N_L^\uparrow - N_L^\downarrow}{N_L^\uparrow + N_L^\downarrow} \)
- \( A_R = \frac{N_R^\uparrow - N_R^\downarrow}{N_R^\uparrow + N_R^\downarrow} \)

### Final Asymmetry:
- \( A_N = \frac{1}{f} \cdot \frac{A_L - A_R}{2} \)

### Error Propagation:
- \( \sigma_{A_L}^2 = \frac{4 N_L^\uparrow N_L^\downarrow}{(N_L^\uparrow + N_L^\downarrow)^3} \)
- \( \sigma_{A_R}^2 = \frac{4 N_R^\uparrow N_R^\downarrow}{(N_R^\uparrow + N_R^\downarrow)^3} \)

### Final Error:
- \( \sigma_{A_N} = \frac{1}{2f} \sqrt{\sigma_{A_L}^2 + \sigma_{A_R}^2} \)

---

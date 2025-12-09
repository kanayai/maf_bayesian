# Experimental Data Provenance

This document records the relationship between the truncated experimental dataset (located in `data/experimental/h`) and the full experimental dataset (located in `data/experimental/full_experimental_data`).

The truncated dataset was derived as a subset of the full dataset, generally by taking the initial sequence of data points up to a load of approximately 10 kN.

## Horizontal Direction ('h') Mappings

The following table maps each file in the current dataset to its source in the full dataset, including the exact Python slice indices used to verify the match.

| Current File | Full Source File | Subset Type | Slice Indices (Python) | Full Max Load |
| :--- | :--- | :--- | :--- | :--- |
| `input_load_angle_exp_1.txt` | `input_load_angle_exp_45_h_1.txt` | Prefix Subset | `[0:27]` | 16.474 kN |
| `input_load_angle_exp_2.txt` | `input_load_angle_exp_45_h_2.txt` | Internal Subset | `[1:22]` (Start skipped) | 16.580 kN |
| `input_load_angle_exp_3.txt` | `input_load_angle_exp_45_h_3.txt` | Internal Subset | `[1:28]` (Start skipped) | 16.620 kN |
| `input_load_angle_exp_4.txt` | `input_load_angle_exp_90_h_1.txt` | Prefix Subset | `[0:20]` | 12.166 kN |
| `input_load_angle_exp_5.txt` | `input_load_angle_exp_90_h_2.txt` | Internal Subset | `[1:17]` (Start skipped) | 13.080 kN |
| `input_load_angle_exp_6.txt` | `input_load_angle_exp_90_h_3.txt` | Internal Subset | `[1:16]` (Start skipped) | 12.437 kN |
| `input_load_angle_exp_7.txt` | `input_load_angle_exp_135_h_1.txt` | Prefix Subset | `[0:34]` | 13.739 kN |
| `input_load_angle_exp_8.txt` | `input_load_angle_exp_135_h_2.txt` | Internal Subset | `[1:33]` (Start skipped) | 14.894 kN |

**Note on "Start skipped":** Files marked as "Internal Subset" are missing the first data point (row 0) present in the full dataset. In the full dataset, this first row corresponds to `Load ≈ 0` and `Extension ≈ 0`.

## Reconstruction

To reconstruct the current dataset from the full dataset, one can simply load the full source file and sub-select the rows using the indices provided above.

Example:
```python
full_data = np.loadtxt("data/experimental/full_experimental_data/input_load_angle_exp_45_h_2.txt", delimiter=",")
# Reconstruct exp_2
current_data_reconstructed = full_data[1:22]
```

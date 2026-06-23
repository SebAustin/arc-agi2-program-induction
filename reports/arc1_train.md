# Evaluation: training

- **Configuration:** offline-enumerative
- **Tasks:** 400
- **pass@2:** 0.0925 (9.25%)
- **Train-solve rate:** 0.0975 (9.75%)
- **Total time:** 431.7s (1.08s/task)

## Solved tasks

| task_id | score | program |
| --- | --- | --- |
| 0d3d703e | 1.00 | `color_map([[1,5],[2,6],[3,4],[4,3],[5,1],[6,2],[8,9],[9,8]])` |
| 1cf80156 | 1.00 | `crop_to_content` |
| 1e0a9b12 | 1.00 | `gravity(down)` |
| 1f85a75f | 1.00 | `keep_largest | crop_to_content` |
| 23b5c85d | 1.00 | `keep_smallest | crop_to_content` |
| 28bf18c6 | 1.00 | `crop_to_content | tile(1,2)` |
| 3906de3d | 1.00 | `gravity(up)` |
| 3af2c5a8 | 1.00 | `mirror_concat_h | mirror_concat_v` |
| 3c9b0459 | 1.00 | `rot180` |
| 496994bd | 1.00 | `repair_symmetry(0)` |
| 4c4377d9 | 1.00 | `flip_v | mirror_concat_v` |
| 6150a2bd | 1.00 | `rot180` |
| 62c24649 | 1.00 | `mirror_concat_h | mirror_concat_v` |
| 67a3c6ac | 1.00 | `flip_h` |
| 67e8384a | 1.00 | `mirror_concat_h | mirror_concat_v` |
| 68b16354 | 1.00 | `flip_v` |
| 6d0aefbc | 1.00 | `mirror_concat_h` |
| 6fa7a44f | 1.00 | `mirror_concat_v` |
| 7468f01a | 1.00 | `flip_h | crop_to_content` |
| 74dd1130 | 1.00 | `transpose` |
| 8be77c9e | 1.00 | `mirror_concat_v` |
| 9172f3a0 | 1.00 | `scale(3,3)` |
| 9565186b | 1.00 | `set_foreground(5)` |
| 9dfd6313 | 1.00 | `transpose` |
| a416b8f3 | 1.00 | `tile(1,2)` |
| a740d043 | 1.00 | `replace_bg(0) | crop_to_content` |
| b1948b0a | 1.00 | `color_map([[6,2]])` |
| b8825c91 | 1.00 | `repair_symmetry(4)` |
| be94b721 | 1.00 | `keep_largest | crop_to_content` |
| c59eb873 | 1.00 | `scale(2,2)` |
| c8f0f002 | 1.00 | `color_map([[7,5]])` |
| c9e6f938 | 1.00 | `mirror_concat_h` |
| d511f180 | 1.00 | `color_map([[5,8],[8,5]])` |
| dae9d2b5 | 1.00 | `overlay_or(6)` |
| ed36ccf7 | 1.00 | `rot90` |
| f25fbde4 | 1.00 | `crop_to_content | scale(2,2)` |
| f25ffba3 | 1.00 | `flip_v | crop_to_content | mirror_concat_v` |

# Evaluation: training

- **Configuration:** offline-enumerative
- **Tasks:** 400
- **pass@2:** 0.0525 (5.25%)
- **Train-solve rate:** 0.0525 (5.25%)
- **Total time:** 493.8s (1.23s/task)

## Solved tasks

| task_id | score | program |
| --- | --- | --- |
| 0c786b71 | 1.00 | `rot180 | mirror_concat_h | mirror_concat_v` |
| 0d3d703e | 1.00 | `color_map([[1,5],[2,6],[3,4],[4,3],[5,1],[6,2],[8,9],[9,8]])` |
| 1cf80156 | 1.00 | `crop_to_content` |
| 1e0a9b12 | 1.00 | `gravity(down)` |
| 1f85a75f | 1.00 | `keep_largest | crop_to_content` |
| 23b5c85d | 1.00 | `keep_smallest | crop_to_content` |
| 28bf18c6 | 1.00 | `crop_to_content | tile(1,2)` |
| 31d5ba1a | 1.00 | `overlay_xor(6)` |
| 3906de3d | 1.00 | `gravity(up)` |
| 3af2c5a8 | 1.00 | `mirror_concat_h | mirror_concat_v` |
| 3c9b0459 | 1.00 | `rot180` |
| 496994bd | 1.00 | `repair_symmetry(0)` |
| 4c4377d9 | 1.00 | `flip_v | mirror_concat_v` |
| 59341089 | 1.00 | `flip_h | mirror_concat_h | mirror_concat_h` |
| 60c09cac | 1.00 | `scale(2,2)` |
| 6150a2bd | 1.00 | `rot180` |
| 62c24649 | 1.00 | `mirror_concat_h | mirror_concat_v` |
| 66f2d22f | 1.00 | `replace_bg(5) | overlay_or(0)` |
| 67a3c6ac | 1.00 | `flip_h` |
| 67e8384a | 1.00 | `mirror_concat_h | mirror_concat_v` |
| 68b16354 | 1.00 | `flip_v` |

| Scenario | Planner | Success | First solution (median s) | Final cost / best R² cost (median) | Nodes (median) | Iterations/s |
|---|---|---:|---:|---:|---:|---:|
| narrow_passage | RRT* (R²) | 100% | 0.002 | 1.001 | 65,124 | 39,248 |
| narrow_passage | RRT* Dubins ρ=0.5 | 95% | 0.043 | 1.032 | 7,791 | 5,390 |
| narrow_passage | RRT* Dubins ρ=1.0 | 70% | 0.194 | 1.040 | 4,874 | 3,865 |
| cluttered | RRT* (R²) | 100% | 0.001 | 1.001 | 57,315 | 48,679 |
| cluttered | RRT* Dubins ρ=0.5 | 100% | 0.033 | 1.062 | 6,548 | 7,374 |
| cluttered | RRT* Dubins ρ=1.0 | 55% | 0.172 | 1.126 | 3,668 | 5,810 |
| maze | RRT* (R²) | 100% | 0.005 | 1.001 | 62,812 | 42,898 |
| maze | RRT* Dubins ρ=0.5 | 80% | 0.104 | 1.058 | 7,288 | 7,668 |
| maze | RRT* Dubins ρ=1.0 | 5% | 0.401 | 1.083 | 22 | 259,915 |

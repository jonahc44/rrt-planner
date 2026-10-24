| Scenario | Planner | Success | First solution (median s) | Final cost / best R² cost (median) | Nodes (median) | Iterations/s |
|---|---|---:|---:|---:|---:|---:|
| narrow_passage | RRT* (R²) | 100% | 0.001 | 1.001 | 73,087 | 43,351 |
| narrow_passage | RRT* Dubins ρ=0.5 | 100% | 0.014 | 1.032 | 8,522 | 5,735 |
| narrow_passage | RRT* Dubins ρ=1.0 | 80% | 0.010 | 1.043 | 5,010 | 4,561 |
| cluttered | RRT* (R²) | 100% | 0.001 | 1.001 | 59,316 | 50,200 |
| cluttered | RRT* Dubins ρ=0.5 | 100% | 0.006 | 1.061 | 6,650 | 7,570 |
| cluttered | RRT* Dubins ρ=1.0 | 85% | 0.083 | 1.118 | 3,658 | 6,059 |
| maze | RRT* (R²) | 100% | 0.005 | 1.001 | 67,342 | 45,340 |
| maze | RRT* Dubins ρ=0.5 | 100% | 0.025 | 1.060 | 7,534 | 8,115 |
| maze | RRT* Dubins ρ=1.0 | 5% | 0.397 | 1.083 | 22 | 256,009 |

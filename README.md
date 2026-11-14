# DIGO
Experiment code for *DIGO: A Dynamic Time-Dependent Path Oracle via Diffusion-Guided Search-Space Prediction* by Haolun Ma, Jing Zhao, Mengxuan Zhang and Lei Li.

## Build and run the search
Use a C++17 compiler, CMake and the Boost multiprecision headers. From the repository root:

~~~sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python3 ges/smoke.py build/libregional.so
~~~

On macOS use `build/libregional.dylib`. The smoke test constructs a three-vertex network in a temporary directory and checks endpoint repair, optimal duration and the earliest departure. The portable build uses Boost exact rational arithmetic. Backend and build flags matter when comparing timings.

## Query with a trained SSP
Install the Python dependencies with `python3 -m pip install -r requirements.txt`. Supply a trusted checkpoint containing a DiT state dictionary or a dictionary with a model entry. Use the calibrated threshold belonging to that checkpoint.

~~~sh
python3 ssp/query.py --library build/libregional.so --graph graph.txt \
  --checkpoint weights.pt --conditions conditions.npz --threshold 0.5 \
  --city BJ --snapshot G0 --role base_test --source 10 --destination 30 --lo 28800 --hi 43200
~~~

The paths, IDs and threshold above are examples. A graph starts with vertex count, directed arc count, knot count and knot spacing in seconds. Each vertex row gives its original ID and row-major grid cell (0 to 4095). Each arc row contains its source and destination IDs followed by its nonnegative periodic FIFO travel-time knots. The city experiments use 96 knots at 900-second spacing. The solver returns JSON including the recovered route, exact duration and earliest minimizing departure on the closed interval. Method 4 repairs disconnected masks and optimizes within the resulting region. Method 0 searches the full graph.

The NPZ describes one OD. Provide normalized origin/destination coordinates and time of day as float32 `odt[N,5]`, integer `query_ids[N]`, boolean road `support[4096]`, and integer `endpoints[2]` in the same grid as the graph. The script performs one flow step per condition, thresholds occupancy, inserts endpoints and unions the masks before search.

## Experiment sources
| Directory | Paper component |
|---|---|
| ccd | Traffic snapshots, shared point routing, representative sampling and label encoding. Historical corpus utilities use their own grid settings. |
| experiments/g0 | Native 64×64 training corpus, spatial sampling comparisons, SSP training, checkpoint selection and occupancy calibration |
| experiments/g1 | Same-departure change detection, ordinary/targeted selection, label refresh, Random FT and CCD FT |
| experiments/baseline | SSE and PiT region-component adapters, training and inference |
| experiments/rolling | Quarter-hour snapshot construction, label preparation and continual adaptation for the daily experiment |
| experiments/sensitivity | Score-qualified neighboring-cell expansion and path-based query construction |
| ges | Full and regional continuous interval search and the repair ablations |

The experiment drivers use the original staged run layout. Initialize a fresh workspace with `python3 experiments/setup.py /absolute/path/to/workspace`. It writes the G0/G1/baseline code and configured protocols under `runs/sdc_ssp64_od49_56/`. Supply your own city snapshots, node coordinates, staged role manifests, label shards and intermediate artifacts expected by each stage. For an already prepared E1 run, execute `python3 experiments/g0/train64.py --run /path/to/e1_v1 --city BJ --method hierarchical_grid --seed 42`. The research training drivers use CUDA. The rolling scripts read their run location from DIGO_ROLLING_RUN. Set DIGO_WORKSPACE for baseline drivers and place upstream [JIT-TP](https://github.com/edenjing/JIT-TP) and [DOT](https://github.com/Logan-Lin/DOT) sources under the E3 run's external_src directory, using the repository names JIT-TP and DOT.

The data-free smoke test exercises the native solver. Reproducing the city tables and curves requires the original inputs, query cohorts, selected checkpoints and evaluation manifests. The source release includes experiment components and does not automate reconstruction of all paper cohorts. Table III reports daily optimizer time over 96 updates. It excludes label preparation and publication latency.

The width study compares closed intervals of 0.5, 1, 2 and 4 hours. Source drivers are in experiments/width. DIGO_WORKER_HOSTS is a comma-separated host list for partitioning a prepared workload. The guarded search backend is a separate CMake target. The comparison cohort analysis in experiments/e2e/selected_cohort.py takes per-method timing and answer records as CSV and writes its selected IDs and metrics to a fresh output directory. Selection uses observed quality and timing, so metrics on this subset must be reported separately from full-workload results.

Set DIGO_E2E_RUN to the prepared timing study directory for its research drivers. Upstream baseline revisions used in the experiment:
- https://github.com/edenjing/JIT-TP @ 259136246ef81ef246834e226e36cc406227bbba
- https://github.com/Logan-Lin/DOT @ 47243eb459602aa655441ed9182bd38c154750b8

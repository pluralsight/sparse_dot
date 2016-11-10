'''Benchmarks using /usr/bin/time'''

import os
import sparse_dot

cmd = '/usr/bin/time --format="time: %E peak_memory: %MkB" python -c "import sparse_dot.testing_utils as t; print t.run_timing_test({}, {}, {})"'

os.system(cmd.format(1000, 1000, 100000))
os.system(cmd.format(2000, 2000, 100000))
os.system(cmd.format(5000, 5000, 100000))
os.system(cmd.format(10000, 10000, 100000))
os.system(cmd.format(10000, 10000, 1000000))
os.system(cmd.format(10000, 10000, 10000000))
os.system(cmd.format(1000, 20000, 10000000))
os.system(cmd.format(5000, 20000, 10000))

# Sample benchmarks table:
# num_rows, num_cols, num_entries, generation_time, processing_time, wall_time, peak_memory
#     1000,     1000,      100000,          0.0617,           0.145,   0:01.00,       46 MB
#     2000,     2000,      100000,          0.2590,           0.251,   0:01.27,       69 MB
#     5000,     5000,      100000,          1.8009,           0.558,   0:03.12,      225 MB
#    10000,    10000,      100000,          7.7644,           1.009,   0:09.55,      797 MB
#    10000,    10000,     1000000,          8.6777,          10.598,   0:20.12,      801 MB
#    10000,    10000,    10000000,         10.7910,         106.426,   1:58.03,     1263 MB
#     1000,    20000,    10000000,          4.2849,           9.958,   0:16.31,      501 MB
#     5000,    20000,       10000,          8.2295,           0.103,   0:09.07,      797 MB

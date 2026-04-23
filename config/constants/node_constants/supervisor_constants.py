MAX_NODE_EXECUTIONS = 12
# Each evaluator pass adds 1 failure for every step with 0 results.
# With MAX_EVALUATOR_RETRIES=1 and MAX_REPLANS=1 that means up to 3 evaluator
# passes before "forced_proceed" is reached. Setting this below 4 lets the
# supervisor fire the partial-finalization path before the retry/replan cycle
# completes, skipping synthesis entirely and producing an empty report.
MAX_SEARCH_FAILURES = 4

T1 = 80      # safe zone
T2 = 105     # optimization zone
T3 = 120     # critical zone
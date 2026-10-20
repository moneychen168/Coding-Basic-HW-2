errors per program x compute-sanitizer tool (clean = 0 errors and test exit 0)

| program | memcheck | racecheck | initcheck | synccheck |
|---|---|---|---|---|
| test_grouped | 0 | ? (no_summary(rc=0)) | 0 | 0 |
| test_reuse_e2e | 0 | ? (TIMEOUT) | 0 | 0 |
| test_graph | 0 | ? (TIMEOUT) | 0 | 0 |
| test_i8_grouped | 3 (ERRORS) | ? (no_summary(rc=0)) | 18990 (ERRORS) | 0 |
| test_operand_reuse | ? (no_summary(rc=255)) | ? (n/a_host_only) | ? (n/a_host_only) | ? (n/a_host_only) |

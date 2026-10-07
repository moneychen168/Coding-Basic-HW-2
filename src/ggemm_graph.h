// ggemm_graph.h -- plan / execute split for repeated calls with identical shapes (owner: agent-G).
//
// ggemm_make_plan   : planning + descriptor build + ONE H2D (plan-owned device blob) + stream
//                     capture of the launch sequence -> instantiated CUDA Graph. Allocates
//                     (plan-owned memory) and synchronizes: call it outside timed regions.
// ggemm_execute_plan: cudaGraphLaunch only (no host planning, no H2D, no allocation).
// ggemm_plan_update_pointers: same shapes, new A/B/C device pointers: patches the plan's device
//                     descriptors with one small H2D (no re-capture; see ggemm_graph.cu).
//
// Workspace / ordering contract:
//  * Descriptors + LPT schedule image (the only state the graph needs across calls) live in
//    plan-owned memory, so ggemm_grouped_dgemm on the same handle can never overwrite them.
//  * Residue / INT32 / shift scratch and the cuBLAS workspace are shared with the handle. They
//    are only live inside one execution, and every execution is ordered against the handle's
//    stream (h->stream): if `stream` != h->stream, execute makes `stream` wait for h->stream
//    and h->stream wait for the graph. So all users of one handle (plain calls, any plan, any
//    stream) form one total order and never overlap on the shared scratch.
//  * A plan must be destroyed before its handle. Plans are not thread-safe (same as handles).
#pragma once
#include "ggemm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggemm_plan_s* ggemm_plan_t;

// path: GGEMM_PATH_EMU_GROUPED or GGEMM_PATH_AUTO (others -> NULL). problems: HOST array as for
// ggemm_grouped_dgemm. Returns NULL on error; *err (optional) receives the GGEMM_ERR_* code.
ggemm_plan_t ggemm_make_plan(ggemm_handle_t h, ggemm_path_t path, const ggemm_config_t* cfg,
                             const ggemm_problem_t* problems, int num_problems);
ggemm_plan_t ggemm_make_plan_ex(ggemm_handle_t h, ggemm_path_t path, const ggemm_config_t* cfg,
                                const ggemm_problem_t* problems, int num_problems, int* err);

// Launches the instantiated graph on `stream` (any stream; ordered against the handle, above).
int ggemm_execute_plan(ggemm_plan_t plan, cudaStream_t stream);

// A/B/C: HOST arrays of num_problems DEVICE pointers (original problem order); any of the three
// may be NULL = keep. Shapes/ld/alpha/beta are frozen in the plan. The patch H2D is enqueued on
// the handle's stream, i.e. after every previously issued execute and before every later one.
int ggemm_plan_update_pointers(ggemm_plan_t plan, const double* const* A, const double* const* B,
                               double* const* C);

// Stats of the planned call (kernel_launches = launches inside the graph).
int ggemm_plan_get_stats(ggemm_plan_t plan, ggemm_stats_t* stats);

void ggemm_destroy_plan(ggemm_plan_t plan);

#ifdef __cplusplus
}
#endif

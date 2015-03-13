/*
 * Copyright © 2013 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \file brw_performance_query.c
 *
 * Implementation of the GL_INTEL_performance_query extension.
 *
 * Currently this driver only exposes the 64bit Pipeline Statistics Registers
 * available with Gen6 and Gen7.5, with support for Observability Counters
 * to be added later for Gen7.5+
 */

#include <linux/perf_event.h>

#include <limits.h>

#include <asm/unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "main/hash.h"
#include "main/macros.h"
#include "main/mtypes.h"
#include "main/performance_query.h"

#include "util/bitset.h"
#include "util/ralloc.h"

#include "brw_context.h"
#include "brw_defines.h"
#include "brw_performance_query.h"
#include "intel_batchbuffer.h"

#define FILE_DEBUG_FLAG DEBUG_PERFMON

/**
 * i965 representation of a performance query object.
 *
 * NB: We want to keep this structure relatively lean considering that
 * applications may expect to allocate enough objects to be able to
 * query around all draw calls in a frame.
 */
struct brw_perf_query_object
{
   /** The base class. */
   struct gl_perf_query_object base;

   const struct brw_perf_query *query;

   struct {
      /**
       * BO containing starting and ending snapshots for the
       * statistics counters.
       */
      drm_intel_bo *bo;

      /**
       * Storage for final pipeline statistics counter results.
       */
      uint64_t *results;

   } pipeline_stats;
};

/** Downcasting convenience macro. */
static inline struct brw_perf_query_object *
brw_perf_query(struct gl_perf_query_object *o)
{
   return (struct brw_perf_query_object *) o;
}

#define SECOND_SNAPSHOT_OFFSET_IN_BYTES 2048

/******************************************************************************/

static GLboolean brw_is_perf_query_ready(struct gl_context *,
					 struct gl_perf_query_object *);

static void
dump_perf_query_callback(GLuint id, void *query_void, void *brw_void)
{
   struct gl_perf_query_object *o = query_void;
   struct brw_perf_query_object *obj = query_void;

   switch(obj->query->kind) {
   case PIPELINE_STATS:
      DBG("%4d: %-6s %-8s BO: %-4s\n",
          id,
          o->Used ? "Dirty," : "New,",
          o->Active ? "Active," : (o->Ready ? "Ready," : "Pending,"),
          obj->pipeline_stats.bo ? "yes" : "no");
      break;
   }
}

void
brw_dump_perf_queries(struct brw_context *brw)
{
   struct gl_context *ctx = &brw->ctx;
   DBG("Queries: (Open queries = %d)\n",
       brw->perfquery.n_active_pipeline_stats_queries);
   _mesa_HashWalk(ctx->PerfQuery.Objects, dump_perf_query_callback, brw);
}

/******************************************************************************/

static void
brw_get_perf_query_info(struct gl_context *ctx,
                        int query_index,
                        const char **name,
                        GLuint *data_size,
                        GLuint *n_counters,
                        GLuint *n_active)
{
   struct brw_context *brw = brw_context(ctx);
   const struct brw_perf_query *query = &brw->perfquery.queries[query_index];

   *name = query->name;
   *data_size = query->data_size;
   *n_counters = query->n_counters;

   switch(query->kind) {
   case PIPELINE_STATS:
      *n_active = brw->perfquery.n_active_pipeline_stats_queries;
      break;
   }
}

static void
brw_get_perf_counter_info(struct gl_context *ctx,
                          int query_index,
                          int counter_index,
                          const char **name,
                          const char **desc,
                          GLuint *offset,
                          GLuint *data_size,
                          GLuint *type_enum,
                          GLuint *data_type_enum,
                          GLuint64 *raw_max)
{
   struct brw_context *brw = brw_context(ctx);
   const struct brw_perf_query *query = &brw->perfquery.queries[query_index];
   const struct brw_perf_query_counter *counter =
      &query->counters[counter_index];

   *name = counter->name;
   *desc = counter->desc;
   *offset = counter->offset;
   *data_size = counter->size;
   *type_enum = counter->type;
   *data_type_enum = counter->data_type;
   *raw_max = counter->raw_max;
}

/**
 * Take a snapshot of any queried pipeline statistics counters.
 */
static void
snapshot_statistics_registers(struct brw_context *brw,
                              struct brw_perf_query_object *obj,
                              uint32_t offset_in_bytes)
{
   const int offset = offset_in_bytes / sizeof(uint64_t);
   const struct brw_perf_query *query = obj->query;
   const int n_counters = query->n_counters;

   brw_emit_mi_flush(brw);

   for (int i = 0; i < n_counters; i++) {
      const struct brw_perf_query_counter *counter = &query->counters[i];

      assert(counter->data_type == GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL);

      brw_store_register_mem64(brw, obj->pipeline_stats.bo,
                               counter->pipeline_stat.reg,
                               offset + i);
   }
}

/**
 * Gather results from pipeline_stats_bo, storing the final values.
 *
 * This allows us to free pipeline_stats_bo (which is 4K) in favor of a much
 * smaller array of final results.
 */
static void
gather_statistics_results(struct brw_context *brw,
                          struct brw_perf_query_object *obj)
{
   const struct brw_perf_query *query = obj->query;
   const int n_counters = query->n_counters;

   obj->pipeline_stats.results = calloc(n_counters, sizeof(uint64_t));
   if (obj->pipeline_stats.results == NULL) {
      _mesa_error_no_memory(__func__);
      return;
   }

   drm_intel_bo_map(obj->pipeline_stats.bo, false);
   uint64_t *start = obj->pipeline_stats.bo->virtual;
   uint64_t *end = start + (SECOND_SNAPSHOT_OFFSET_IN_BYTES / sizeof(uint64_t));

   for (int i = 0; i < n_counters; i++) {
      const struct brw_perf_query_counter *counter = &query->counters[i];
      obj->pipeline_stats.results[i] = end[i] - start[i];

      if (counter->pipeline_stat.numerator !=
          counter->pipeline_stat.denominator) {
         obj->pipeline_stats.results[i] *= counter->pipeline_stat.numerator;
         obj->pipeline_stats.results[i] /= counter->pipeline_stat.denominator;
      }
   }

   drm_intel_bo_unmap(obj->pipeline_stats.bo);
   drm_intel_bo_unreference(obj->pipeline_stats.bo);
   obj->pipeline_stats.bo = NULL;
}

/******************************************************************************/

/**
 * Driver hook for glBeginPerfQueryINTEL().
 */
static GLboolean
brw_begin_perf_query(struct gl_context *ctx,
                     struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);

   assert(!o->Active);
   assert(!o->Used || o->Ready); /* no in-flight query to worry about */

   DBG("Begin(%d)\n", o->Id);

   switch(obj->query->kind) {
   case PIPELINE_STATS:
      if (obj->pipeline_stats.bo) {
         drm_intel_bo_unreference(obj->pipeline_stats.bo);
         obj->pipeline_stats.bo = NULL;
      }

      obj->pipeline_stats.bo =
         drm_intel_bo_alloc(brw->bufmgr, "perf. query stats bo", 4096, 64);

      /* Take starting snapshots. */
      snapshot_statistics_registers(brw, obj, 0);

      free(obj->pipeline_stats.results);
      obj->pipeline_stats.results = NULL;

      ++brw->perfquery.n_active_pipeline_stats_queries;
      break;
   }

   return true;
}

/**
 * Driver hook for glEndPerfQueryINTEL().
 */
static void
brw_end_perf_query(struct gl_context *ctx,
                     struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);

   DBG("End(%d)\n", o->Id);

   switch(obj->query->kind) {
   case PIPELINE_STATS:
      /* Take ending snapshots. */
      snapshot_statistics_registers(brw, obj,
                                    SECOND_SNAPSHOT_OFFSET_IN_BYTES);
      --brw->perfquery.n_active_pipeline_stats_queries;
      break;
   }
}

static void
brw_wait_perf_query(struct gl_context *ctx, struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);
   drm_intel_bo *bo = NULL;

   assert(!o->Ready);

   switch(obj->query->kind) {
   case PIPELINE_STATS:
      bo = obj->pipeline_stats.bo;
      break;
   }

   if (bo == NULL)
      return;

   /* If the current batch references our results bo then we need to
    * flush first... */
   if (drm_intel_bo_references(brw->batch.bo, bo))
      intel_batchbuffer_flush(brw);

   if (unlikely(brw->perf_debug)) {
      if (drm_intel_bo_busy(bo))
         perf_debug("Stalling GPU waiting for a performance query object.\n");
   }

   drm_intel_bo_wait_rendering(bo);
}

/**
 * Is a performance query result available?
 */
static GLboolean
brw_is_perf_query_ready(struct gl_context *ctx,
                        struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);

   if (o->Ready)
      return true;

   switch(obj->query->kind) {
   case PIPELINE_STATS:
      return (obj->pipeline_stats.bo &&
              !drm_intel_bo_references(brw->batch.bo, obj->pipeline_stats.bo) &&
              !drm_intel_bo_busy(obj->pipeline_stats.bo));
   }

   unreachable("missing ready check for unknown query kind");
   return false;
}

static int
get_pipeline_stats_data(struct brw_context *brw,
                        struct brw_perf_query_object *obj,
                        size_t data_size,
                        uint8_t *data)

{
   int n_counters = obj->query->n_counters;
   uint8_t *p = data;

   if (!obj->pipeline_stats.results) {
      gather_statistics_results(brw, obj);

      /* Check if we did really get the results */
      if (!obj->pipeline_stats.results)
         return 0;
   }

   for (int i = 0; i < n_counters; i++) {
      *((uint64_t *)p) = obj->pipeline_stats.results[i];
      p += 8;
   }

   return p - data;
}

/**
 * Get the performance query result.
 */
static void
brw_get_perf_query_data(struct gl_context *ctx,
                        struct gl_perf_query_object *o,
                        GLsizei data_size,
                        GLuint *data,
                        GLuint *bytes_written)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);
   int written = 0;

   assert(brw_is_perf_query_ready(ctx, o));

   DBG("GetData(%d)\n", o->Id);
   brw_dump_perf_queries(brw);

   /* This hook should only be called when results are available. */
   assert(o->Ready);

   switch(obj->query->kind) {
   case PIPELINE_STATS:
      written = get_pipeline_stats_data(brw, obj, data_size, (uint8_t *)data);
      break;
   }

   if (bytes_written)
      *bytes_written = written;
}

static struct gl_perf_query_object *
brw_new_perf_query_object(struct gl_context *ctx, int query_index)
{
   struct brw_context *brw = brw_context(ctx);
   const struct brw_perf_query *query = &brw->perfquery.queries[query_index];
   struct brw_perf_query_object *obj =
      calloc(1, sizeof(struct brw_perf_query_object));

   if (!obj)
      return NULL;

   obj->query = query;

   return &obj->base;
}

/**
 * Delete a performance query object.
 */
static void
brw_delete_perf_query(struct gl_context *ctx,
                      struct gl_perf_query_object *o)
{
   struct brw_perf_query_object *obj = brw_perf_query(o);

   assert(!o->Active);
   assert(!o->Used || o->Ready); /* no in-flight query to worry about */

   DBG("Delete(%d)\n", o->Id);

   switch(obj->query->kind) {
   case PIPELINE_STATS:
      if (obj->pipeline_stats.bo) {
         drm_intel_bo_unreference(obj->pipeline_stats.bo);
         obj->pipeline_stats.bo = NULL;
      }

      free(obj->pipeline_stats.results);
      obj->pipeline_stats.results = NULL;
      break;
   }

   free(obj);
}

#define SCALED_NAMED_STAT(REG, NUM, DEN, NAME, DESC)        \
   {                                                        \
      .name = NAME,                                         \
      .desc = DESC,                                         \
      .type = GL_PERFQUERY_COUNTER_RAW_INTEL,               \
      .data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL,  \
      .size = sizeof(uint64_t),                             \
      .pipeline_stat.reg = REG,                             \
      .pipeline_stat.numerator = NUM,                       \
      .pipeline_stat.denominator = DEN,                     \
   }
#define NAMED_STAT(REG, NAME, DESC)    SCALED_NAMED_STAT(REG, 1, 1, NAME, DESC)
#define STAT(REG, DESC)                SCALED_NAMED_STAT(REG, 1, 1, #REG, DESC)
#define SCALED_STAT(REG, N, D, DESC)   SCALED_NAMED_STAT(REG, N, D, #REG, DESC)

static struct brw_perf_query_counter gen6_pipeline_statistics[] = {
   STAT(IA_VERTICES_COUNT,   "N vertices submitted"),
   STAT(IA_PRIMITIVES_COUNT, "N primitives submitted"),
   STAT(VS_INVOCATION_COUNT, "N vertex shader invocations"),
   STAT(GS_INVOCATION_COUNT, "N geometry shader invocations"),
   STAT(GS_PRIMITIVES_COUNT, "N geometry shader primitives emitted"),
   STAT(CL_INVOCATION_COUNT, "N primitives entering clipping"),
   STAT(CL_PRIMITIVES_COUNT, "N primitives leaving clipping"),
   STAT(PS_INVOCATION_COUNT, "N fragment shader invocations"),
   STAT(PS_DEPTH_COUNT,      "N z-pass fragments"),

   NAMED_STAT(GEN6_SO_PRIM_STORAGE_NEEDED, "SO_PRIM_STORAGE_NEEDED",
              "N geometry shader stream-out primitives (total)"),
   NAMED_STAT(GEN6_SO_NUM_PRIMS_WRITTEN,   "SO_NUM_PRIMS_WRITTEN",
              "N geometry shader stream-out primitives (written)"),
};

static struct brw_perf_query_counter gen7_pipeline_statistics[] = {

   STAT(IA_VERTICES_COUNT,   "N vertices submitted"),
   STAT(IA_PRIMITIVES_COUNT, "N primitives submitted"),
   STAT(VS_INVOCATION_COUNT, "N vertex shader invocations"),
   STAT(HS_INVOCATION_COUNT, "N hull shader invocations"),
   STAT(DS_INVOCATION_COUNT, "N domain shader invocations"),
   STAT(GS_INVOCATION_COUNT, "N geometry shader invocations"),
   STAT(GS_PRIMITIVES_COUNT, "N geometry shader primitives emitted"),
   STAT(CL_INVOCATION_COUNT, "N primitives entering clipping"),
   STAT(CL_PRIMITIVES_COUNT, "N primitives leaving clipping"),

   /* Implement the "WaDividePSInvocationCountBy4:HSW,BDW" workaround:
    * "Invocation counter is 4 times actual.  WA: SW to divide HW reported
    *  PS Invocations value by 4."
    *
    * Prior to Haswell, invocation count was counted by the WM, and it
    * buggily counted invocations in units of subspans (2x2 unit). To get the
    * correct value, the CS multiplied this by 4. With HSW the logic moved,
    * and correctly emitted the number of pixel shader invocations, but,
    * whomever forgot to undo the multiply by 4.
    */
   SCALED_STAT(PS_INVOCATION_COUNT, 1, 4, "N fragment shader invocations"),

   STAT(PS_DEPTH_COUNT,      "N z-pass fragments"),

   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(0), "SO_PRIM_STORAGE_NEEDED (Stream 0)",
              "N stream-out (stream 0) primitives (total)"),
   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(1), "SO_PRIM_STORAGE_NEEDED (Stream 1)",
              "N stream-out (stream 1) primitives (total)"),
   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(2), "SO_PRIM_STORAGE_NEEDED (Stream 2)",
              "N stream-out (stream 2) primitives (total)"),
   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(3), "SO_PRIM_STORAGE_NEEDED (Stream 3)",
              "N stream-out (stream 3) primitives (total)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(0), "SO_NUM_PRIMS_WRITTEN (Stream 0)",
              "N stream-out (stream 0) primitives (written)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(1), "SO_NUM_PRIMS_WRITTEN (Stream 1)",
              "N stream-out (stream 1) primitives (written)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(2), "SO_NUM_PRIMS_WRITTEN (Stream 2)",
              "N stream-out (stream 2) primitives (written)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(3), "SO_NUM_PRIMS_WRITTEN (Stream 3)",
              "N stream-out (stream 3) primitives (written)"),
};

#undef STAT
#undef NAMED_STAT

static void
add_pipeline_statistics_query(struct brw_context *brw,
                              const char *name,
                              struct brw_perf_query_counter *counters,
                              int n_counters)
{
   struct brw_perf_query *query =
      &brw->perfquery.queries[brw->perfquery.n_queries++];

   query->kind = PIPELINE_STATS;
   query->name = name;
   query->data_size = sizeof(uint64_t) * n_counters;
   query->n_counters = n_counters;
   query->counters = counters;

   for (int i = 0; i < n_counters; i++) {
      struct brw_perf_query_counter *counter = &counters[i];
      counter->offset = sizeof(uint64_t) * i;
   }
}

void
brw_init_performance_queries(struct brw_context *brw)
{
   struct gl_context *ctx = &brw->ctx;

   ctx->Driver.GetPerfQueryInfo = brw_get_perf_query_info;
   ctx->Driver.GetPerfCounterInfo = brw_get_perf_counter_info;
   ctx->Driver.NewPerfQueryObject = brw_new_perf_query_object;
   ctx->Driver.DeletePerfQuery = brw_delete_perf_query;
   ctx->Driver.BeginPerfQuery = brw_begin_perf_query;
   ctx->Driver.EndPerfQuery = brw_end_perf_query;
   ctx->Driver.WaitPerfQuery = brw_wait_perf_query;
   ctx->Driver.IsPerfQueryReady = brw_is_perf_query_ready;
   ctx->Driver.GetPerfQueryData = brw_get_perf_query_data;

   if (brw->gen == 6) {
      add_pipeline_statistics_query(brw, "Gen6 Pipeline Statistics Registers",
                                    gen6_pipeline_statistics,
                                    (sizeof(gen6_pipeline_statistics)/
                                     sizeof(gen6_pipeline_statistics[0])));
   } else if (brw->gen == 7) {
      add_pipeline_statistics_query(brw, "Gen7 Pipeline Statistics Registers",
                                    gen7_pipeline_statistics,
                                    (sizeof(gen7_pipeline_statistics)/
                                     sizeof(gen7_pipeline_statistics[0])));
   }

   ctx->PerfQuery.NumQueries = brw->perfquery.n_queries;
}

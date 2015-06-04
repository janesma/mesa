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
 * Currently there are two possible counter sources exposed here:
 *
 * On Gen6+ hardware we have numerous 64bit Pipeline Statistics Registers
 * that we can snapshot at the beginning and end of a query.
 *
 * On Gen7.5+ we have Observability Architecture counters which are
 * covered in separate document from the rest of the PRMs.  It is available at:
 * https://01.org/linuxgraphics/documentation/driver-documentation-prms
 * => 2013 Intel Core Processor Family => Observability Performance Counters
 * (This one volume covers Sandybridge, Ivybridge, Baytrail, and Haswell,
 * though notably we currently only support OA counters for Haswell+)
 */

#include <limits.h>
#include <dirent.h>

#include <asm/unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <xf86drm.h>
#include <i915_drm.h>

#include "main/hash.h"
#include "main/macros.h"
#include "main/mtypes.h"
#include "main/performance_query.h"

#include "util/bitset.h"
#include "util/ralloc.h"
#include "util/hash_table.h"

#include "glsl/list.h"

#include "brw_context.h"
#include "brw_defines.h"
#include "brw_performance_query.h"
#include "brw_oa_hsw.h"
#include "intel_batchbuffer.h"

#define FILE_DEBUG_FLAG DEBUG_PERFMON

#define MAX_OA_REPORT_COUNTERS 62

/* Samples read from i915 perf file descriptor */
struct oa_sample {
   struct drm_i915_perf_record_header header;
   uint8_t oa_report[];
};
#define MAX_OA_SAMPLE_SIZE (8 +   /* drm_i915_perf_record_header */ \
                            256)  /* OA counter report */

/**
 * Periodic OA samples are read into these buffer structures that are
 * appended to the brw->perfquery.sample_buffers linked list.
 *
 * NB: Periodic OA reports may relate to multiple queries and queries
 * take references on a tail of this linked list by incrementing
 * buf->refcount for one buffer in the list.
 *
 * A reference on any buffer effectively holds a reference on all
 * following buffers and query with a NULL tail pointer effectively
 * holds a reference on all buffers.
 *
 * After a query accumulates its samples it drops its tail reference
 * and we remove any unneeded buffers from the tail of the list.
 *
 * Once we are finished with a sample buffer we cache the buffer for
 * re-use until the next point when all query objects are deleted.
 */
struct brw_oa_sample_buf {
   struct exec_node link;
   int refcount;
   int len;
   uint8_t buf[MAX_OA_SAMPLE_SIZE * 15];
};

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

   /* See query->kind to know which state below is in use... */
   union {
      struct {

         /**
          * BO containing OA counter snapshots at query Begin/End time.
          */
         drm_intel_bo *bo;
         int current_report_id;

         /**
          * Reference into brw->perfquery.sample_buffers list.
          * (See struct brw_oa_sample_buf description for more details)
          */
         struct exec_node *samples_tail;

         /**
          * Storage for the final accumulated OA counters.
          */
         uint64_t accumulator[MAX_OA_REPORT_COUNTERS];

         /**
          * false while in the unresolved_elements list, and set to true when
          * the final, end MI_RPC snapshot has been accumulated.
          */
         bool results_accumulated;

      } oa;

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
   struct gl_context *ctx = brw_void;
   struct gl_perf_query_object *o = query_void;
   struct brw_perf_query_object *obj = query_void;

   switch(obj->query->kind) {
   case OA_COUNTERS:
      DBG("%4d: %-6s %-8s BO: %-4s OA data: %-10s %-15s\n",
          id,
          o->Used ? "Dirty," : "New,",
          o->Active ? "Active," : (o->Ready ? "Ready," : "Pending,"),
          obj->oa.bo ? "yes," : "no,",
          brw_is_perf_query_ready(ctx, o) ? "ready," : "not ready,",
          obj->oa.results_accumulated ? "accumulated" : "not accumulated");
      break;
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
   DBG("Queries: (Open queries = %d, OA users = %d)\n",
       brw->perfquery.n_active_oa_queries, brw->perfquery.n_oa_users);
   _mesa_HashWalk(ctx->PerfQuery.Objects, dump_perf_query_callback, brw);
}

/******************************************************************************/

static struct brw_oa_sample_buf *
append_empty_sample_buf(struct brw_context *brw)
{
   struct exec_node *node = exec_list_pop_head(&brw->perfquery.free_sample_buffers);
   struct brw_oa_sample_buf *buf;

   if (node)
      buf = exec_node_data(struct brw_oa_sample_buf, node, link);
   else {
      buf = ralloc_size(brw, sizeof(*buf));

      exec_node_init(&buf->link);
      buf->refcount = 0;
      buf->len = 0;
   }

   exec_list_push_tail(&brw->perfquery.sample_buffers, &buf->link);

   return buf;
}

static void
reap_sample_buffer(struct brw_context *brw, struct brw_oa_sample_buf *buf)
{
   exec_node_remove(&buf->link);
   exec_list_push_tail(&brw->perfquery.free_sample_buffers, &buf->link);
}

static void
reap_tail_sample_buffers(struct brw_context *brw)
{
   /* Queries that have no tail pointer, effectively hold a reference
    * to the full list of samples... */
   for (int i = 0; i < brw->perfquery.unresolved_elements; i++) {
      struct brw_perf_query_object *obj = brw->perfquery.unresolved[i];

      if (!obj->oa.samples_tail)
         return;
   }

   foreach_list_typed_safe(struct brw_oa_sample_buf, buf, link,
                           &brw->perfquery.sample_buffers)
   {
      if (buf->refcount == 0)
         reap_sample_buffer(brw, buf);
      else
         return;
   }
}

static void
free_sample_bufs(struct brw_context *brw)
{
   foreach_list_typed_safe(struct brw_oa_sample_buf, buf, link,
                           &brw->perfquery.free_sample_buffers)
      ralloc_free(buf);

   exec_list_make_empty(&brw->perfquery.free_sample_buffers);
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
   case OA_COUNTERS:
      *n_active = brw->perfquery.n_active_oa_queries;
      break;

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
 * Emit an MI_REPORT_PERF_COUNT command packet.
 *
 * This writes the current OA counter values to buffer.
 */
static void
emit_mi_report_perf_count(struct brw_context *brw,
                          drm_intel_bo *bo,
                          uint32_t offset_in_bytes,
                          uint32_t report_id)
{
   assert(offset_in_bytes % 64 == 0);

   /* Reports apparently don't always get written unless we flush first. */
   brw_emit_mi_flush(brw);

   if (brw->gen < 8) {
      BEGIN_BATCH(3);
      OUT_BATCH(GEN6_MI_REPORT_PERF_COUNT);
      OUT_RELOC(bo, I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
                offset_in_bytes);
      OUT_BATCH(report_id);
      ADVANCE_BATCH();
   } else
      unreachable("Unsupported generation for OA performance counters.");

   /* Reports apparently don't always get written unless we flush after. */
   brw_emit_mi_flush(brw);
}

/**
 * Add a query to the global list of "unresolved queries."
 *
 * Queries are "unresolved" until all the counter snapshots have been
 * accumulated via accumulate_oa_snapshots() after the end MI_REPORT_PERF_COUNT
 * has landed in query->oa.bo.
 */
static void
add_to_unresolved_query_list(struct brw_context *brw,
                             struct brw_perf_query_object *obj)
{
   if (brw->perfquery.unresolved_elements >=
       brw->perfquery.unresolved_array_size) {
      brw->perfquery.unresolved_array_size *= 1.5;
      brw->perfquery.unresolved = reralloc(brw, brw->perfquery.unresolved,
                                           struct brw_perf_query_object *,
                                           brw->perfquery.unresolved_array_size);
   }

   brw->perfquery.unresolved[brw->perfquery.unresolved_elements++] = obj;
}

/**
 * Remove a query from the global list of "unresolved queries." once
 * the end MI_RPC OA counter snapshot has been accumulated, or when
 * discarding unwanted query results.
 */
static void
drop_from_unresolved_query_list(struct brw_context *brw,
                                struct brw_perf_query_object *obj)
{
   for (int i = 0; i < brw->perfquery.unresolved_elements; i++) {
      if (brw->perfquery.unresolved[i] == obj) {
         int last_elt = --brw->perfquery.unresolved_elements;

         if (i == last_elt)
            brw->perfquery.unresolved[i] = NULL;
         else
            brw->perfquery.unresolved[i] = brw->perfquery.unresolved[last_elt];

         break;
      }
   }

   if (obj->oa.samples_tail) {
      struct brw_oa_sample_buf *buf =
         exec_node_data(struct brw_oa_sample_buf, obj->oa.samples_tail, link);

      assert(buf->refcount > 0);
      buf->refcount--;

      obj->oa.samples_tail = NULL;
   }

   reap_tail_sample_buffers(brw);
}

static void
init_oa_sys_vars(struct brw_context *brw)
{
   const struct brw_device_info *info = brw->intelScreen->devinfo;
   int threads_per_eu = 7;

   brw->perfquery.sys_vars.timestamp_frequency = 12500000;

   if (brw->is_haswell) {
      if (info->gt == 1) {
         brw->perfquery.sys_vars.n_eus = 10;
         brw->perfquery.sys_vars.n_eu_slices = 1;
         brw->perfquery.sys_vars.slice_mask = 0x1;
         brw->perfquery.sys_vars.subslice_mask = 0x1;
      } else if (info->gt == 2) {
         brw->perfquery.sys_vars.n_eus = 20;
         brw->perfquery.sys_vars.n_eu_slices = 1;
         brw->perfquery.sys_vars.slice_mask = 0x1;
         brw->perfquery.sys_vars.subslice_mask = 0x3;
      } else if (info->gt == 3) {
         brw->perfquery.sys_vars.n_eus = 40;
         brw->perfquery.sys_vars.n_eu_slices = 2;
         brw->perfquery.sys_vars.slice_mask = 0x3;
         brw->perfquery.sys_vars.subslice_mask = 0xf;
      }
   } else {
      brw->perfquery.sys_vars.n_eus = 0;
      brw->perfquery.sys_vars.n_eu_slices = 0;
      brw->perfquery.sys_vars.slice_mask = 0;
      brw->perfquery.sys_vars.subslice_mask = 0;
   }

   brw->perfquery.sys_vars.eu_threads_count =
      brw->perfquery.sys_vars.n_eus * threads_per_eu;
}

static uint64_t
read_report_timestamp(struct brw_context *brw, uint32_t *report)
{
   uint64_t tmp = ((uint64_t)report[1]) * 1000000000;

   return tmp ? tmp / brw->perfquery.sys_vars.timestamp_frequency : 0;
}

static void
accumulate_uint32(const uint32_t *report0,
                  const uint32_t *report1,
                  uint64_t *accumulator)
{
   *accumulator += (uint32_t)(*report1 - *report0);
}

/**
 * Given pointers to starting and ending OA snapshots, add the deltas for each
 * counter to the results.
 */
static void
add_deltas(struct brw_context *brw,
           struct brw_perf_query_object *obj,
           const uint32_t *start,
           const uint32_t *end)
{
   const struct brw_perf_query *query = obj->query;
   uint64_t *accumulator = obj->oa.accumulator;
   int i;

   switch (query->oa_format) {
   case I915_OA_FORMAT_A45_B8_C8:
      accumulate_uint32(start + 1, end + 1, accumulator); /* timestamp */

      for (i = 0; i < 61; i++)
         accumulate_uint32(start + 3 + i, end + 3 + i, accumulator + 1 + i);

      break;
   default:
      unreachable("Can't accumulate OA counters in unknown format");
   }
}

static bool
inc_n_oa_users(struct brw_context *brw)
{
   if (brw->perfquery.n_oa_users == 0 &&
       drmIoctl(brw->perfquery.oa_stream_fd,
                I915_PERF_IOCTL_ENABLE, 0) < 0)
   {
      return false;
   }
   ++brw->perfquery.n_oa_users;

   return true;
}

static void
dec_n_oa_users(struct brw_context *brw)
{
   /* Disabling the i915 perf stream will effectively disable the OA
    * counters.  Note it's important to be sure there are no outstanding
    * MI_RPC commands at this point since they could stall the CS
    * indefinitely once OACONTROL is disabled.
    */
   --brw->perfquery.n_oa_users;
   if (brw->perfquery.n_oa_users == 0 &&
       drmIoctl(brw->perfquery.oa_stream_fd,
                I915_PERF_IOCTL_DISABLE, 0) < 0)
   {
      DBG("WARNING: Error disabling i915 perf stream: %m\n");
   }
}

/* In general if we see anything spurious while accumulating results,
 * we don't try and continue accumulating the current query, hoping
 * for the best, we scrap anything outstanding, and then hope for the
 * best with new queries. */
static void
discard_all_queries(struct brw_context *brw)
{
   while (brw->perfquery.unresolved_elements) {
      struct brw_perf_query_object *obj = brw->perfquery.unresolved[0];

      obj->oa.results_accumulated = true;
      drop_from_unresolved_query_list(brw, brw->perfquery.unresolved[0]);

      dec_n_oa_users(brw);
   }
}

static bool
read_oa_samples(struct brw_context *brw)
{
   while (1) {
      struct brw_oa_sample_buf *buf = append_empty_sample_buf(brw);
      int len;

      while ((len = read(brw->perfquery.oa_stream_fd, buf->buf,
                         sizeof(buf->buf))) < 0 && errno == EINTR)
         ;

      if (len <= 0) {
         reap_sample_buffer(brw, buf);

         if (len < 0) {
            if (errno == EAGAIN)
               return true;
            else {
               DBG("Error reading i915 perf samples: %m\n");
               return false;
            }
         } else {
            DBG("Spurios EOF reading i915 perf samples: %m\n");
            return false;
         }
      }

      buf->len = len;
   }

   unreachable("not reached");
   return false;
}

/**
 * Accumulate OA counter results from a series of snapshots.
 *
 * N.B. We write snapshots for the beginning and end of a query into
 * query->oa.bo as well as collect periodic snapshots from the Linux
 * perf interface.
 *
 * These periodic snapshots help to ensure we handle counter overflow
 * correctly by being frequent enough to ensure we don't miss multiple
 * overflows of a counter between snapshots.
 */
static void
accumulate_oa_snapshots(struct brw_context *brw,
                        struct brw_perf_query_object *obj)
{
   struct gl_perf_query_object *o = &obj->base;
   uint32_t *query_buffer;
   uint32_t *start;
   uint64_t start_timestamp;
   uint32_t *last;
   uint32_t *end;
   uint64_t end_timestamp;

   assert(o->Ready);

   if (!read_oa_samples(brw))
      goto error;

   drm_intel_bo_map(obj->oa.bo, false);
   query_buffer = obj->oa.bo->virtual;

   start = last = query_buffer;
   end = query_buffer + (SECOND_SNAPSHOT_OFFSET_IN_BYTES / sizeof(uint32_t));

   if (start[0] != obj->oa.current_report_id) {
      DBG("Spurious start report id=%"PRIu32"\n", start[0]);
      goto error;
   }
   if (end[0] != (obj->oa.current_report_id + 1)) {
      DBG("Spurious end report id=%"PRIu32"\n", end[0]);
      goto error;
   }

   start_timestamp = read_report_timestamp(brw, start);
   end_timestamp = read_report_timestamp(brw, end);

   /* See if we have any periodic reports to accumulate too... */

   foreach_list_typed(struct brw_oa_sample_buf, buf, link,
                      &brw->perfquery.sample_buffers)
   {
      int offset = 0;

      while (offset < buf->len) {
         const struct drm_i915_perf_record_header *header =
            (const struct drm_i915_perf_record_header *)(buf->buf + offset);

         assert(header->size != 0);
         assert(header->size <= buf->len);

         offset += header->size;

         switch (header->type) {
         case DRM_I915_PERF_RECORD_SAMPLE: {
            struct oa_sample *sample = (struct oa_sample *)header;
            uint32_t *report = (uint32_t *)sample->oa_report;
            uint64_t timestamp = read_report_timestamp(brw, report);

            if (timestamp >= end_timestamp)
               goto end;

            if (timestamp > start_timestamp) {
               add_deltas(brw, obj, last, report);
               last = report;
            }

            break;
         }

         case DRM_I915_PERF_RECORD_OA_BUFFER_LOST:
             DBG("i915 perf: OA error: all reports lost\n");
             break;
         case DRM_I915_PERF_RECORD_OA_REPORT_LOST:
             DBG("i915 perf: OA report lost\n");
             break;

         default:
            DBG("i915 perf: Spurious header type = %d\n", header->type);
         }
      }
   }

end:

   add_deltas(brw, obj, last, end);

   DBG("Marking %d resolved - results gathered\n", o->Id);

   drm_intel_bo_unmap(obj->oa.bo);
   obj->oa.results_accumulated = true;
   drop_from_unresolved_query_list(brw, obj);
   dec_n_oa_users(brw);

   return;

error:

   drm_intel_bo_unmap(obj->oa.bo);
   discard_all_queries(brw);
}

/******************************************************************************/

static bool
open_i915_perf_oa_stream(struct brw_context *brw,
                         int metrics_set_id,
                         int report_format,
                         int period_exponent,
                         int drm_fd,
                         uint32_t ctx_id)
{
   uint64_t properties[] = {
      /* Single context sampling */
      DRM_I915_PERF_PROP_CTX_HANDLE, ctx_id,

      /* Include OA reports in samples */
      DRM_I915_PERF_PROP_SAMPLE_OA, true,

      /* OA unit configuration */
      DRM_I915_PERF_PROP_OA_METRICS_SET, metrics_set_id,
      DRM_I915_PERF_PROP_OA_FORMAT, report_format,
      DRM_I915_PERF_PROP_OA_EXPONENT, period_exponent,
   };
   struct drm_i915_perf_open_param param = {
      .flags = I915_PERF_FLAG_FD_CLOEXEC |
               I915_PERF_FLAG_FD_NONBLOCK |
               I915_PERF_FLAG_DISABLED,
      .num_properties = ARRAY_SIZE(properties) / 2,
      .properties_ptr = (uint64_t)properties
   };
   int fd = drmIoctl(drm_fd, DRM_IOCTL_I915_PERF_OPEN, &param);
   if (fd == -1) {
      DBG("Error opening i915 perf OA stream: %m\n");
      return false;
   }

   brw->perfquery.oa_stream_fd = fd;

   brw->perfquery.current_oa_metrics_set_id = metrics_set_id;
   brw->perfquery.current_oa_format = report_format;

   return true;
}

/**
 * Driver hook for glBeginPerfQueryINTEL().
 */
static GLboolean
brw_begin_perf_query(struct gl_context *ctx,
                     struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);
   const struct brw_perf_query *query = obj->query;

   assert(!o->Active);
   assert(!o->Used || o->Ready); /* no in-flight query to worry about */

   DBG("Begin(%d)\n", o->Id);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      /* If the OA counters aren't already on, enable them. */
      if (brw->perfquery.oa_stream_fd == -1) {
         __DRIscreen *screen = brw->intelScreen->driScrnPriv;
         uint32_t ctx_id = drm_intel_gem_context_get_context_id(brw->hw_ctx);
         int period_exponent;

         /* The timestamp for HSW+ increments every 80ns
          *
          * The period_exponent gives a sampling period as follows:
          *   sample_period = 80ns * 2^(period_exponent + 1)
          *
          * The overflow period for Haswell can be calculated as:
          *
          * 2^32 / (n_eus * max_gen_freq * 2)
          * (E.g. 40 EUs @ 1GHz = ~53ms)
          *
          * We currently sample every 42 milliseconds...
          */
         period_exponent = 18;

         if (!open_i915_perf_oa_stream(brw,
                                       query->oa_metrics_set_id,
                                       query->oa_format,
                                       period_exponent,
                                       screen->fd, /* drm fd */
                                       ctx_id))
            return false;
      } else {
         /* Opening an i915 perf stream implies exclusive access to
          * the OA unit which will generate counter reports for a
          * specific counter set with a specific layout/format so we
          * can't begin any OA based queries that require a different
          * counter set or format unless we get an opportunity to
          * close the stream and open a new one...
          */
         if (brw->perfquery.current_oa_metrics_set_id !=
             query->oa_metrics_set_id ||
             brw->perfquery.current_oa_format != query->oa_format)
         {
            return false;
         }
      }

      if (!inc_n_oa_users(brw)) {
         DBG("WARNING: Error enabling i915 perf stream: %m\n");
         return false;
      }

      if (obj->oa.bo) {
         drm_intel_bo_unreference(obj->oa.bo);
         obj->oa.bo = NULL;
      }

      obj->oa.bo =
         drm_intel_bo_alloc(brw->bufmgr, "perf. query OA bo", 4096, 64);
#ifdef DEBUG
      /* Pre-filling the BO helps debug whether writes landed. */
      drm_intel_bo_map(obj->oa.bo, true);
      memset((char *) obj->oa.bo->virtual, 0x80, 4096);
      drm_intel_bo_unmap(obj->oa.bo);
#endif

      obj->oa.current_report_id = brw->perfquery.next_query_start_report_id;
      brw->perfquery.next_query_start_report_id += 2;

      /* Take a starting OA counter snapshot. */
      emit_mi_report_perf_count(brw, obj->oa.bo, 0,
                                obj->oa.current_report_id);
      ++brw->perfquery.n_active_oa_queries;

      obj->oa.samples_tail = exec_list_get_head(&brw->perfquery.sample_buffers);
      if (obj->oa.samples_tail) {
         struct brw_oa_sample_buf *buf =
            exec_node_data(struct brw_oa_sample_buf, obj->oa.samples_tail, link);

         buf->refcount++;
      }

      memset(obj->oa.accumulator, 0, sizeof(obj->oa.accumulator));
      obj->oa.results_accumulated = false;

      add_to_unresolved_query_list(brw, obj);
      break;

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

   if (INTEL_DEBUG & DEBUG_PERFMON)
      brw_dump_perf_queries(brw);

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
   case OA_COUNTERS:

      /* NB: It's possible that the query will have already been marked
       * as 'accumulated' if an error was seen while reading samples
       * from perf. In this case we mustn't try and emit a closing
       * MI_RPC command in case the OA unit has already been disabled
       */
      if (!obj->oa.results_accumulated) {
         /* Take an ending OA counter snapshot. */
         emit_mi_report_perf_count(brw, obj->oa.bo,
                                   SECOND_SNAPSHOT_OFFSET_IN_BYTES,
                                   obj->oa.current_report_id + 1);
      }

      --brw->perfquery.n_active_oa_queries;

      /* NB: even though the query has now ended, it can't be resolved
       * until the end MI_REPORT_PERF_COUNT snapshot has been written
       * to query->oa.bo */
      break;

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
   case OA_COUNTERS:
      bo = obj->oa.bo;
      break;

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
   case OA_COUNTERS:
      return (obj->oa.results_accumulated ||
              (obj->oa.bo &&
               !drm_intel_bo_references(brw->batch.bo, obj->oa.bo) &&
               !drm_intel_bo_busy(obj->oa.bo)));

   case PIPELINE_STATS:
      return (obj->pipeline_stats.bo &&
              !drm_intel_bo_references(brw->batch.bo, obj->pipeline_stats.bo) &&
              !drm_intel_bo_busy(obj->pipeline_stats.bo));
   }

   unreachable("missing ready check for unknown query kind");
   return false;
}

static int
get_oa_counter_data(struct brw_context *brw,
                    struct brw_perf_query_object *obj,
                    size_t data_size,
                    uint8_t *data)
{
   const struct brw_perf_query *query = obj->query;
   int n_counters = query->n_counters;
   int written = 0;

   if (!obj->oa.results_accumulated) {
      accumulate_oa_snapshots(brw, obj);
      assert(obj->oa.results_accumulated);
   }

   for (int i = 0; i < n_counters; i++) {
      const struct brw_perf_query_counter *counter = &query->counters[i];
      uint64_t *out_uint64;
      float *out_float;

      if (counter->size) {
         switch (counter->data_type) {
         case GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL:
            out_uint64 = (uint64_t *)(data + counter->offset);
            *out_uint64 = counter->oa_counter_read_uint64(brw, query,
                                                          obj->oa.accumulator);
            break;
         case GL_PERFQUERY_COUNTER_DATA_FLOAT_INTEL:
            out_float = (float *)(data + counter->offset);
            *out_float = counter->oa_counter_read_float(brw, query,
                                                        obj->oa.accumulator);
            break;
         default:
            /* So far we aren't using uint32, double or bool32... */
            unreachable("unexpected counter data type");
         }
         written = counter->offset + counter->size;
      }
   }

   return written;
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

   if (INTEL_DEBUG & DEBUG_PERFMON)
      brw_dump_perf_queries(brw);

   /* This hook should only be called when results are available. */
   assert(o->Ready);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      written = get_oa_counter_data(brw, obj, data_size, (uint8_t *)data);
      break;

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

   brw->perfquery.n_query_instances++;

   return &obj->base;
}

static void
close_perf(struct brw_context *brw)
{
   if (brw->perfquery.oa_stream_fd != -1) {
      close(brw->perfquery.oa_stream_fd);
      brw->perfquery.oa_stream_fd = -1;

      free_sample_bufs(brw);
   }
}

/**
 * Delete a performance query object.
 */
static void
brw_delete_perf_query(struct gl_context *ctx,
                      struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);

   assert(!o->Active);
   assert(!o->Used || o->Ready); /* no in-flight query to worry about */

   DBG("Delete(%d)\n", o->Id);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      if (obj->oa.bo) {
         if (!obj->oa.results_accumulated) {
            drop_from_unresolved_query_list(brw, obj);
            dec_n_oa_users(brw);
         }

         drm_intel_bo_unreference(obj->oa.bo);
         obj->oa.bo = NULL;
      }

      obj->oa.results_accumulated = false;
      break;

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

   if (--brw->perfquery.n_query_instances == 0)
      close_perf(brw);
}

/******************************************************************************/

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

static struct brw_perf_query *
append_query(struct brw_context *brw)
{
   brw->perfquery.queries =
      reralloc(brw, brw->perfquery.queries,
               struct brw_perf_query, ++brw->perfquery.n_queries);

   return &brw->perfquery.queries[brw->perfquery.n_queries - 1];
}

static void
add_pipeline_statistics_query(struct brw_context *brw,
                              const char *name,
                              struct brw_perf_query_counter *counters,
                              int n_counters)
{
   struct brw_perf_query *query = append_query(brw);

   if (!query)
      return;

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

static bool
read_file_uint64(const char *file, uint64_t *val)
{
    char buf[32];
    int fd, n;

    fd = open(file, 0);
    if (fd < 0)
	return false;
    n = read(fd, buf, sizeof (buf) - 1);
    close(fd);
    if (n < 0)
	return false;

    buf[n] = '\0';
    *val = strtoull(buf, NULL, 0);

    return true;
}

static void
enumerate_sysfs_metrics(struct brw_context *brw)
{
   __DRIscreen *screen = brw->intelScreen->driScrnPriv;
   struct stat sb;
   int min, maj;
   char buf[128];
   DIR *drmdir, *metricsdir = NULL;
   int name_max;
   int entry_len;
   struct dirent *entry0, *entry1;
   struct dirent *drm_entry;
   struct dirent *metric_entry;

   if (fstat(screen->fd, &sb)) {
      DBG("Failed to stat DRM fd\n");
      return;
   }

   maj = major(sb.st_rdev);
   min = minor(sb.st_rdev);

   if (!S_ISCHR(sb.st_mode)) {
      DBG("DRM fd is not a character device as expected\n");
      return;
   }

   snprintf(buf, sizeof(buf), "/sys/dev/char/%d:%d/device/drm", maj, min);

   drmdir = opendir(buf);
   if (!drmdir) {
      DBG("Failed to open %s: %m\n", buf);
      return;
   }

   name_max = pathconf(buf, _PC_NAME_MAX);
   if (name_max == -1) /* Limit not defined, or error */
      name_max = 255; /* Take a guess */

   entry_len = offsetof(struct dirent, d_name) + name_max + 1;
   entry0 = malloc(entry_len);
   entry1 = malloc(entry_len);

   while (readdir_r(drmdir, entry0, &drm_entry) == 0 && drm_entry != NULL) {
      if (drm_entry->d_type == DT_DIR &&
          strncmp(drm_entry->d_name, "card", 4) == 0)
      {
         snprintf(buf, sizeof(buf),
                  "/sys/dev/char/%d:%d/device/drm/%s/metrics",
                  maj, min, drm_entry->d_name);

         metricsdir = opendir(buf);
         if (!metricsdir) {
            DBG("Failed to open %s: %m\n", buf);
            goto close_drm_dir;
         }

         while (readdir_r(metricsdir, entry1, &metric_entry) == 0 &&
                metric_entry != NULL)
         {
            struct hash_entry *entry;

            if (metric_entry->d_type != DT_DIR ||
                metric_entry->d_name[0] == '.')
               continue;

            DBG("metric set: %s\n", metric_entry->d_name);
            entry = _mesa_hash_table_search(brw->perfquery.oa_metrics_table,
                                            metric_entry->d_name);
            if (entry) {
               struct brw_perf_query *query;
               uint64_t id;

               snprintf(buf, sizeof(buf),
                        "/sys/dev/char/%d:%d/device/drm/%s/metrics/%s/id",
                        maj, min, drm_entry->d_name, metric_entry->d_name);

               if (!read_file_uint64(buf, &id)) {
                  DBG("Failed to read metric set id from %s: %m", buf);
                  continue;
               }

               query = append_query(brw);
               *query = *(struct brw_perf_query *)entry->data;
               query->oa_metrics_set_id = id;

               DBG("metric set known by kernel: id = %" PRIu64"\n",
                   query->oa_metrics_set_id);
            } else
               DBG("metric set not known by kernel (skipping)\n");
         }

         break;
      }
   }

   if (metricsdir)
      closedir(metricsdir);

close_drm_dir:
   closedir(drmdir);

   free(entry0);
   free(entry1);
}

void
brw_init_performance_queries(struct brw_context *brw)
{
   struct gl_context *ctx = &brw->ctx;
   struct stat sb;

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
      add_pipeline_statistics_query(brw, "Pipeline Statistics Registers",
                                    gen6_pipeline_statistics,
                                    (sizeof(gen6_pipeline_statistics)/
                                     sizeof(gen6_pipeline_statistics[0])));
   } else {
      add_pipeline_statistics_query(brw, "Pipeline Statistics Registers",
                                    gen7_pipeline_statistics,
                                    (sizeof(gen7_pipeline_statistics)/
                                     sizeof(gen7_pipeline_statistics[0])));
   }

   /* The existence of this sysctl parameter implies the kernel supports
    * OA metrics... */
   if (stat("/proc/sys/dev/i915/perf_stream_paranoid", &sb) == 0) {
      brw->perfquery.oa_metrics_table =
         _mesa_hash_table_create(NULL, _mesa_key_hash_string,
                                 _mesa_key_string_equal);

      init_oa_sys_vars(brw);

      /* These function add the queries applicable to the system to
       * oa_metrics_table using the query's guid as a key */
      switch (brw->gen) {
      case 7:
         if (brw->is_haswell)
            brw_oa_register_queries_hsw(brw);
         break;
      default:
         unreachable("Unexpected gen during performance queries init");
      }

      enumerate_sysfs_metrics(brw);
   }

   ctx->PerfQuery.NumQueries = brw->perfquery.n_queries;

   brw->perfquery.unresolved =
      ralloc_array(brw, struct brw_perf_query_object *, 2);
   brw->perfquery.unresolved_elements = 0;
   brw->perfquery.unresolved_array_size = 2;

   exec_list_make_empty(&brw->perfquery.sample_buffers);
   exec_list_make_empty(&brw->perfquery.free_sample_buffers);

   brw->perfquery.page_size = sysconf(_SC_PAGE_SIZE);

   brw->perfquery.oa_stream_fd = -1;

   brw->perfquery.next_query_start_report_id = 1000;
}

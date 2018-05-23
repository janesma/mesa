/*
 * Copyright © 2018 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir_instr_set.h"
#include "nir_search_helpers.h"
#include "nir_builder.h"
#include "util/u_vector.h"

bool nir_opt_comparison_pre(nir_shader *shader);

/* Partial redundancy elimination of compares
 *
 * Seaches for comparisons of the form 'a cmp b' that dominate arithmetic
 * instructions like 'b - a'.  The comparison is replaced by the arithmetic
 * instruction, and the result is compared with zero.  For example,
 *
 *       vec1 32 ssa_111 = flt 0.37, ssa_110.w
 *       if ssa_111 {
 *               block block_1:
 *              vec1 32 ssa_112 = fadd ssa_110.w, -0.37
 *              ...
 *
 * becomes
 *
 *       vec1 32 ssa_111 = fadd ssa_110.w, -0.37
 *       vec1 32 ssa_112 = flt 0.0, ssa_111
 *       if ssa_112 {
 *               block block_1:
 *              ...
 */

struct herp {
   struct exec_list blocks;
   struct exec_list reusable_blocks;
};

struct derp {
   struct exec_node node;
   struct u_vector instructions;
};

static void
herp_init(struct herp *x)
{
   exec_list_make_empty(&x->blocks);
   exec_list_make_empty(&x->reusable_blocks);
}

static void
herp_finish(struct herp *x)
{
   struct derp *n;

   while ((n = (struct derp *) exec_list_pop_head(&x->blocks)) != NULL) {
      u_vector_finish(&n->instructions);
      free(n);
   }

   while ((n = (struct derp *) exec_list_pop_head(&x->reusable_blocks)) != NULL) {
      free(n);
   }
}

static struct derp *
push_block(struct herp *x)
{
   struct derp *block_instructions =
      (struct derp *) exec_list_pop_head(&x->reusable_blocks);

   if (block_instructions == NULL) {
      block_instructions = calloc(1, sizeof(struct derp));

      if (block_instructions == NULL)
         return NULL;
   }

   if (!u_vector_init(&block_instructions->instructions,
                      sizeof(struct nir_alu_instr *),
                      8 * sizeof(struct nir_alu_instr *)))
      return NULL;

   exec_list_push_tail(&x->blocks, &block_instructions->node);

   return block_instructions;
}

static void
pop_block(struct herp *x, struct derp *block_instructions)
{
   u_vector_finish(&block_instructions->instructions);
   exec_node_remove(&block_instructions->node);
   exec_list_push_head(&x->reusable_blocks, &block_instructions->node);
}

static void
add_instruction_for_block(struct derp *block_instructions,
                          struct nir_alu_instr *alu)
{
   struct nir_alu_instr **data =
      u_vector_add(&block_instructions->instructions);

   *data = alu;
}

static void
do_it(nir_builder *build, nir_alu_instr *cmp, nir_alu_instr *alu, bool zero_on_left)
{
   void *const mem_ctx = ralloc_parent(cmp);
   static const uint8_t identity_swizzle[] = { 0, 1, 2, 3 };

   nir_alu_instr *mov = nir_alu_instr_create(mem_ctx, nir_op_imov);
   mov->dest.write_mask = cmp->dest.write_mask;
   nir_ssa_dest_init(&mov->instr, &mov->dest.dest,
                     cmp->dest.dest.ssa.num_components,
                     cmp->dest.dest.ssa.bit_size, NULL);

   nir_alu_instr *fadd = nir_alu_instr_create(mem_ctx, nir_op_fadd);
   nir_ssa_dest_init(&fadd->instr, &fadd->dest.dest,
                     alu->dest.dest.ssa.num_components,
                     alu->dest.dest.ssa.bit_size, NULL);
   fadd->dest.write_mask = alu->dest.write_mask;
   fadd->dest.saturate = alu->dest.saturate;

   nir_alu_src_copy(&fadd->src[0], &alu->src[0], fadd);
   nir_alu_src_copy(&fadd->src[1], &alu->src[1], fadd);

   build->cursor = nir_before_instr(&cmp->instr);
   nir_builder_instr_insert(build, &fadd->instr);

   nir_ssa_def *const zero =
      nir_imm_floatN_t(build, 0.0, alu->dest.dest.ssa.bit_size);

   if (zero_on_left) {
      mov->src[0].src = nir_src_for_ssa(nir_build_alu(build,
                                                      cmp->op,
                                                      zero,
                                                      &fadd->dest.dest.ssa,
                                                      NULL,
                                                      NULL));
   } else {
      mov->src[0].src = nir_src_for_ssa(nir_build_alu(build,
                                                      cmp->op,
                                                      &fadd->dest.dest.ssa,
                                                      zero,
                                                      NULL,
                                                      NULL));
   }

   memcpy(mov->src[0].swizzle, identity_swizzle, sizeof(identity_swizzle));

   nir_builder_instr_insert(build, &mov->instr);

   nir_ssa_def_rewrite_uses(&cmp->dest.dest.ssa,
                            nir_src_for_ssa(&mov->dest.dest.ssa));

   /* We know this one has no more uses because we just rewrote them all,
    * so we can remove it.  The rest of the matched expression, however, we
    * don't know so much about.  We'll just let dead code clean them up.
    */
   nir_instr_remove(&cmp->instr);
}

static bool
comparison_pre_block(nir_block *block, struct herp *x, nir_builder *build)
{
   bool progress = false;

   struct derp *block_instructions = push_block(x);

   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_alu)
         continue;

      struct nir_alu_instr *const alu = nir_instr_as_alu(instr);

      if (alu->dest.dest.ssa.num_components != 1)
         continue;

      static const uint8_t swizzle[4] = { 0, 0, 0, 0 };

      switch (alu->op) {
      case nir_op_fadd: {
         /* If the instruction is fadd, check it against comparison
          * instructions that dominate it.
          */
         struct derp *b = (struct derp *) exec_list_get_head_raw(&x->blocks);
         while (b->node.next != NULL) {
            nir_alu_instr **a;

            u_vector_foreach(a, &b->instructions) {
               nir_alu_instr *const cmp = *a;

               if (cmp == NULL)
                  continue;

               if ((nir_alu_srcs_equal(cmp, alu, 0, 0) &&
                    nir_alu_srcs_negative_equal(cmp, alu, 1, 1)) ||
                   (nir_alu_srcs_equal(cmp, alu, 0, 1) &&
                    nir_alu_srcs_negative_equal(cmp, alu, 1, 0))) {
                  /* These are the cases where (A cmp B) matches either (A +
                   * -B) or (-B + A)
                   *
                   *    A cmp B <=> A + -B cmp 0
                   */
                  do_it(build, cmp, alu, false);

                  *a = NULL;
                  progress = true;
                  break;
               } else if (nir_alu_srcs_equal(cmp, alu, 1, 0) &&
                          nir_alu_srcs_negative_equal(cmp, alu, 0, 1)) {
                  /* This is the case where (A cmp B) matches (B + -A).
                   *
                   *    A cmp B <=> 0 cmp B + -A
                   */
                  do_it(build, cmp, alu, true);

                  *a = NULL;
                  progress = true;
                  break;
               }
            }

            b = (struct derp *) b->node.next;
         }

         break;
      }

      case nir_op_flt:
      case nir_op_fge:
      case nir_op_fne:
      case nir_op_feq:
         /* If the instruction is a comparison that is used by an if-statement
          * or a bcsel and neither operand is immediate value 0, add it to the
          * set.
          */
         if (is_used_by_if(alu) &&
             is_not_const_zero(alu, 0, 1, swizzle) &&
             is_not_const_zero(alu, 1, 1, swizzle))
            add_instruction_for_block(block_instructions, alu);

         break;

      default:
         break;
      }
   }

   for (unsigned i = 0; i < block->num_dom_children; i++) {
      nir_block *child = block->dom_children[i];

      if (comparison_pre_block(child, x, build))
         progress = true;
   }

   pop_block(x, block_instructions);

   return progress;
}

static bool
nir_opt_comparison_pre_impl(nir_function_impl *impl)
{
   struct herp x;
   nir_builder build;

   herp_init(&x);
   nir_builder_init(&build, impl);

   nir_metadata_require(impl, nir_metadata_dominance);

   const bool progress = comparison_pre_block(nir_start_block(impl), &x, &build);

   herp_finish(&x);

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);

   return progress;
}

bool
nir_opt_comparison_pre(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_opt_comparison_pre_impl(function->impl);
   }

   return progress;
}


#ifndef __GDEV_NVIDIA_NVE4_H__
#define __GDEV_NVIDIA_NVE4_H__

#include "gdev_device.h"
#include "gdev_conf.h"

struct gdev_nve4_compute_desc{
	uint32_t unk0[8];
	uint32_t entry;
	uint32_t unk9[3];
	uint32_t griddim_x :31;
	uint32_t unk12 :1;
	uint16_t griddim_y;
	uint16_t griddim_z;
	uint32_t unk14[3];
	uint16_t shared_size; /* must be aligned to 0x100  */
	uint16_t unk15;
	uint16_t unk16;
	uint16_t blockdim_x;
	uint16_t blockdim_y;
	uint16_t blockdim_z;
	uint32_t cb_mask      : 8;
	uint32_t unk20_8      : 21;
	uint32_t cache_split  : 2;
	uint32_t unk20_31     : 1;
	uint32_t unk21[8];
	struct {
	    uint32_t address_l;
	    uint32_t address_h : 8;
	    uint32_t reserved  : 7;
	    uint32_t size      : 17;
	} cb[8];
	uint32_t local_size_p : 20;
	uint32_t unk45_20     : 7;
	uint32_t bar_alloc    : 5;
	uint32_t local_size_n : 20;
	uint32_t unk46_20     : 4;
	uint32_t gpr_alloc    : 8;
	uint32_t cstack_size  : 20;
	uint32_t unk47_20     : 12;
	uint32_t unk48[16];
};

/* 
 * The following macro is from Mesa 
 * https://github.com/mesa3d/mesa/blob/268dc60d3a091bc563e319c38e74cc10e544aa8a/src/gallium/drivers/nouveau/nvc0/clc3c0qmd.h
 */

#define RUN_CTA_IN_ONE_SM_PARTITION_FALSE         0x00000000
#define RUN_CTA_IN_ONE_SM_PARTITION_TRUE          0x00000001

#define IS_QUEUE_FALSE                            0x00000000
#define IS_QUEUE_TRUE                             0x00000001

#define ADD_TO_HEAD_OF_QMD_GROUP_LINKED_LIST_FALSE 0x00000000
#define ADD_TO_HEAD_OF_QMD_GROUP_LINKED_LIST_TRUE 0x00000001

#define SEMAPHORE_RELEASE_ENABLE0_FALSE           0x00000000
#define SEMAPHORE_RELEASE_ENABLE0_TRUE            0x00000001

#define SEMAPHORE_RELEASE_ENABLE1_FALSE           0x00000000
#define SEMAPHORE_RELEASE_ENABLE1_TRUE            0x00000001

#define REQUIRE_SCHEDULING_PCAS_FALSE             0x00000000
#define REQUIRE_SCHEDULING_PCAS_TRUE              0x00000001

#define DEPENDENT_QMD_SCHEDULE_ENABLE_FALSE       0x00000000
#define DEPENDENT_QMD_SCHEDULE_ENABLE_TRUE        0x00000001

#define DEPENDENT_QMD_TYPE_QUEUE                  0x00000000
#define DEPENDENT_QMD_TYPE_GRID                   0x00000001

#define DEPENDENT_QMD_FIELD_COPY_FALSE            0x00000000
#define DEPENDENT_QMD_FIELD_COPY_TRUE             0x00000001

#define INVALIDATE_TEXTURE_HEADER_CACHE_FALSE     0x00000000
#define INVALIDATE_TEXTURE_HEADER_CACHE_TRUE      0x00000001

#define INVALIDATE_TEXTURE_SAMPLER_CACHE_FALSE    0x00000000
#define INVALIDATE_TEXTURE_SAMPLER_CACHE_TRUE     0x00000001

#define INVALIDATE_TEXTURE_DATA_CACHE_FALSE       0x00000000
#define INVALIDATE_TEXTURE_DATA_CACHE_TRUE        0x00000001

#define INVALIDATE_SHADER_DATA_CACHE_FALSE        0x00000000
#define INVALIDATE_SHADER_DATA_CACHE_TRUE         0x00000001

#define INVALIDATE_INSTRUCTION_CACHE_FALSE        0x00000000
#define INVALIDATE_INSTRUCTION_CACHE_TRUE         0x00000001

#define INVALIDATE_SHADER_CONSTANT_CACHE_FALSE    0x00000000
#define INVALIDATE_SHADER_CONSTANT_CACHE_TRUE     0x00000001

#define RELEASE_MEMBAR_TYPE_FE_NONE               0x00000000
#define RELEASE_MEMBAR_TYPE_FE_SYSMEMBAR          0x00000001

#define CWD_REFERENCE_COUNT_INCR_ENABLE_FALSE     0x00000000
#define CWD_REFERENCE_COUNT_INCR_ENABLE_TRUE      0x00000001

#define CWD_MEMBAR_TYPE_L1_NONE                   0x00000000
#define CWD_MEMBAR_TYPE_L1_SYSMEMBAR              0x00000001
#define CWD_MEMBAR_TYPE_L1_MEMBAR                 0x00000003

#define SEQUENTIALLY_RUN_CTAS_FALSE               0x00000000
#define SEQUENTIALLY_RUN_CTAS_TRUE                0x00000001

#define CWD_REFERENCE_COUNT_DECR_ENABLE_FALSE     0x00000000
#define CWD_REFERENCE_COUNT_DECR_ENABLE_TRUE      0x00000001

#define API_VISIBLE_CALL_LIMIT__32                0x00000000
#define API_VISIBLE_CALL_LIMIT_NO_CHECK           0x00000001

#define SAMPLER_INDEX_INDEPENDENTLY               0x00000000
#define SAMPLER_INDEX_VIA_HEADER_INDEX            0x00000001

#define CONSTANT_BUFFER_VALID_FALSE               0x00000000
#define CONSTANT_BUFFER_VALID_TRUE                0x00000001

#define RELEASE0_REDUCTION_OP_RED_ADD             0x00000000
#define RELEASE0_REDUCTION_OP_RED_MIN             0x00000001
#define RELEASE0_REDUCTION_OP_RED_MAX             0x00000002
#define RELEASE0_REDUCTION_OP_RED_INC             0x00000003
#define RELEASE0_REDUCTION_OP_RED_DEC             0x00000004
#define RELEASE0_REDUCTION_OP_RED_AND             0x00000005
#define RELEASE0_REDUCTION_OP_RED_OR              0x00000006
#define RELEASE0_REDUCTION_OP_RED_XOR             0x00000007

#define RELEASE0_REDUCTION_FORMAT_UNSIGNED_32     0x00000000
#define RELEASE0_REDUCTION_FORMAT_SIGNED_32       0x00000001

#define RELEASE0_REDUCTION_ENABLE_FALSE           0x00000000
#define RELEASE0_REDUCTION_ENABLE_TRUE            0x00000001

#define RELEASE0_STRUCTURE_SIZE_FOUR_WORDS        0x00000000
#define RELEASE0_STRUCTURE_SIZE_ONE_WORD          0x00000001

#define RELEASE1_REDUCTION_OP_RED_ADD             0x00000000
#define RELEASE1_REDUCTION_OP_RED_MIN             0x00000001
#define RELEASE1_REDUCTION_OP_RED_MAX             0x00000002
#define RELEASE1_REDUCTION_OP_RED_INC             0x00000003
#define RELEASE1_REDUCTION_OP_RED_DEC             0x00000004
#define RELEASE1_REDUCTION_OP_RED_AND             0x00000005
#define RELEASE1_REDUCTION_OP_RED_OR              0x00000006
#define RELEASE1_REDUCTION_OP_RED_XOR             0x00000007

#define RELEASE1_REDUCTION_FORMAT_UNSIGNED_32     0x00000000
#define RELEASE1_REDUCTION_FORMAT_SIGNED_32       0x00000001

#define RELEASE1_REDUCTION_ENABLE_FALSE           0x00000000
#define RELEASE1_REDUCTION_ENABLE_TRUE            0x00000001

#define RELEASE1_STRUCTURE_SIZE_FOUR_WORDS        0x00000000
#define RELEASE1_STRUCTURE_SIZE_ONE_WORD          0x00000001

#define CONSTANT_BUFFER_INVALIDATE_FALSE          0x00000000
#define CONSTANT_BUFFER_INVALIDATE_TRUE           0x00000001

#define HW_ONLY_SPAN_LIST_HEAD_INDEX_VALID_FALSE  0x00000000
#define HW_ONLY_SPAN_LIST_HEAD_INDEX_VALID_TRUE   0x00000001

struct gdev_gv100_compute_desc {
    uint32_t outer_put;
    uint32_t outer_get :31;
    uint32_t outer_sticky_overflow :1;
    uint32_t inner_get :31;
    uint32_t inner_overflow :1;
    uint32_t inner_put :31;
    uint32_t inner_sticky_overflow :1;

    uint32_t qmd_group_id :6;
    uint32_t sm_global_caching_enable :1;
    uint32_t run_cta_in_one_sm_partition :1;
    uint32_t is_queue :1;
    uint32_t add_to_head_of_qmd_group_linked_list :1;
    uint32_t semaphore_release_enable0 :1;
    uint32_t semaphore_release_enable1 :1;
    uint32_t require_scheduling_pcas :1;
    uint32_t dependent_qmd_schedule_enable :1;
    uint32_t dependent_qmd_type :1;
    uint32_t dependent_qmd_field_copy :1;
    uint16_t qmd_reserved_b;

    uint32_t circular_queue_size :25;
    uint32_t qmd_reserved_c :1;
    uint32_t invalidate_texture_header_cache :1;
    uint32_t invalidate_texture_sampler_cache :1;
    uint32_t invalidate_texture_data_cache :1;
    uint32_t invalidate_shader_data_cache :1;
    uint32_t invalidate_instruction_cache :1;
    uint32_t invalidate_shader_constant_cache :1;

    uint32_t cta_raster_width_resume;
    uint16_t cta_raster_height_resume;
    uint16_t cta_raster_depth_resume;
    uint32_t program_offset;

    uint32_t circular_queue_addr_lower;
    uint32_t circular_queue_addr_upper :8;
    uint32_t qmd_reserved_d :8;
    uint16_t circular_queue_entry_size;

    uint32_t cwd_reference_count_id :6;
    uint32_t cwd_reference_count_delta_minus_one :8;
    uint32_t release_membar_type :1;
    uint32_t cwd_reference_count_incr_enable :1;
    uint32_t cwd_membar_type :2;
    uint32_t sequentially_run_ctas :1;
    uint32_t cwd_reference_count_decr_enable :1;
    uint32_t dummy1 :6;
    uint32_t api_visible_call_limit :1;
    uint32_t dummy2 :3;
    uint32_t sampler_index :1;
    uint32_t dummy3 :1;

    uint32_t cta_raster_width;
    uint16_t cta_raster_height;
    uint16_t qmd_reserved13a;
    uint16_t cta_raster_depth;
    uint16_t qmd_reserved14a;
    uint32_t dependent_qmd_pointer;

    uint32_t queue_entries_per_cta_minus_one :7;
    uint32_t dummy4 :3;
    uint32_t coalesce_waiting_period :8;
    uint32_t dummy5 :14;
    uint32_t shared_memory_size :18;
    uint32_t min_sm_config_shared_mem_size :7;
    uint32_t max_sm_config_shared_mem_size :7;
    uint32_t qmd_version :4;
    uint32_t qmd_major_version :4;
    uint32_t qmd_reserved_h :8;
    uint16_t cta_thread_dimension0;
    uint16_t cta_thread_dimension1;
    uint16_t cta_thread_dimension2;

    uint8_t  constant_buffer_valid;
    uint32_t register_count_v :9;
    uint32_t target_sm_config_shared_mem_size :7;
    uint8_t free_cta_slots_empty_sm;

    uint32_t sm_disable_mask_lower;
    uint32_t sm_disable_mask_upper;
    uint32_t release0_address_lower;

    uint32_t release0_address_upper :8;
    uint32_t qmd_reserved_j :8;
    uint32_t dummy6 :4;
    uint32_t release0_reduction_op :3;
    uint32_t qmd_reserved_k :1;
    uint32_t release0_reduction_format :2;
    uint32_t release0_reduction_enable :1;
    uint32_t dummy7 :4;
    uint32_t release0_structure_size :1;

    uint32_t release0_payload;
    uint32_t release1_address_lower;

    uint32_t release1_address_upper :8;
    uint32_t qmd_reserved_l :8;
    uint32_t dummy8 :4;
    uint32_t release1_reduction_op :3;
    uint32_t qmd_reserved_m :1;
    uint32_t release1_reduction_format :2;
    uint32_t release1_reduction_enable :1;
    uint32_t dummy9 :4;
    uint32_t release1_structure_size :1;

    uint32_t release1_payload;

    uint32_t shader_local_memory_low_size :24;
    uint32_t qmd_reserved_n :3;
    uint32_t barrier_count :5;
    uint32_t shader_local_memory_high_size :24;
    uint32_t register_count :8;
    uint32_t shader_local_memory_crs_size :24;
    uint32_t sass_version :8;

    struct {
        uint32_t address_l;
        uint32_t address_h : 17;
        uint32_t reserved  : 1;
        uint32_t invalidate: 1;
        uint32_t size_shifted :13;
    } cb[8];

    uint32_t program_address_lower;
    uint32_t program_address_upper :17;
    uint32_t qmd_reserved_s :15;

    uint32_t hw_only_inner_get :31;
    uint32_t hw_only_require_scheduling_pcas :1;

    uint32_t hw_only_inner_put :31;
    uint32_t hw_only_scg_type :1;

    uint32_t hw_only_span_list_head_index :30;
    uint32_t qmd_reserved_q :1;
    uint32_t hw_only_span_list_head_index_valid :1;

    uint32_t hw_only_sked_next_qmd_pointer;
    uint32_t qmd_spare_g;
    uint32_t qmd_spare_h;
    uint32_t qmd_spare_i;
    uint32_t qmd_spare_j;
    uint32_t qmd_spare_k;
    uint32_t qmd_spare_l;
    uint32_t qmd_spare_m;
    uint32_t qmd_spare_n;
    uint32_t debug_id_upper;
    uint32_t debug_id_lower;
};

struct gdev_tu102_compute_desc {
    uint32_t outer_put;
    uint32_t outer_get :31;
    uint32_t outer_sticky_overflow :1;
    uint32_t inner_get :31;
    uint32_t inner_overflow :1;
    uint32_t inner_put :31;
    uint32_t inner_sticky_overflow :1;

    uint32_t qmd_group_id :6;
    uint32_t sm_global_caching_enable :1;
    uint32_t run_cta_in_one_sm_partition :1;
    uint32_t is_queue :1;
    uint32_t add_to_head_of_qmd_group_linked_list :1;
    uint32_t semaphore_release_enable0 :1;
    uint32_t semaphore_release_enable1 :1;
    uint32_t require_scheduling_pcas :1;
    uint32_t dependent_qmd_schedule_enable :1;
    uint32_t dependent_qmd_type :1;
    uint32_t dependent_qmd_field_copy :1;
    uint16_t qmd_reserved_b;

    uint32_t circular_queue_size :25;
    uint32_t qmd_reserved_c :1;
    uint32_t invalidate_texture_header_cache :1;
    uint32_t invalidate_texture_sampler_cache :1;
    uint32_t invalidate_texture_data_cache :1;
    uint32_t invalidate_shader_data_cache :1;
    uint32_t invalidate_instruction_cache :1;
    uint32_t invalidate_shader_constant_cache :1;

    uint32_t cta_raster_width_resume;
    uint16_t cta_raster_height_resume;
    uint16_t cta_raster_depth_resume;
    uint32_t program_prefetch_addr_lower_shifted;

    uint32_t circular_queue_addr_lower;
    uint32_t circular_queue_addr_upper :8;
    uint32_t qmd_reserved_d :8;
    uint16_t circular_queue_entry_size;

    uint32_t cwd_reference_count_id :6;
    uint32_t cwd_reference_count_delta_minus_one :8;
    uint32_t release_membar_type :1;
    uint32_t cwd_reference_count_incr_enable :1;
    uint32_t cwd_membar_type :2;
    uint32_t sequentially_run_ctas :1;
    uint32_t cwd_reference_count_decr_enable :1;
    uint32_t dummy1 :6;
    uint32_t api_visible_call_limit :1;
    uint32_t dummy2 :3;
    uint32_t sampler_index :1;
    uint32_t dummy3 :1;

    uint32_t cta_raster_width;
    uint16_t cta_raster_height;
    uint16_t qmd_reserved13a;
    uint16_t cta_raster_depth;
    uint16_t qmd_reserved14a;
    uint32_t dependent_qmd_pointer;

    uint32_t dummy4 :10;
    uint32_t coalesce_waiting_period :8;
    uint32_t queue_entries_per_cta_log2 :5;
    uint32_t dummy5 :9;
    uint32_t shared_memory_size :18;
    uint32_t min_sm_config_shared_mem_size :7;
    uint32_t max_sm_config_shared_mem_size :7;
    uint32_t qmd_version :4;
    uint32_t qmd_major_version :4;
    uint32_t qmd_reserved_h :8;
    uint16_t cta_thread_dimension0;
    uint16_t cta_thread_dimension1;
    uint16_t cta_thread_dimension2;

    uint8_t  constant_buffer_valid;
    uint32_t register_count_v :9;
    uint32_t target_sm_config_shared_mem_size :7;
    uint8_t free_cta_slots_empty_sm;

    uint32_t sm_disable_mask_lower;
    uint32_t sm_disable_mask_upper;
    uint32_t release0_address_lower;

    uint32_t release0_address_upper :8;
    uint32_t qmd_reserved_j :8;
    uint32_t dummy6 :4;
    uint32_t release0_reduction_op :3;
    uint32_t qmd_reserved_k :1;
    uint32_t release0_reduction_format :2;
    uint32_t release0_reduction_enable :1;
    uint32_t dummy7 :4;
    uint32_t release0_structure_size :1;

    uint32_t release0_payload;
    uint32_t release1_address_lower;

    uint32_t release1_address_upper :8;
    uint32_t qmd_reserved_l :8;
    uint32_t dummy8 :4;
    uint32_t release1_reduction_op :3;
    uint32_t qmd_reserved_m :1;
    uint32_t release1_reduction_format :2;
    uint32_t release1_reduction_enable :1;
    uint32_t dummy9 :4;
    uint32_t release1_structure_size :1;

    uint32_t release1_payload;

    uint32_t shader_local_memory_low_size :24;
    uint32_t qmd_reserved_n :3;
    uint32_t barrier_count :5;
    uint32_t shader_local_memory_high_size :24;
    uint32_t register_count :8;
    uint32_t program_prefetch_addr_upper_shifted :9;
    uint32_t program_prefetch_size :9;
    uint32_t qmd_reserved_a :6;
    uint32_t sass_version :8;

    struct {
        uint32_t address_l;
        uint32_t address_h : 17;
        uint32_t reserved  : 1;
        uint32_t invalidate: 1;
        uint32_t size_shifted :13;
    } cb[8];

    uint32_t program_address_lower;
    uint32_t program_address_upper :17;
    uint32_t qmd_reserved_s :15;

    uint32_t hw_only_inner_get :31;
    uint32_t hw_only_require_scheduling_pcas :1;

    uint32_t hw_only_inner_put :31;
    uint32_t hw_only_scg_type :1;

    uint32_t hw_only_span_list_head_index :30;
    uint32_t qmd_reserved_q :1;
    uint32_t hw_only_span_list_head_index_valid :1;

    uint32_t hw_only_sked_next_qmd_pointer;
    uint32_t qmd_spare_g;
    uint32_t qmd_spare_h;
    uint32_t qmd_spare_i;
    uint32_t qmd_spare_j;
    uint32_t qmd_spare_k;
    uint32_t qmd_spare_l;
    uint32_t qmd_spare_m;
    uint32_t qmd_spare_n;
    uint32_t debug_id_upper;
    uint32_t debug_id_lower;
};

#define GDEV_SUBCH_NV_P2MF GDEV_SUBCH_NV_M2MF

struct gdev_nve4_compute_desc *gdev_nve4_compute_desc_set(struct gdev_ctx *ctx, struct gdev_nve4_compute_desc *desc, struct gdev_kernel *k);
#endif

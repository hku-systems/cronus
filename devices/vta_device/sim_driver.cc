/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file sim_driver.cc
 * \brief VTA driver for simulated backend.
 */
#include <vta/driver.h>
#include <vta/hw_spec.h>
#include <vta/sim_tlpp.h>
#include <vta/vta_log.h>
#include <type_traits>
#include <mutex>
#include <map>
#include <unordered_map>
#include <cstring>
#include <sstream>
#include <atomic>
#include <immintrin.h>

#include "vmem/virtual_memory.h"

#include "stmd_spin.hpp"

#define MAX_CONCURRENCY 8

namespace vta {
namespace sim {

STMD tpool = STMD(8);

typedef std::int_fast32_t i32;
typedef std::uint_fast32_t u32;
typedef std::int_fast64_t i64;
typedef std::uint_fast64_t ui64;

/*! \brief debug flag for skipping computation */
enum DebugFlagMask {
  kSkipExec = 1
};

/*!
 * \brief Helper class to pack and unpack bits
 *  Applies truncation when pack to low level bits.
 *
 * \tparam bits The number of bits in integer.
 * \note This implementation relies on little endian.
 */
template<uint32_t bits>
class BitPacker {
 public:
  explicit BitPacker(void* data) {
    data_ = static_cast<uint32_t*>(data);
  }

  uint32_t GetUnsigned(uint32_t index) const {
    if (bits == 32) {
      return data_[index];
    } else if (bits == 16) {
      return reinterpret_cast<uint16_t*>(data_)[index];
    } else if (bits == 8) {
      return reinterpret_cast<uint8_t*>(data_)[index];
    } else {
      uint32_t offset = index / kNumPackElem;
      uint32_t shift = index % kNumPackElem;
      return (data_[offset] >> shift) & kMask;
    }
  }

  int32_t GetSigned(uint32_t index) const {
    if (bits == 32) {
      return reinterpret_cast<int32_t*>(data_)[index];
    } else if (bits == 16) {
      return reinterpret_cast<int16_t*>(data_)[index];
    } else if (bits == 8) {
      return reinterpret_cast<int8_t*>(data_)[index];
    } else {
      uint32_t offset = index / kNumPackElem;
      uint32_t shift = (index % kNumPackElem) * bits;
      int32_t uvalue = static_cast<int32_t>(
          (data_[offset] >> shift) & kMask);
      int kleft = 32 - bits;
      return (uvalue << kleft) >> kleft;
    }
  }

  void SetUnsigned(uint32_t index, uint32_t value) {
    if (bits == 32) {
      data_[index] = value;
    } else if (bits == 16) {
      reinterpret_cast<uint16_t*>(data_)[index] = value;
    } else if (bits == 8) {
      reinterpret_cast<uint8_t*>(data_)[index] = value;
    } else {
      uint32_t offset = index / kNumPackElem;
      uint32_t shift = (index % kNumPackElem) * bits;
      data_[offset] &= (~(kMask << shift));
      data_[offset] |= (value & kMask) << shift;
    }
  }

  inline void SetSigned(uint32_t index, int32_t value) {
    if (bits == 32) {
      reinterpret_cast<int32_t*>(data_)[index] = value;
    } else if (bits == 16) {
      reinterpret_cast<int16_t*>(data_)[index] = value;
    } else if (bits == 8) {
      reinterpret_cast<int8_t*>(data_)[index] = value;
    } else {
      uint32_t offset = index / kNumPackElem;
      uint32_t shift = (index % kNumPackElem) * bits;
      data_[offset] &= (~(kMask << shift));
      data_[offset] |= static_cast<uint32_t>(value & kMask) << shift;
    }
  }

 private:
  uint32_t* data_;
  static constexpr uint32_t kNumPackElem = 32 / bits;
  static constexpr uint32_t kMask = (1U << (bits >= 32U ? 31U : bits)) - 1U;
};

/*!
 * \brief DRAM memory manager
 *  Implements simple paging to allow physical address translation.
 */
using DRAM = ::vta::vmem::VirtualMemoryManager;

/*!
 * \brief Register file.
 * \tparam kBits Number of bits of one value.
 * \tparam kLane Number of lanes in one element.
 * \tparam kMaxNumElem Maximum number of element.
 */
template<int kBits, int kLane, int kMaxNumElem>
class SRAM {
 public:
  /*! \brief Bytes of single vector element */
  static const int kElemBytes = (kBits * kLane + 7) / 8;
  /*! \brief content data type */
  using DType = typename std::aligned_storage<kElemBytes, kElemBytes>::type;
  SRAM() {
    data_ = new DType[kMaxNumElem];
  }
  ~SRAM() {
    delete [] data_;
  }
  // Get the i-th index
  void* BeginPtr(uint32_t index) {
    CHECK_LT(index, kMaxNumElem);
    return &(data_[index]);
  }
  // Execute the load instruction on this SRAM
  void Load(const VTAMemInsn* op,
            DRAM* dram,
            vta_phy_addr_t offset,
            uint64_t* load_counter,
            bool skip_exec) {
    load_counter[0] += (op->x_size * op->y_size) * kElemBytes;
    if (skip_exec) return;
    DType* sram_ptr = data_ + op->sram_base;
    uint8_t* dram_ptr = static_cast<uint8_t*>(dram->GetAddr(
        op->dram_base * kElemBytes + offset));
    uint64_t xtotal = op->x_size + op->x_pad_0 + op->x_pad_1;
    uint32_t ytotal = op->y_size + op->y_pad_0 + op->y_pad_1;
    uint64_t sram_end = op->sram_base + xtotal * ytotal;
    CHECK_LE(sram_end, kMaxNumElem);
    memset(sram_ptr, 0, kElemBytes * xtotal * op->y_pad_0);
    sram_ptr += xtotal * op->y_pad_0;

    tpool.parallelize_loop(0, op->y_size, [&sram_ptr, &dram_ptr, &op](const u32 y){
      // for (uint32_t y = start; y < end; ++y) {
        DType* _sram_ptr = sram_ptr + y * (op->x_pad_0 + op->x_size + op->x_pad_1);
        uint8_t* _dram_ptr = dram_ptr + y * (kElemBytes * op->x_stride);      
          
        memset(_sram_ptr, 0, kElemBytes * op->x_pad_0);
        _sram_ptr += op->x_pad_0;
        memcpy(_sram_ptr, _dram_ptr, kElemBytes * op->x_size);
        _sram_ptr += op->x_size;
        memset(_sram_ptr, 0, kElemBytes * op->x_pad_1);
        _sram_ptr += op->x_pad_1;
        _dram_ptr += kElemBytes * op->x_stride;
      // }
    }, 8);
    sram_ptr += (op->x_pad_0 + op->x_size + op->x_pad_1) * op->y_size;
    dram_ptr += (kElemBytes * op->x_stride) * op->y_size;

    memset(sram_ptr, 0, kElemBytes * xtotal * op->y_pad_1);
  }

  // This is for load 8bits to ACC only
  void Load_int8(const VTAMemInsn* op,
            DRAM* dram,
            vta_phy_addr_t offset,
            uint64_t* load_counter,
            bool skip_exec) {
    CHECK_EQ(kBits, VTA_ACC_WIDTH);

    // TODO(zhanghao): extend to other width
    CHECK_EQ(VTA_ACC_WIDTH, 32);
    CHECK_EQ(VTA_INP_WIDTH, 8);

    int factor = VTA_ACC_WIDTH / VTA_INP_WIDTH;
    load_counter[0] += (op->x_size * op->y_size) * kElemBytes;
    if (skip_exec) return;
    DType* sram_ptr = data_ + op->sram_base;
    int8_t* dram_ptr = static_cast<int8_t*>(dram->GetAddr(
        op->dram_base * kElemBytes / factor + offset));
    uint64_t xtotal = op->x_size + op->x_pad_0 + op->x_pad_1;
    uint32_t ytotal = op->y_size + op->y_pad_0 + op->y_pad_1;
    uint64_t sram_end = op->sram_base + xtotal * ytotal;
    CHECK_LE(sram_end, kMaxNumElem);
    memset(sram_ptr, 0, kElemBytes * xtotal * op->y_pad_0);
    sram_ptr += xtotal * op->y_pad_0;

    tpool.parallelize_loop(0, op->y_size, [&sram_ptr, &dram_ptr, factor, &op](u32 y){
      // for (uint32_t y = start; y < end; ++y) {
        DType* _sram_ptr = sram_ptr + y * (op->x_pad_0 + op->x_size + op->x_pad_1);
        int8_t* _dram_ptr = dram_ptr + y * (kElemBytes / factor * op->x_stride);

        memset(_sram_ptr, 0, kElemBytes * op->x_pad_0);
        _sram_ptr += op->x_pad_0;

        int32_t* sram_ele_ptr = (int32_t*)_sram_ptr;
        // __m512i mm_sram;
        for (uint32_t x = 0; x < op->x_size * VTA_BATCH * VTA_BLOCK_OUT / 16; ++x) {
          *(sram_ele_ptr + x) = (int32_t)*(_dram_ptr + x);
          // mm_sram = _mm512_load_si512(_dram_ptr + x * 16);
          // _mm512_store_si512(sram_ele_ptr + x * 16, mm_sram);
        }
        _sram_ptr += op->x_size;

        memset(_sram_ptr, 0, kElemBytes * op->x_pad_1);
        _sram_ptr += op->x_pad_1;

        // dram one element is 1 bytes rather than 4 bytes
        _dram_ptr += kElemBytes / factor * op->x_stride;
      // }
    }, 8);
    sram_ptr += (op->x_pad_0 + op->x_size + op->x_pad_1) * op->y_size;
    dram_ptr += (kElemBytes / factor * op->x_stride) * op->y_size;

    memset(sram_ptr, 0, kElemBytes * xtotal * op->y_pad_1);
  }


  // Execute the store instruction on this SRAM apply trucation.
  // This relies on the elements is 32 bits
  template<int target_bits>
  void TruncStore(const VTAMemInsn* op, DRAM* dram, vta_phy_addr_t offset) {
    CHECK_EQ(op->x_pad_0, 0);
    CHECK_EQ(op->x_pad_1, 0);
    CHECK_EQ(op->y_pad_0, 0);
    CHECK_EQ(op->y_pad_1, 0);
    int target_width = (target_bits * kLane + 7) / 8;
    BitPacker<kBits> src(data_ + op->sram_base);
    BitPacker<target_bits> dst(dram->GetAddr(op->dram_base * target_width + offset));
    tpool.parallelize_loop(0, op->y_size, [&op, &src, &dst, offset](u32 y){
      // for (uint32_t y = 0; y < op->y_size; ++y) {
        for (uint32_t x = 0; x < op->x_size; ++x) {
          uint32_t sram_base = y * op->x_size + x;
          uint32_t dram_base = y * op->x_stride + x;
          for (int i = 0; i < kLane; ++i) {
            dst.SetSigned(dram_base * kLane + i,
                          src.GetSigned(sram_base * kLane +i));
          }
        }
      // }
    });
  }

 private:
  /*! \brief internal data content */
  DType* data_;
};


/*!
 * \brief Memory information of special memory region.
 *  Use MemoryInfo as its container type
 */
class Profiler {
 public:
  /*! \brief The memory load statistics */
  uint64_t inp_load_nbytes{0};
  /*! \brief The memory load statistics */
  uint64_t wgt_load_nbytes{0};
  /*! \brief The ACC memory load statistics */
  uint64_t acc_load_nbytes{0};
  /*! \brief The ACC memory load statistics */
  uint64_t uop_load_nbytes{0};
  /*! \brief The ACC memory load statistics */
  uint64_t out_store_nbytes{0};
  /*! \brief instr counter for gemm */
  uint64_t gemm_counter{0};
  /*! \brief instr counter for ALU ops */
  uint64_t alu_counter{0};
  /*! \brief set debug mode */
  int64_t debug_flag{0};
  /*! \brief clear the profiler */
  void Clear() {
    inp_load_nbytes = 0;
    wgt_load_nbytes = 0;
    acc_load_nbytes = 0;
    uop_load_nbytes = 0;
    out_store_nbytes = 0;
    gemm_counter = 0;
    alu_counter = 0;
  }
  /*! \return Whether we should skip execution. */
  bool SkipExec() const {
    return (debug_flag & DebugFlagMask::kSkipExec) != 0;
  }

  std::string AsJSON() {
    std::ostringstream os;
    os << "{\n"
       << " \"inp_load_nbytes\":" << inp_load_nbytes << ",\n"
       << " \"wgt_load_nbytes\":" << wgt_load_nbytes << ",\n"
       << " \"acc_load_nbytes\":" << acc_load_nbytes << ",\n"
       << " \"uop_load_nbytes\":" << uop_load_nbytes << ",\n"
       << " \"out_store_nbytes\":" << out_store_nbytes << ",\n"
       << " \"gemm_counter\":" << gemm_counter << ",\n"
       << " \"alu_counter\":" << alu_counter << "\n"
       <<"}\n";
    return os.str();
  }

  static Profiler* ThreadLocal() {
    static thread_local Profiler inst;
    return &inst;
  }
};


// Simulate device
// TODO(tqchen,thierry): queue based event driven simulation.
class Device {
 public:
  Device() {
    prof_ = Profiler::ThreadLocal();
    dram_ = DRAM::Global();
  }

  int Run(vta_phy_addr_t insn_phy_addr,
          vta_phy_addr_t phy_addr_offset,
          uint32_t insn_count,
          uint32_t wait_cycles) {
    int idx = -1;
    for (int i = 0;i < MAX_CONCURRENCY;i++) {
      bool expected = false;
      if (std::atomic_compare_exchange_strong(&sram_used[i], &expected, true))
        idx = i;
    }
    if (idx == -1) {
      // no avaliable sram to use
      return 1;
    }
    VTAGenericInsn* insn = static_cast<VTAGenericInsn*>(
        dram_->GetAddr(insn_phy_addr + phy_addr_offset));
    finish_counter_ = 0;
    for (uint32_t i = 0; i < insn_count; ++i) {
      // this->Run(insn + i);
      this->Run_Insn(insn + i, this, phy_addr_offset, idx);
    }
    // this->TlppSynchronization();
    sram_used[idx] = false;
    return 0;
  }

 private:
  static void Run_Insn(const VTAGenericInsn* insn, void * dev, vta_phy_addr_t offset, int idx) {
    Device * device = reinterpret_cast<Device *> (dev);
    const VTAMemInsn* mem = reinterpret_cast<const VTAMemInsn*>(insn);
    const VTAGemInsn* gem = reinterpret_cast<const VTAGemInsn*>(insn);
    const VTAAluInsn* alu = reinterpret_cast<const VTAAluInsn*>(insn);
    switch (mem->opcode) {
      case VTA_OPCODE_LOAD: device->RunLoad(mem, offset, idx); break;
      case VTA_OPCODE_STORE: device->RunStore(mem, offset, idx); break;
      case VTA_OPCODE_GEMM: device->RunGEMM(gem, idx); break;
      case VTA_OPCODE_ALU: device->RunALU(alu, idx); break;
      case VTA_OPCODE_FINISH: ++(device->finish_counter_); break;
      default: {
        LOG(FATAL) << "Unknown op_code" << mem->opcode;
      }
    }
  }

 private:

  void RunLoad(const VTAMemInsn* op, vta_phy_addr_t offset, int idx) {
    if (op->x_size == 0) return;
    if (op->memory_type == VTA_MEM_ID_INP) {
      inp_[idx].Load(op, dram_, offset, &(prof_->inp_load_nbytes), prof_->SkipExec());
    } else if (op->memory_type == VTA_MEM_ID_WGT) {
      wgt_[idx].Load(op, dram_, offset, &(prof_->wgt_load_nbytes), prof_->SkipExec());
    } else if (op->memory_type == VTA_MEM_ID_ACC) {
      acc_[idx].Load(op, dram_, offset, &(prof_->acc_load_nbytes), prof_->SkipExec());
    } else if (op->memory_type == VTA_MEM_ID_UOP) {
      // always load in uop, since uop is stateful
      // subsequent non-debug mode exec can depend on it.
      uop_[idx].Load(op, dram_, offset, &(prof_->uop_load_nbytes), false);
    } else if (op->memory_type == VTA_MEM_ID_ACC_8BIT) {
      acc_[idx].Load_int8(op, dram_, offset, &(prof_->acc_load_nbytes), prof_->SkipExec());
    } else {
      LOG(FATAL) << "Unknown memory_type=" << op->memory_type;
    }
  }

  void RunStore(const VTAMemInsn* op, vta_phy_addr_t offset, int idx) {
    if (op->x_size == 0) return;
    if (op->memory_type == VTA_MEM_ID_OUT) {
      prof_->out_store_nbytes += (
          op->x_size * op->y_size * VTA_BATCH * VTA_BLOCK_OUT * VTA_OUT_WIDTH / 8);
      if (!prof_->SkipExec()) {
        acc_[idx].TruncStore<VTA_OUT_WIDTH>(op, dram_, offset);
      }
    } else {
      LOG(FATAL) << "Store do not support memory_type="
                 << op->memory_type;
    }
  }

  void RunGEMM(const VTAGemInsn* op, int idx) {
    if (!op->reset_reg) {
      prof_->gemm_counter += op->iter_out * op->iter_in * (op->uop_end - op->uop_bgn);
      if (prof_->SkipExec()) return;
      if (op->iter_out > op->iter_in) {
        tpool.parallelize_loop(0, op->iter_out, [&op, idx, this](u32 y){
        // for (uint32_t y = start; y < end; ++y) {
          for (uint32_t x = 0; x < op->iter_in; ++x) {
            for (uint32_t uindex = op->uop_bgn; uindex < op->uop_end; ++uindex) {
              VTAUop* uop_ptr = static_cast<VTAUop*>(uop_[idx].BeginPtr(uindex));
              // Read in memory indices
              uint32_t acc_idx = uop_ptr->dst_idx;
              uint32_t inp_idx = uop_ptr->src_idx;
              uint32_t wgt_idx = uop_ptr->wgt_idx;

              acc_idx += y * op->dst_factor_out + x * op->dst_factor_in;
              inp_idx += y * op->src_factor_out + x * op->src_factor_in;
              wgt_idx += y * op->wgt_factor_out + x * op->wgt_factor_in;
              BitPacker<VTA_ACC_WIDTH> acc(acc_[idx].BeginPtr(acc_idx));
              BitPacker<VTA_INP_WIDTH> inp(inp_[idx].BeginPtr(inp_idx));
              BitPacker<VTA_WGT_WIDTH> wgt(wgt_[idx].BeginPtr(wgt_idx));

              // gemm loop
              for (uint32_t i = 0; i < VTA_BATCH; ++i) {
                for (uint32_t j = 0; j < VTA_BLOCK_OUT; ++j) {
                  uint32_t acc_offset = i * VTA_BLOCK_OUT + j;
                  int32_t sum = acc.GetSigned(acc_offset);
                  for (uint32_t k = 0; k < VTA_BLOCK_IN; ++k) {
                    sum +=
                        inp.GetSigned(i * VTA_BLOCK_IN + k) *
                        wgt.GetSigned(j * VTA_BLOCK_IN + k);
                  }
                  acc.SetSigned(acc_offset, sum);
                }
              }
            }
          }
        // }
      });
      } else {
        tpool.parallelize_loop(0, op->iter_in, [&op, idx, this](u32 x){
        for (uint32_t y = 0; y < op->iter_out; ++y) {
          // for (uint32_t x = 0; x < op->iter_in; ++x) {
            for (uint32_t uindex = op->uop_bgn; uindex < op->uop_end; ++uindex) {
              VTAUop* uop_ptr = static_cast<VTAUop*>(uop_[idx].BeginPtr(uindex));
              // Read in memory indices
              uint32_t acc_idx = uop_ptr->dst_idx;
              uint32_t inp_idx = uop_ptr->src_idx;
              uint32_t wgt_idx = uop_ptr->wgt_idx;

              acc_idx += y * op->dst_factor_out + x * op->dst_factor_in;
              inp_idx += y * op->src_factor_out + x * op->src_factor_in;
              wgt_idx += y * op->wgt_factor_out + x * op->wgt_factor_in;
              BitPacker<VTA_ACC_WIDTH> acc(acc_[idx].BeginPtr(acc_idx));
              BitPacker<VTA_INP_WIDTH> inp(inp_[idx].BeginPtr(inp_idx));
              BitPacker<VTA_WGT_WIDTH> wgt(wgt_[idx].BeginPtr(wgt_idx));

              // gemm loop
              for (uint32_t i = 0; i < VTA_BATCH; ++i) {
                for (uint32_t j = 0; j < VTA_BLOCK_OUT; ++j) {
                  uint32_t acc_offset = i * VTA_BLOCK_OUT + j;
                  int32_t sum = acc.GetSigned(acc_offset);
                  for (uint32_t k = 0; k < VTA_BLOCK_IN; ++k) {
                    sum +=
                        inp.GetSigned(i * VTA_BLOCK_IN + k) *
                        wgt.GetSigned(j * VTA_BLOCK_IN + k);
                  }
                  acc.SetSigned(acc_offset, sum);
                }
              }
            }
          // }
        }
      });
      }
    } else {
      if (prof_->SkipExec()) return;
      // reset
      tpool.parallelize_loop(0, op->iter_out, [&op, idx, this](u32 y){
        // for (uint32_t y = start; y < end; ++y) {
          for (uint32_t x = 0; x < op->iter_in; ++x) {
            for (uint32_t uindex = op->uop_bgn; uindex < op->uop_end; ++uindex) {
              VTAUop* uop_ptr = static_cast<VTAUop*>(uop_[idx].BeginPtr(uindex));
              uint32_t acc_idx = uop_ptr->dst_idx;
              acc_idx += y * op->dst_factor_out + x * op->dst_factor_in;
              BitPacker<VTA_ACC_WIDTH> acc(acc_[idx].BeginPtr(acc_idx));
              for (uint32_t i = 0; i < VTA_BATCH * VTA_BLOCK_OUT; ++i) {
                acc.SetSigned(i, 0);
              }
            }
          }
        // }
      }); 
    }
  }

  void RunALU(const VTAAluInsn* op, int idx) {
    if (op->use_imm) {
      RunALU_<true>(op, idx);
    } else {
      RunALU_<false>(op, idx);
    }
  }

  template<bool use_imm>
  void RunALU_(const VTAAluInsn* op, int idx) {
    switch (op->alu_opcode) {
      case VTA_ALU_OPCODE_ADD: {
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            return x + y;
          }, idx);
      }
      case VTA_ALU_OPCODE_MAX: {
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            return std::max(x, y);
          }, idx);
      }
      case VTA_ALU_OPCODE_MIN: {
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            return std::min(x, y);
          }, idx);
      }
      case VTA_ALU_OPCODE_SHR: {
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            if (y >= 0) {
              return x >> y;
            } else {
              return x << (-y);
            }
          }, idx);
      }
      case VTA_ALU_OPCODE_MUL: {
        return RunALULoop<use_imm>(op, [](int32_t x, int32_t y) {
            return x * y;
          }, idx);
      }
      default: {
        LOG(FATAL) << "Unknown ALU code " << op->alu_opcode;
      }
    }
  }

  template<bool use_imm, typename F>
  void RunALULoop(const VTAAluInsn* op, F func, int idx) {
    prof_->alu_counter += op->iter_out * op->iter_in * (op->uop_end - op->uop_bgn);
    if (prof_->SkipExec()) return;
    tpool.parallelize_loop(0, op->iter_in, [&op, &func, idx, this](u32 x){
      for (int y = 0; y < op->iter_out; ++y) {
        // for (int x = start; x < end; ++x) {
          for (int k = op->uop_bgn; k < op->uop_end; ++k) {
            // Read micro op
            VTAUop* uop_ptr = static_cast<VTAUop*>(uop_[idx].BeginPtr(k));
            uint32_t dst_index = uop_ptr->dst_idx;
            uint32_t src_index = uop_ptr->src_idx;
            dst_index += y * op->dst_factor_out + x * op->dst_factor_in;
            src_index += y * op->src_factor_out + x * op->src_factor_in;
            BitPacker<VTA_ACC_WIDTH> dst(acc_[idx].BeginPtr(dst_index));
            BitPacker<VTA_ACC_WIDTH> src(acc_[idx].BeginPtr(src_index));
            for (int k = 0; k < VTA_BATCH * VTA_BLOCK_OUT; ++k) {
              if (use_imm) {
                dst.SetSigned(k, func(dst.GetSigned(k), op->imm));
              } else {
                dst.SetSigned(k, func(dst.GetSigned(k), src.GetSigned(k)));
              }
            }
          }
        // }
      }
    });    
  }
  // the finish counter
  int finish_counter_{0};
  // Prof_
  Profiler* prof_;
  // The DRAM interface
  DRAM* dram_;
  // The SRAM
  SRAM<VTA_INP_WIDTH, VTA_BATCH * VTA_BLOCK_IN, VTA_INP_BUFF_DEPTH> inp_[MAX_CONCURRENCY];
  SRAM<VTA_WGT_WIDTH, VTA_BLOCK_IN * VTA_BLOCK_OUT, VTA_WGT_BUFF_DEPTH> wgt_[MAX_CONCURRENCY];
  SRAM<VTA_ACC_WIDTH, VTA_BATCH * VTA_BLOCK_OUT, VTA_ACC_BUFF_DEPTH> acc_[MAX_CONCURRENCY];
  SRAM<VTA_UOP_WIDTH, 1, VTA_UOP_BUFF_DEPTH> uop_[MAX_CONCURRENCY];
  std::atomic<bool> sram_used[MAX_CONCURRENCY];
};

}  // namespace sim
}  // namespace vta

void* VTAMemAlloc(size_t size, int cached) {
  return vta::sim::DRAM::Global()->Alloc(size);
}

void VTAMemFree(void* buf) {
  vta::sim::DRAM::Global()->Free(buf);
}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) {
  return vta::sim::DRAM::Global()->GetPhyAddr(buf);
}

void VTAMemCopyFromHost(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void VTAMemCopyToHost(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void VTAFlushCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
}

void VTAInvalidateCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
}

VTADeviceHandle VTADeviceAlloc() {
  return new vta::sim::Device();
}

void VTADeviceFree(VTADeviceHandle handle) {
  delete static_cast<vta::sim::Device*>(handle);
}

int VTADeviceRun(VTADeviceHandle handle,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  return static_cast<vta::sim::Device*>(handle)->Run(
      insn_phy_addr, 0, insn_count, wait_cycles);
}

void VTAProgram(const char* bitstream) {
}

VTADeviceHandle global_handle = nullptr;

int VTADeviceExec(vta_phy_addr_t insn_phy_addr,
                 vta_phy_addr_t offset,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  if (global_handle == nullptr) {
    global_handle = VTADeviceAlloc();
  }
  fprintf(stderr, "executing %d\n", offset);
  return static_cast<vta::sim::Device*>(global_handle)->Run(
      insn_phy_addr, offset, insn_count, wait_cycles);
}

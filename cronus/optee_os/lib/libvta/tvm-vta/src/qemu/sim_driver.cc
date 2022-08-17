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
// #include <tvm/runtime/registry.h>
#include <vta/sim_tlpp.h>
#include <type_traits>
#include <mutex>
#include <map>
#include <unordered_map>
#include <cstring>
#include <sstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#define BUDDY_ALLOC_IMPLEMENTATION
#include "buddy.h"

namespace vta {
/*
namespace sim {

using tvm::runtime::TVMRetValue;
using tvm::runtime::TVMArgs;

TVM_REGISTER_GLOBAL("vta.simulator.profiler_clear")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    
  });
TVM_REGISTER_GLOBAL("vta.simulator.profiler_status")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = std::string("{}");
  });
TVM_REGISTER_GLOBAL("vta.simulator.profiler_debug_mode")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    
  });

}  // namespace sim
*/

namespace qemu {

  #define DEV_NAME "/dev/tvm-vta-0"
  #define TOTAL_PAGES 32768

  enum {
    IOCTL_CMD_EXEC = 1
  };

  typedef struct {
    uint32_t insn_phy_addr;
    uint32_t insn_count;
    uint32_t wait_cycles;
    uint32_t status;
  } vta_exec_t;

  class Device {
  private:
    int fd = 0;
    char* memory = NULL;
    int cur_idx = 0;

    char *buddy_metadata;
    char *buddy_arena;
    struct buddy *buddy;

    void init(const char* dev_name, int total_pages) {
      struct stat st_dev;

      fd = open(dev_name, O_RDWR);
      if (fd < 0){
          fprintf(stderr,"open: %s failed\n", dev_name);
          return;
      }

      memory = (char*)mmap (NULL, total_pages * 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

      if (memory == MAP_FAILED) {
        fprintf(stderr, "error in mmap\n");
        close(fd);
        fd = 0;
        return;
      }
      uint64_t *ff = (uint64_t*)memory;
      size_t arena_size = total_pages * 4096;
      buddy_metadata = new char[buddy_sizeof(arena_size)];
      buddy_arena = memory;
      buddy = buddy_init((unsigned char*)buddy_metadata, (unsigned char*)buddy_arena, arena_size);
    }

  public:
    Device() {
      init(DEV_NAME, TOTAL_PAGES);
    }

    ~Device() {
      delete[] buddy_metadata;
      munmap(memory, TOTAL_PAGES * 4096);
      close(fd);
    }

    void reset() {
      cur_idx = 0;
    }

    char* alloc(int size) {
      // fprintf(stderr, "alloc %d %d\n", size, size / 4096);
      char* cur = (char*)buddy_malloc(buddy, size);
      if (!cur) {
        fprintf(stderr, "error in malloc\n");
        return nullptr;
      }
      return cur;
    }

    void free(char* buf) {
      // do nothing
      buddy_free(buddy, buf);
      return;
    }

    long virt_to_phys(char* buf) {
      return (buf - memory) + 0x1000;
    }

    int Run(vta_phy_addr_t insn_phy_addr,
            uint32_t insn_count,
            uint32_t wait_cycles) {
      if (fd <= 0)
        return 1;
      vta_exec_t exec = {
        .insn_phy_addr = insn_phy_addr,
        .insn_count = insn_count,
        .wait_cycles = wait_cycles,
        .status = 1
      };
      ioctl(fd, IOCTL_CMD_EXEC, &exec);
      return 0;
    }
  };
}

}  // namespace vta

vta::qemu::Device* global_dev = new vta::qemu::Device();

void* VTAMemAlloc(size_t size, int cached) {
  return global_dev->alloc(size);
}

void VTAMemFree(void* buf) {
  global_dev->free((char*)buf);
}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) {
  return global_dev->virt_to_phys((char*)buf);
}

void VTAMemCopyFromHost(void* dst, const void* src, size_t size) {
  // fprintf(stderr, "from host %x -> %x, %d\n", src, dst, size);
  uint32_t* dst32 = (uint32_t*) dst;
  uint32_t* src32 = (uint32_t*) src;
  for (int i = 0;i < size / sizeof(uint32_t);i++) {
    dst32[i] = src32[i];
  }
}

void VTAMemCopyToHost(void* dst, const void* src, size_t size) {
  // fprintf(stderr, "from device %x -> %x, %d\n", src, dst, size);
  uint32_t* dst32 = (uint32_t*) dst;
  uint32_t* src32 = (uint32_t*) src;
  for (int i = 0;i < size / sizeof(uint32_t);i++) {
    dst32[i] = src32[i];
  }
}

void VTAFlushCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
}

void VTAInvalidateCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
}

VTADeviceHandle VTADeviceAlloc() {
  return global_dev;
}

void VTADeviceFree(VTADeviceHandle handle) {
  global_dev->reset();
}

int VTADeviceRun(VTADeviceHandle handle,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  return static_cast<vta::qemu::Device*>(handle)->Run(
      insn_phy_addr, insn_count, wait_cycles);
}

void VTAProgram(const char* bitstream) {
}

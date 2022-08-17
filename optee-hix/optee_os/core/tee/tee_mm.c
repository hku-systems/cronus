/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <linux/pfn_t.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/slab_def.h>
#include <linux_headers/linux_include/linux/compiler.h>

#include "tee-port.h"

const uint8_t *__vheap_start =   (uint8_t *)0x13400000;
const uint8_t *__vheap_end =     (uint8_t *)0x333fffff;
const uint8_t *__vmap_space =    (uint8_t *)0x33400000;
const uint8_t *__vmap_space_end = (uint8_t *)0x533fffff;

s64			physvirt_offset;

extern int msize(void *ptr);

int iommu_map(struct iommu_domain *domain, unsigned long iova,
		     phys_addr_t paddr, size_t size, int prot) NO_IMPL_0;

size_t iommu_unmap(struct iommu_domain *domain, unsigned long iova,
			  size_t size) NO_IMPL_0;

extern void *phys_to_virt_io(uintptr_t pa);

int ioremap_page_range(unsigned long addr,
		       unsigned long end, phys_addr_t phys_addr, pgprot_t prot) {
    void* ret = phys_to_virt_io(phys_addr);
#ifdef DEBUG_CRONUS_MAPPING
    printk("call %s with (%lx,%lx) -> %lx with %lx\n", __FUNCTION__, addr, end, phys_addr, prot);
#endif
    if (ret != (void*)addr)
        panic("error in %s\n", __FUNCTION__);
    return 0;
}

void __iomem *__ioremap(phys_addr_t phys_addr, size_t size, pgprot_t prot) {
    void* ret = phys_to_virt_io(phys_addr);
#ifdef DEBUG_CRONUS_MAPPING
    printk("call %s with %lx, %lx, %lx -> %lx\n", __FUNCTION__ ,phys_addr, size, prot, ret);
#endif
    if (ret == NULL)
        panic("error in %s\n", __FUNCTION__);
    return ret;
}

void iounmap(volatile void __iomem *addr) {}

int remap_pfn_range(struct vm_area_struct * s, unsigned long addr,
			unsigned long pfn, unsigned long size, pgprot_t pgp) {
    unimplemented();
}

void unmap_kernel_range(unsigned long addr, unsigned long size) {
    unimplemented();
}

void __percpu *__alloc_percpu(size_t size, size_t align) {
    // TODO: enable smp
    uint32_t align_mask = align - 1;
    void* ret = malloc(size + align_mask);
    return (void*)((uintptr_t)(ret + align_mask) & ~align_mask);
}

void free_percpu(void __percpu *__pdata) {
    free(__pdata);
}

void *__kmalloc(size_t size, gfp_t flags)
{
    void* ret = malloc(size);

    if (!ret) {
        printk("OOM when allocating %d bytes", size);
        return ret;
    }

    if (flags & __GFP_ZERO) {
        memset(ret, 0, size);
    }

    return ret;
}

void *__kmalloc_track_caller(size_t size, gfp_t gfpflags, unsigned long caller)
{
    return malloc(size);
}

void * krealloc(const void *ptr, size_t size, gfp_t gfp) {
    return realloc(ptr, size);
}

size_t ksize(const void *ptr) {
    return msize(ptr);
}

void kfree(const void *x) {
    return free(x);
}

struct kmem_cache *kmem_cache_create(const char *name, unsigned int size,
                                     unsigned int align, slab_flags_t flags,
                                     void (*ctor)(void *)) {
    struct kmem_cache *kmc = kzalloc(sizeof(struct kmem_cache), GFP_KERNEL);

    if (align < ARCH_SLAB_MINALIGN)
        align = ARCH_SLAB_MINALIGN;

    align = ALIGN(align, sizeof(void *));

    kmc->align = align;
    kmc->ctor = ctor;
    kmc->object_size = size;
    return kmc;
}

void *kmem_cache_alloc(struct kmem_cache *cache, gfp_t flags) {
    uint32_t align = cache->align - 1;
    size_t align_total_meta = cache->object_size + align + sizeof(uint8_t);
    void* allocated = __kmalloc(align_total_meta, flags);
    void* ret = (void*)((uintptr_t)(allocated + align) & ~align);
    uint8_t *offset = ret + cache->object_size;
    *offset = ret - allocated;

    if (cache->ctor) {
      (*cache->ctor)(ret);
    }

    return ret;
}
void kmem_cache_free(struct kmem_cache *cache, void *ptr) {
    uint8_t *offset = ptr + cache->object_size;
    free(ptr - *offset);
}

// TODO: fix me
void *vmalloc(unsigned long size) {
    return kmalloc(size, GFP_KERNEL);
}

void *vzalloc(unsigned long size) {
    void* ret = vmalloc(size);
    if (!ret)
        return ret;
    memset(ret, 0, size);
    return ret;
}

void vfree(const void *addr) {
    free(addr);
}

static int vmap_pages_cur   = 0;
extern int vmap_pages_cnt   = 0;
uint8_t * vmap_cache = NULL;
struct page *vmemmap = NULL;
int vmemmap_pfn_offset = 0;

extern long __flush_cache_user_range(unsigned long start, unsigned long end);

static void init_vmap_cache() {
    long vmap_size = __vmap_space_end - __vmap_space + 1;
    int block_size = sizeof(struct page) + 4096;
    vmap_pages_cnt = vmap_size / block_size;
    vmap_cache = __vmap_space;
    vmemmap = vmap_cache + 4096 * vmap_pages_cnt;
    vmemmap_pfn_offset = PHYS_PFN(((long) 1 << 32) + (long)(__vmap_space - __vheap_start));
    physvirt_offset = ((long) 1 << 32) - (unsigned long)__vheap_start;
    memset(vmemmap, 0, sizeof(struct page) * vmap_pages_cnt);
    __flush_cache_user_range((long)__vmap_space, (long)__vmap_space_end);
    printk("init vmap at %lx/%lx with %lx block (%lx bytes)\n", vmap_cache, vmemmap, vmap_pages_cnt, block_size);
}

extern void *vmap(struct page **pages, unsigned int count,
			unsigned long flags, pgprot_t prot) {
    if (!vmap_cache) {
        panic("vmap_cache is null\n");
   }

    int first_pfn = page_to_pfn(pages[0]);
    int last_pfn = first_pfn;

    for (int i = 1;i < count;i++) {
        int cur = page_to_pfn(pages[i]);
        if (cur != last_pfn + 1)
            panic("pages are not consecutive\n");
        else
            last_pfn = cur;
    }
#ifdef DEBUG_CRONUS_MAPPING
    printk("vmap at %lx with %lx pages\n", first_pfn, count);
#endif
    return (void*) (vmap_cache + 4096 * (first_pfn - vmemmap_pfn_offset));
}

extern void vunmap(const void *addr) {}

extern long si_mem_available(void) {
    return 512 * 1024 * 1024;
}
extern void si_meminfo(struct sysinfo * val) NO_IMPL;

void *__vmalloc_node(unsigned long size, unsigned long align, gfp_t gfp_mask,
		int node, const void *caller) NO_IMPL;

void unmap_mapping_range(struct address_space *mapping,
		loff_t const holebegin, loff_t const holelen, int even_cows) NO_IMPL;

struct page *
__alloc_pages_nodemask(gfp_t gfp_mask, unsigned int order, int preferred_nid,
                       nodemask_t *nodemask) {
    struct page* p;

    if (!vmemmap) {
        init_vmap_cache();
    }

    int required_pages = (1 << order);
    if (vmap_pages_cur + required_pages >= vmap_pages_cnt) {
        printk("vmap failed");
        return NULL;
    }

    p = &vmemmap[vmap_pages_cur];
    vmap_pages_cur += required_pages;
#ifdef DEBUG_CRONUS_MAPPING
    printk("allocating %d page %lx -> %lx %lx\n", required_pages, p, page_to_pfn(p) * 4096 + vmap_cache, page_to_pfn(p));
#endif
    return p;
}

unsigned long __get_free_pages(gfp_t gfp_mask, unsigned int order) NO_IMPL_0;
void free_pages(unsigned long addr, unsigned int order) NO_IMPL;

void __free_pages(struct page *page, unsigned int order) NO_IMPL;

void copy_page(void *to, const void *from) NO_IMPL;

void clear_page(void *to) NO_IMPL;

void clear_page_mlock(struct page *page) NO_IMPL;

bool is_vmalloc_addr(const void *x) {
    const void *end = vmap_cache + vmap_pages_cnt * 4096;
    return (x >= vmap_cache && x < end);
}

void mark_page_accessed(struct page *page) NO_IMPL;
int set_page_dirty(struct page *page) NO_IMPL;

void __put_page(struct page *page) NO_IMPL;

//struct kmem_cache *
//        kmalloc_caches[NR_KMALLOC_TYPES][KMALLOC_SHIFT_HIGH + 1]; // TODO

unsigned long max_pfn;

void * high_memory = NULL;

int alloc_contig_range(unsigned long start, unsigned long end,
                       unsigned migratetype, gfp_t gfp_mask) NO_IMPL_0;
void free_contig_range(unsigned long pfn, unsigned int nr_pages) NO_IMPL;

void *vmalloc_32(unsigned long size) NO_IMPL_NULL;
void *vmalloc_user(unsigned long size) NO_IMPL_NULL;
struct page *vmalloc_to_page(const void *addr) NO_IMPL_NULL;
unsigned long do_mmap(struct file *file, unsigned long addr,
	unsigned long len, unsigned long prot, unsigned long flags,
	unsigned long pgoff, unsigned long *populate, struct list_head *uf) NO_IMPL_0;


int __mm_populate(unsigned long addr, unsigned long len,
			int ignore_errors) NO_IMPL_0;

u64 kimage_voffset = 0;

long strnlen_user(const char __user *str, long count) {
    return strnlen(str, count);
}

uint32_t do_copy_to_user(void *uaddr, const void *kaddr, size_t len);
uint32_t do_copy_from_user(void *kaddr, const void *uaddr, size_t len);
uint32_t do_clear_user(void *uaddr, size_t len);

unsigned long _copy_from_user(void *to, const void __user *from, unsigned long n) {
    if (do_copy_from_user(to, from , n)) {
        return n;
    }
    return 0;
}

unsigned long _copy_to_user(void __user *to, const void *from, unsigned long n) {
    if (do_copy_to_user(to, from , n)) {
        return n;
    }
    return 0;
}
unsigned long _clear_user(void __user *to, unsigned long n) {
    if (do_clear_user(to, n)) {
        return n;
    }
    return 0;
}
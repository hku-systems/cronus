/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <asm/irq.h>

#include <linux/compat.h>
#include <linux/fs.h>
#include <linux/i2c.h>
#include <linux/interrupt.h>
#include <linux/mount.h>
#include <linux/kernel.h>
#include <linux/platform_device.h>
#include <linux/rcupdate.h>
#include <linux/sched.h>
#include <linux/slab_def.h>
#include <linux/slab.h>
#include <linux/srcu.h>

#include "tee-port.h"

void add_taint(unsigned flag, enum lockdep_ok l) NO_IMPL;

void dump_stack(void) NO_IMPL;

int cpu_number = 1;

const char *print_tainted(void) NO_IMPL;

int ___ratelimit(struct ratelimit_state *rs, const char *func) NO_IMPL;

void device_release_driver(struct device *dev) NO_IMPL;

struct device* device_list[10];
int device_cnt = 0;

int driver_register(struct device_driver *drv) {
    if (strcmp(drv->name, "nouveau") == 0) {
        device_list[1]->driver = drv;
        drv->bus->probe(device_list[1]);
    }
    if (strcmp(drv->name, "vta_pci") == 0) {
        if (device_cnt >= 2) {
            device_list[2]->driver = drv;
            drv->bus->probe(device_list[2]);
        }
    }
}

bool gfp_pfmemalloc_allowed(gfp_t gfp_mask) NO_IMPL;

int autoremove_wake_function(struct wait_queue_entry *wq_entry, unsigned mode, int sync, void *key) {
    unimplemented();
}

int simple_pin_fs(struct file_system_type *fst, struct vfsmount **mount, int *count) {
    struct vfsmount *vfs = malloc(sizeof(struct vfsmount));
    *mount = vfs;
    unstable();
}

void simple_release_fs(struct vfsmount **mount, int *count) {
    free(*mount);
    *mount = NULL;
}

struct inode *alloc_anon_inode(struct super_block *sb) {
    struct inode *node = malloc(sizeof(struct inode));
    memset(node, 0, sizeof(struct inode));
    return node;
}

void set_user_nice(struct task_struct *p, long nice) NO_IMPL;

void hdmi_avi_infoframe_init(struct hdmi_avi_infoframe *frame) NO_IMPL;
ssize_t hdmi_infoframe_pack(union hdmi_infoframe *frame, void *buffer,
			    size_t size) NO_IMPL;
int hdmi_vendor_infoframe_init(struct hdmi_vendor_infoframe *frame) NO_IMPL;

// srcu impl TODO: performance
int __srcu_read_lock(struct srcu_struct *ssp) {
    mutex_lock(&ssp->srcu_cb_mutex);
	return 0;
}

void __srcu_read_unlock(struct srcu_struct *ssp, int idx) {
    mutex_unlock(&ssp->srcu_cb_mutex);
}

/**
  * RCU impl
  * 1. we do not impl call_rcu -> memory leak
  * 2. we wait for all other to get out of the read-side critical session
  */

volatile unsigned long long rcu_cpu_locks = 0;

void call_rcu(struct rcu_head *head, rcu_callback_t func) {}
int rcuwait_wake_up(struct rcuwait *w) { return 0; }
void kvfree_call_rcu(struct rcu_head *head, rcu_callback_t func) {}
void synchronize_rcu(void) {
	// wait until all others get out of read-critical sections
	while (rcu_cpu_locks) {}
}

void __rcu_read_lock(void) {
	rcu_cpu_locks = 1;
}

void __rcu_read_unlock(void) {
	rcu_cpu_locks = 0;
}

loff_t noop_llseek(struct file *file, loff_t offset, int whence) {
    unimplemented();
}

static inline int __device_attach(struct device *dev, bool allow_async) {
    unstable();
    for (int i = 0;i < device_cnt;i++) {
        if (dev == device_list[i]) {
            printk("device %s has been added to list", dev_name(dev));
            return 1;
        }
    }
    printk("device %s added to list-%d", dev_name(dev), device_cnt);
    device_list[device_cnt++] = dev;
    if (dev->bus->dma_configure) {
        dev->bus->dma_configure(dev);
    }
    return 0;
}

void device_initial_probe(struct device *dev) {
    __device_attach(dev, true);
}

int device_attach(struct device *dev) {
    return __device_attach(dev, false);
}

void device_remove_properties(struct device *dev) NO_IMPL;
void driver_deferred_probe_del(struct device *dev) NO_IMPL;

extern void __do_panic(const char *file __maybe_unused,
		const int line __maybe_unused,
		const char *func __maybe_unused,
		const char *msg __maybe_unused);

extern void __panic(const char *file, const int line, const char *func);

void panic(const char *fmt, ...) {
    va_list ap;
	va_start(ap, fmt);
	trace_vprintf(0, 0, 0, 0, fmt, ap);
	va_end(ap);
	__do_panic(__FILE__, __LINE__, __FUNCTION__, "");
}

// TODO: fix this, we have to enforce a empty impl
int software_node_notify(struct device *dev, unsigned long action) {
    return 0;
}

int driver_attach(struct device_driver *drv) {
    unimplemented();
}

int blocking_notifier_call_chain(struct blocking_notifier_head *nh,
		unsigned long val, void *v) {
    unimplemented();
}

#define DMA_BIT_MASK(n)	(((n) == 64) ? ~0ULL : ((1ULL<<(n))-1))

int of_dma_configure_id(struct device *dev,
		     struct device_node *np,
		     bool force_dma, const u32 *id) {
	// u64 dma_addr, paddr, size = 0;
	// int ret;
	// bool coherent;
	// unsigned long offset;
	// const struct iommu_ops *iommu;
	// u64 mask, end;

    // dma_addr = offset = 0;

	// /*
	//  * If @dev is expected to be DMA-capable then the bus code that created
	//  * it should have initialised its dma_mask pointer by this point. For
	//  * now, we'll continue the legacy behaviour of coercing it to the
	//  * coherent mask if not, but we'll no longer do so quietly.
	//  */
	// if (!dev->dma_mask) {
	// 	dev_warn(dev, "DMA mask not set\n");
	// 	dev->dma_mask = &dev->coherent_dma_mask;
	// }

	// if (!size && dev->coherent_dma_mask)
	// 	size = max(dev->coherent_dma_mask, dev->coherent_dma_mask + 1);
	// else if (!size)
	// 	size = 1ULL << 32;

	// dev->dma_pfn_offset = offset;

	// /*
	//  * Limit coherent and dma mask based on size and default mask
	//  * set by the driver.
	//  */
	// end = dma_addr + size - 1;
	// mask = DMA_BIT_MASK(ilog2(end) + 1);
	// dev->coherent_dma_mask &= mask;
	// *dev->dma_mask &= mask;
	// /* ...but only set bus limit if we found valid dma-ranges earlier */
	// if (!ret)
	// 	dev->bus_dma_limit = end;

	// coherent = true;
	// dev_dbg(dev, "device is%sdma coherent\n",
	// 	coherent ? " " : " not ");

	// iommu = NULL;
	// if (PTR_ERR(iommu) == -EPROBE_DEFER)
	// 	return -EPROBE_DEFER;

	// dev_dbg(dev, "device is%sbehind an iommu\n",
	// 	iommu ? " " : " not ");

	dev->dma_coherent = true;

	return 0;
}

void of_device_uevent(struct device *dev, struct kobj_uevent_env *env) NO_IMPL;

struct device_node *of_chosen;

gfp_t gfp_allowed_mask;

struct pglist_data __refdata contig_page_data;

u64 vabits_actual = 54;

atomic_long_t vm_node_stat[NR_VM_NODE_STAT_ITEMS];

bool arm64_use_ng_mappings = false;

struct task_struct task_local[CFG_TEE_CORE_NB_CORE];

int oops_in_progress = 0;

void* get_thread_local_task(void) {
    size_t pos = __get_core_pos();
    return &task_local[pos];
}

void orderly_poweroff(bool force) NO_IMPL;

int fb_get_options(const char *name, char **option) NO_IMPL;
int remove_conflicting_pci_framebuffers(struct pci_dev *pdev,
                                               const char *name) NO_IMPL;

void kill_anon_super(struct super_block *sb) NO_IMPL;

// i2c dummy
int i2c_add_adapter(struct i2c_adapter *adap) NO_IMPL;
void i2c_del_adapter(struct i2c_adapter *adap) NO_IMPL;
int i2c_bit_add_bus(struct i2c_adapter *adap) NO_IMPL;
const struct i2c_algorithm i2c_bit_algo;
struct i2c_client *
i2c_new_client_device(struct i2c_adapter *adap, struct i2c_board_info const *info) NO_IMPL;
int i2c_transfer(struct i2c_adapter *adap, struct i2c_msg *msgs, int num) NO_IMPL;
void i2c_unregister_device(struct i2c_client *client) NO_IMPL;

// component ops
int component_add(struct device *device, const struct component_ops *component_op) NO_IMPL;
void component_del(struct device *device, const struct component_ops *component_op) NO_IMPL;

void fput(struct file *file) NO_IMPL;
void iput(struct inode *node) NO_IMPL;

extern int request_irq_internal(int irq, irq_handler_t handler, irq_handler_t thread_fn, const char *name, void *dev);

int request_threaded_irq(unsigned int irq, irq_handler_t handler,
                     irq_handler_t thread_fn,
                     unsigned long flags, const char *name, void *dev) {
    request_irq_internal(irq, handler, thread_fn, name, dev);
	return 0;
}

const void *free_irq(unsigned int irq, void *dev_id) NO_IMPL;

bool irq_work_queue(struct irq_work *work) NO_IMPL;

struct pseudo_fs_context *init_pseudo(struct fs_context *fc,
                                      unsigned long magic) NO_IMPL;

void put_pid(struct pid *pid) NO_IMPL;

int register_shrinker(struct shrinker *shrinker) NO_IMPL;
void unregister_shrinker(struct shrinker *shrinker) NO_IMPL;

struct file *shmem_file_setup(const char *name,
                              loff_t size, unsigned long flags) {
	// TODO: hack
	return (struct file*) 1;
}
struct page *shmem_read_mapping_page_gfp(struct address_space *mapping,
                                         pgoff_t index, gfp_t gfp_mask) NO_IMPL_0;

#define __ARM64_FTR_BITS(SIGNED, VISIBLE, STRICT, TYPE, SHIFT, WIDTH, SAFE_VAL) \
	{						\
		.sign = SIGNED,				\
		.visible = VISIBLE,			\
		.strict = STRICT,			\
		.type = TYPE,				\
		.shift = SHIFT,				\
		.width = WIDTH,				\
		.safe_val = SAFE_VAL,			\
	}

/* Define a feature with unsigned values */
#define ARM64_FTR_BITS(VISIBLE, STRICT, TYPE, SHIFT, WIDTH, SAFE_VAL) \
	__ARM64_FTR_BITS(FTR_UNSIGNED, VISIBLE, STRICT, TYPE, SHIFT, WIDTH, SAFE_VAL)

#define ARM64_FTR_END					\
	{						\
		.width = 0,				\
	}

static const struct arm64_ftr_bits ftr_ctr[] = {
        ARM64_FTR_BITS(FTR_VISIBLE, FTR_STRICT, FTR_EXACT, 31, 1, 1), /* RES1 */
        ARM64_FTR_BITS(FTR_VISIBLE, FTR_STRICT, FTR_LOWER_SAFE, CTR_DIC_SHIFT, 1, 1),
        ARM64_FTR_BITS(FTR_VISIBLE, FTR_STRICT, FTR_LOWER_SAFE, CTR_IDC_SHIFT, 1, 1),
        ARM64_FTR_BITS(FTR_VISIBLE, FTR_STRICT, FTR_HIGHER_OR_ZERO_SAFE, CTR_CWG_SHIFT, 4, 0),
        ARM64_FTR_BITS(FTR_VISIBLE, FTR_STRICT, FTR_HIGHER_OR_ZERO_SAFE, CTR_ERG_SHIFT, 4, 0),
        ARM64_FTR_BITS(FTR_VISIBLE, FTR_STRICT, FTR_LOWER_SAFE, CTR_DMINLINE_SHIFT, 4, 1),
        /*
         * Linux can handle differing I-cache policies. Userspace JITs will
         * make use of *minLine.
         * If we have differing I-cache policies, report it as the weakest - VIPT.
         */
        ARM64_FTR_BITS(FTR_VISIBLE, FTR_NONSTRICT, FTR_EXACT, CTR_L1IP_SHIFT, 2, ICACHE_POLICY_VIPT),	/* L1Ip */
        ARM64_FTR_BITS(FTR_VISIBLE, FTR_STRICT, FTR_LOWER_SAFE, CTR_IMINLINE_SHIFT, 4, 0),
        ARM64_FTR_END,
};


struct arm64_ftr_reg arm64_ftr_reg_ctrel0 = {
        .name		= "SYS_CTR_EL0",
        .ftr_bits	= ftr_ctr
};

struct file *anon_inode_getfile(const char *name,
				const struct file_operations *fops,
				void *priv, int flags) NO_IMPL;

void inode_set_bytes(struct inode *inode, loff_t bytes) NO_IMPL;

long compat_ptr_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	if (!file->f_op->unlocked_ioctl)
		return -ENOIOCTLCMD;

	return file->f_op->unlocked_ioctl(file, cmd, (unsigned long)compat_ptr(arg));
}

struct file *alloc_file_pseudo(struct inode *node, struct vfsmount *vfs,
	const char *c, int flags, const struct file_operations *fops) NO_IMPL_NULL;

unsigned long __fdget(unsigned int fd) NO_IMPL_0;
void fd_install(unsigned int fd, struct file *file) NO_IMPL;
struct file *fget(unsigned int fd) NO_IMPL_NULL;
int get_unused_fd_flags(unsigned flags) NO_IMPL_0;
void put_unused_fd(unsigned int fd) NO_IMPL;
struct file * dentry_open(const struct path *p, int i, const struct cred *c) NO_IMPL_NULL;

int overflowuid = 0;

int send_sig(int signum, struct task_struct *task, int d) NO_IMPL;


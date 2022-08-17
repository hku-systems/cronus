// SPDX-License-Identifier: BSD-2-Clause
/*
 * Copyright (c) 2014, STMicroelectronics International N.V.
 * Copyright (c) 2015-2020 Linaro Limited
 * Copyright (c) 2020, Arm Limited.
 */

#include <assert.h>
#include <compiler.h>
#include <crypto/crypto.h>
#include <ctype.h>
#include <initcall.h>
#include <keep.h>
#include <kernel/ldelf_loader.h>
#include <kernel/linker.h>
#include <kernel/panic.h>
#include <kernel/tee_misc.h>
#include <kernel/tee_ta_manager.h>
#include <kernel/thread.h>
#include <kernel/ts_store.h>
#include <kernel/user_access.h>
#include <kernel/user_mode_ctx.h>
#include <kernel/user_ta.h>
#include <mm/core_memprot.h>
#include <mm/core_mmu.h>
#include <mm/file.h>
#include <mm/fobj.h>
#include <mm/mobj.h>
#include <mm/pgt_cache.h>
#include <mm/tee_mm.h>
#include <mm/tee_pager.h>
#include <mm/vm.h>
#include <optee_rpc_cmd.h>
#include <printk.h>
#include <signed_hdr.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/queue.h>
#include <ta_pub_key.h>
#include <tee/arch_svc.h>
#include <tee/tee_cryp_utl.h>
#include <tee/tee_obj.h>
#include <tee/tee_svc_cryp.h>
#include <tee/tee_svc.h>
#include <tee/tee_svc_storage.h>
#include <tee/uuid.h>
#include <trace.h>
#include <types_ext.h>
#include <utee_defines.h>
#include <util.h>

static void init_utee_param(struct utee_params *up,
			const struct tee_ta_param *p, void *va[TEE_NUM_PARAMS])
{
	size_t n;

	up->types = p->types;
	for (n = 0; n < TEE_NUM_PARAMS; n++) {
		uintptr_t a;
		uintptr_t b;

		switch (TEE_PARAM_TYPE_GET(p->types, n)) {
		case TEE_PARAM_TYPE_MEMREF_INPUT:
		case TEE_PARAM_TYPE_MEMREF_OUTPUT:
		case TEE_PARAM_TYPE_MEMREF_INOUT:
			a = (uintptr_t)va[n];
			b = p->u[n].mem.size;
			break;
		case TEE_PARAM_TYPE_VALUE_INPUT:
		case TEE_PARAM_TYPE_VALUE_INOUT:
			a = p->u[n].val.a;
			b = p->u[n].val.b;
			break;
		default:
			a = 0;
			b = 0;
			break;
		}
		/* See comment for struct utee_params in utee_types.h */
		up->vals[n * 2] = a;
		up->vals[n * 2 + 1] = b;
	}
}

static void update_from_utee_param(struct tee_ta_param *p,
			const struct utee_params *up)
{
	size_t n;

	for (n = 0; n < TEE_NUM_PARAMS; n++) {
		switch (TEE_PARAM_TYPE_GET(p->types, n)) {
		case TEE_PARAM_TYPE_MEMREF_OUTPUT:
		case TEE_PARAM_TYPE_MEMREF_INOUT:
			/* See comment for struct utee_params in utee_types.h */
			p->u[n].mem.size = up->vals[n * 2 + 1];
			break;
		case TEE_PARAM_TYPE_VALUE_OUTPUT:
		case TEE_PARAM_TYPE_VALUE_INOUT:
			/* See comment for struct utee_params in utee_types.h */
			p->u[n].val.a = up->vals[n * 2];
			p->u[n].val.b = up->vals[n * 2 + 1];
			break;
		default:
			break;
		}
	}
}

static bool inc_recursion(void)
{
	struct thread_specific_data *tsd = thread_get_tsd();

	if (tsd->syscall_recursion >= CFG_CORE_MAX_SYSCALL_RECURSION) {
		DMSG("Maximum allowed recursion depth reached (%u)",
		     CFG_CORE_MAX_SYSCALL_RECURSION);
		return false;
	}

	tsd->syscall_recursion++;
	return true;
}

static void dec_recursion(void)
{
	struct thread_specific_data *tsd = thread_get_tsd();

	assert(tsd->syscall_recursion);
	tsd->syscall_recursion--;
}

static TEE_Result user_ta_enter(struct ts_session *session,
				enum utee_entry_func func, uint32_t cmd)
{
	TEE_Result res = TEE_SUCCESS;
	struct utee_params *usr_params = NULL;
	uaddr_t usr_stack = 0;
	struct user_ta_ctx *utc = to_user_ta_ctx(session->ctx);
	struct tee_ta_session *ta_sess = to_ta_session(session);
	struct ts_session *ts_sess __maybe_unused = NULL;
	void *param_va[TEE_NUM_PARAMS] = { NULL };

	if (!inc_recursion()) {
		/* Using this error code since we've run out of resources. */
		res = TEE_ERROR_OUT_OF_MEMORY;
		goto out_clr_cancel;
	}
	if (ta_sess->param) {
		/* Map user space memory */
		res = vm_map_param(&utc->uctx, ta_sess->param, param_va);
		if (res != TEE_SUCCESS)
			goto out;
	}

	/* Switch to user ctx */
	ts_push_current_session(session);

	/* Make room for usr_params at top of stack */
	usr_stack = utc->uctx.stack_ptr;
	usr_stack -= ROUNDUP(sizeof(struct utee_params), STACK_ALIGNMENT);
	usr_params = (struct utee_params *)usr_stack;
	if (ta_sess->param)
		init_utee_param(usr_params, ta_sess->param, param_va);
	else
		memset(usr_params, 0, sizeof(*usr_params));

	res = thread_enter_user_mode(func, kaddr_to_uref(session),
				     (vaddr_t)usr_params, cmd, usr_stack,
				     utc->uctx.entry_func, utc->uctx.is_32bit,
				     &utc->ta_ctx.panicked,
				     &utc->ta_ctx.panic_code);

	thread_user_clear_vfp(&utc->uctx);

	if (utc->ta_ctx.panicked) {
		abort_print_current_ts();
		DMSG("tee_user_ta_enter: TA panicked with code 0x%x",
		     utc->ta_ctx.panic_code);
		res = TEE_ERROR_TARGET_DEAD;
	} else {
		/*
		 * According to GP spec the origin should allways be set to
		 * the TA after TA execution
		 */
		ta_sess->err_origin = TEE_ORIGIN_TRUSTED_APP;
	}

	if (ta_sess->param) {
		/* Copy out value results */
		update_from_utee_param(ta_sess->param, usr_params);

		/*
		 * Clear out the parameter mappings added with
		 * vm_clean_param() above.
		 */
		vm_clean_param(&utc->uctx);
	}


	ts_sess = ts_pop_current_session();
	assert(ts_sess == session);

out:
	dec_recursion();
out_clr_cancel:
	/*
	 * Clear the cancel state now that the user TA has returned. The next
	 * time the TA will be invoked will be with a new operation and should
	 * not have an old cancellation pending.
	 */
	ta_sess->cancel = false;

	return res;
}

static TEE_Result user_ta_enter_open_session(struct ts_session *s)
{
	return user_ta_enter(s, UTEE_ENTRY_FUNC_OPEN_SESSION, 0);
}

static TEE_Result user_ta_enter_invoke_cmd(struct ts_session *s, uint32_t cmd)
{
	return user_ta_enter(s, UTEE_ENTRY_FUNC_INVOKE_COMMAND, cmd);
}

static void user_ta_enter_close_session(struct ts_session *s)
{
	/* Only if the TA was fully initialized by ldelf */
	if (!to_user_ta_ctx(s->ctx)->uctx.is_initializing)
		user_ta_enter(s, UTEE_ENTRY_FUNC_CLOSE_SESSION, 0);
}

static void dump_state_no_ldelf_dbg(struct user_ta_ctx *utc)
{
	user_mode_ctx_print_mappings(&utc->uctx);
}

static void user_ta_dump_state(struct ts_ctx *ctx)
{
	struct user_ta_ctx *utc = to_user_ta_ctx(ctx);

	if (utc->uctx.dump_entry_func) {
		TEE_Result res = ldelf_dump_state(&utc->uctx);

		if (!res || res == TEE_ERROR_TARGET_DEAD)
			return;
		/*
		 * Fall back to dump_state_no_ldelf_dbg() if
		 * ldelf_dump_state() fails for some reason.
		 *
		 * If ldelf_dump_state() failed with panic
		 * we are done since abort_print_current_ts() will be
		 * called which will dump the memory map.
		 */
	}

	dump_state_no_ldelf_dbg(utc);
}

#ifdef CFG_FTRACE_SUPPORT
static void user_ta_dump_ftrace(struct ts_ctx *ctx)
{
	uint32_t prot = TEE_MATTR_URW;
	struct user_ta_ctx *utc = to_user_ta_ctx(ctx);
	struct thread_param params[3] = { };
	TEE_Result res = TEE_SUCCESS;
	struct mobj *mobj = NULL;
	uint8_t *ubuf = NULL;
	void *buf = NULL;
	size_t pl_sz = 0;
	size_t blen = 0, ld_addr_len = 0;
	vaddr_t va = 0;

	res = ldelf_dump_ftrace(&utc->uctx, NULL, &blen);
	if (res != TEE_ERROR_SHORT_BUFFER)
		return;

#define LOAD_ADDR_DUMP_SIZE	64
	pl_sz = ROUNDUP(blen + sizeof(TEE_UUID) + LOAD_ADDR_DUMP_SIZE,
			SMALL_PAGE_SIZE);

	mobj = thread_rpc_alloc_payload(pl_sz);
	if (!mobj) {
		EMSG("Ftrace thread_rpc_alloc_payload failed");
		return;
	}

	buf = mobj_get_va(mobj, 0);
	if (!buf)
		goto out_free_pl;

	res = vm_map(&utc->uctx, &va, mobj->size, prot, VM_FLAG_EPHEMERAL,
		     mobj, 0);
	if (res)
		goto out_free_pl;

	ubuf = (uint8_t *)va + mobj_get_phys_offs(mobj, mobj->phys_granule);
	memcpy(ubuf, &ctx->uuid, sizeof(TEE_UUID));
	ubuf += sizeof(TEE_UUID);

	ld_addr_len = snprintk((char *)ubuf, LOAD_ADDR_DUMP_SIZE,
			       "TEE load address @ %#"PRIxVA"\n",
			       VCORE_START_VA);
	ubuf += ld_addr_len;

	res = ldelf_dump_ftrace(&utc->uctx, ubuf, &blen);
	if (res) {
		EMSG("Ftrace dump failed: %#"PRIx32, res);
		goto out_unmap_pl;
	}

	params[0] = THREAD_PARAM_VALUE(INOUT, 0, 0, 0);
	params[1] = THREAD_PARAM_MEMREF(IN, mobj, 0, sizeof(TEE_UUID));
	params[2] = THREAD_PARAM_MEMREF(IN, mobj, sizeof(TEE_UUID),
					blen + ld_addr_len);

	res = thread_rpc_cmd(OPTEE_RPC_CMD_FTRACE, 3, params);
	if (res)
		EMSG("Ftrace thread_rpc_cmd res: %#"PRIx32, res);

out_unmap_pl:
	res = vm_unmap(&utc->uctx, va, mobj->size);
	assert(!res);
out_free_pl:
	thread_rpc_free_payload(mobj);
}
#endif /*CFG_FTRACE_SUPPORT*/

#ifdef CFG_TA_GPROF_SUPPORT
static void user_ta_gprof_set_status(enum ts_gprof_status status)
{
	if (status == TS_GPROF_SUSPEND)
		tee_ta_update_session_utime_suspend();
	else
		tee_ta_update_session_utime_resume();
}
#endif /*CFG_TA_GPROF_SUPPORT*/

static void free_utc(struct user_ta_ctx *utc)
{
	tee_pager_rem_um_areas(&utc->uctx);

	/*
	 * Close sessions opened by this TA
	 * Note that tee_ta_close_session() removes the item
	 * from the utc->open_sessions list.
	 */
	while (!TAILQ_EMPTY(&utc->open_sessions)) {
		tee_ta_close_session(TAILQ_FIRST(&utc->open_sessions),
				     &utc->open_sessions, KERN_IDENTITY);
	}

	vm_info_final(&utc->uctx);

	/* Free cryp states created by this TA */
	tee_svc_cryp_free_states(utc);
	/* Close cryp objects opened by this TA */
	tee_obj_close_all(utc);
	/* Free emums created by this TA */
	tee_svc_storage_close_all_enum(utc);
	free(utc);
}

static void user_ta_ctx_destroy(struct ts_ctx *ctx)
{
	free_utc(to_user_ta_ctx(ctx));
}

static uint32_t user_ta_get_instance_id(struct ts_ctx *ctx)
{
	return to_user_ta_ctx(ctx)->uctx.vm_info.asid;
}

static const struct ts_ops user_ta_ops __rodata_unpaged = {
	.enter_open_session = user_ta_enter_open_session,
	.enter_invoke_cmd = user_ta_enter_invoke_cmd,
	.enter_close_session = user_ta_enter_close_session,
	.dump_state = user_ta_dump_state,
#ifdef CFG_FTRACE_SUPPORT
	.dump_ftrace = user_ta_dump_ftrace,
#endif
	.destroy = user_ta_ctx_destroy,
	.get_instance_id = user_ta_get_instance_id,
	.handle_svc = user_ta_handle_svc,
#ifdef CFG_TA_GPROF_SUPPORT
	.gprof_set_status = user_ta_gprof_set_status,
#endif
};

/*
 * Break unpaged attribute dependency propagation to user_ta_ops structure
 * content thanks to a runtime initialization of the ops reference.
 */
static const struct ts_ops *_user_ta_ops;

static TEE_Result init_user_ta(void)
{
	_user_ta_ops = &user_ta_ops;

	return TEE_SUCCESS;
}
service_init(init_user_ta);

static void set_ta_ctx_ops(struct tee_ta_ctx *ctx)
{
	ctx->ts_ctx.ops = _user_ta_ops;
}

bool is_user_ta_ctx(struct ts_ctx *ctx)
{
	return ctx && ctx->ops == _user_ta_ops;
}

static TEE_Result check_ta_store(void)
{
	const struct ts_store_ops *op = NULL;

	SCATTERED_ARRAY_FOREACH(op, ta_stores, struct ts_store_ops)
		DMSG("TA store: \"%s\"", op->description);

	return TEE_SUCCESS;
}
service_init(check_ta_store);

TEE_Result tee_ta_init_user_ta_session(const TEE_UUID *uuid,
				       struct tee_ta_session *s)
{
	TEE_Result res = TEE_SUCCESS;
	struct user_ta_ctx *utc = NULL;

	utc = calloc(1, sizeof(struct user_ta_ctx));
	if (!utc)
		return TEE_ERROR_OUT_OF_MEMORY;

	utc->uctx.is_initializing = true;
	TAILQ_INIT(&utc->open_sessions);
	TAILQ_INIT(&utc->cryp_states);
	TAILQ_INIT(&utc->objects);
	TAILQ_INIT(&utc->storage_enums);
	condvar_init(&utc->ta_ctx.busy_cv);
	utc->ta_ctx.ref_count = 1;

	utc->uctx.ts_ctx = &utc->ta_ctx.ts_ctx;

	/*
	 * Set context TA operation structure. It is required by generic
	 * implementation to identify userland TA versus pseudo TA contexts.
	 */
	set_ta_ctx_ops(&utc->ta_ctx);

	utc->ta_ctx.ts_ctx.uuid = *uuid;
	res = vm_info_init(&utc->uctx);
	if (res)
		goto out;

	mutex_lock(&tee_ta_mutex);
	s->ts_sess.ctx = &utc->ta_ctx.ts_ctx;
	s->ts_sess.handle_svc = s->ts_sess.ctx->ops->handle_svc;
	/*
	 * Another thread trying to load this same TA may need to wait
	 * until this context is fully initialized. This is needed to
	 * handle single instance TAs.
	 */
	TAILQ_INSERT_TAIL(&tee_ctxes, &utc->ta_ctx, link);
	mutex_unlock(&tee_ta_mutex);

	/*
	 * We must not hold tee_ta_mutex while allocating page tables as
	 * that may otherwise lead to a deadlock.
	 */
	ts_push_current_session(&s->ts_sess);

	res = ldelf_load_ldelf(&utc->uctx);
	if (!res)
		res = ldelf_init_with_ldelf(&s->ts_sess, &utc->uctx);

	ts_pop_current_session();

	mutex_lock(&tee_ta_mutex);

	if (!res) {
		utc->uctx.is_initializing = false;
	} else {
		s->ts_sess.ctx = NULL;
		TAILQ_REMOVE(&tee_ctxes, &utc->ta_ctx, link);
	}

	/* The state has changed for the context, notify eventual waiters. */
	condvar_broadcast(&tee_ta_init_cv);

	mutex_unlock(&tee_ta_mutex);

out:
	if (res) {
		condvar_destroy(&utc->ta_ctx.busy_cv);
		pgt_flush_ctx(&utc->ta_ctx.ts_ctx);
		free_utc(utc);
	}

	return res;
}

#define POSIX_FS posix_direct_ree_fs

#define TRACE_RET(ret) \
	if (ret != TEE_SUCCESS) \
		DMSG("TA_SYSCALL %s -> %lx\n", __FUNCTION__, ret); \
	return ret;

#define TRACE_RET2(ret, sysret) \
	if (ret != TEE_SUCCESS) \
		EMSG("TA_SYSCALL %s -> %lx (%d)\n", __FUNCTION__, ret, *sysret); \
	return ret;

// TODO: read write on dev
TEE_Result syscall_fs_read(int fd, void *buf, size_t count, size_t off, ssize_t *rsize) {
	TEE_Result ret = POSIX_FS.read(fd, buf, count, off, rsize);
	TRACE_RET2(ret, rsize)
}

TEE_Result syscall_fs_write(int fd, const void *buf, size_t count, size_t off, ssize_t *wsize) {
	TEE_Result ret =  POSIX_FS.write(fd, buf, count, off, wsize);
	TRACE_RET2(ret, wsize)
}

TEE_Result syscall_dev_ioctl(int fd, unsigned long request, void* args, int *ret_val) {
	TEE_Result ret = posix_device_ctl.ioctl(fd, request, args, ret_val);
	TRACE_RET2(ret, ret_val)
}

// TODO: better fd management, current POSIX_DEV_MINIMUM_FD is for static fd partition
TEE_Result syscall_posix_open(const char *filename, int flag, int mode, int *fd) {
	TEE_Result ret;
	if (filename == NULL) {
		TRACE_RET(TEE_ERROR_BAD_PARAMETERS)
	}

	if (strlen(filename) < strlen(dev_prefix) || strncmp(filename, dev_prefix, strlen(dev_prefix))) {
		ret = POSIX_FS.open(filename, flag, mode, fd);
	} else {
		ret = posix_device_ctl.open(filename, flag, mode, fd);
	}
	TRACE_RET2(ret, fd);
}

TEE_Result syscall_posix_close(int fd) {
	TEE_Result ret = (fd >= POSIX_DEV_MINIMUM_FD)? posix_device_ctl.close(fd) : POSIX_FS.close(fd);
	TRACE_RET(ret);
}

#define PROT_READ	0x1		/* page can be read */
#define PROT_WRITE	0x2		/* page can be written */
#define PROT_EXEC	0x4		/* page can be executed */
#define PROT_SEM	0x8		/* page may be used for atomic ops */
/*			0x10		   reserved for arch-specific use */
/*			0x20		   reserved for arch-specific use */
#define PROT_NONE	0x0		/* page can not be accessed */
#define PROT_GROWSDOWN	0x01000000	/* mprotect flag: extend change to start of growsdown vma */
#define PROT_GROWSUP	0x02000000	/* mprotect flag: extend change to end of growsup vma */

#define MAP_SHARED	0x01		/* Share changes */
#define MAP_PRIVATE	0x02		/* Changes are private */
#define MAP_SHARED_VALIDATE 0x03	/* share + validate extension flags */

extern void* mmap_fd(void *addr, size_t length, int prot, int flag,
				int fd, long offset);

TEE_Result syscall_mmap(void *addr, size_t length, int prot, int flags,
                int fd, long offset, void **mapped_addr) {
	TEE_Result ret = TEE_SUCCESS;
	struct ts_session *sess = ts_get_current_session();
	struct user_mode_ctx *uctx = to_user_mode_ctx(sess->ctx);
	struct fobj *f = NULL;
	struct mobj *mobj = NULL;
	uint32_t vm_flags = 0;
	vaddr_t *va = mapped_addr;
	int tee_prot = 0;
	int num_rounded_bytes;
	long ta_map_base = 0;
	size_t ta_size;

	*va = 0;

	if (addr) {
		printk("mmap at %lx is not implemented\n", addr);
	}

	if (fd != 0) {
		*mapped_addr = mmap_fd(addr, length, prot, flags, fd, offset);
		if (*mapped_addr == NULL) {
			return TEE_ERROR_BAD_PARAMETERS;
		}
		return TEE_SUCCESS;
	}

	if (flags & MAP_SHARED)
		vm_flags |= VM_FLAG_SHAREABLE;
	if (prot & PROT_READ)
		tee_prot |= TEE_MATTR_UR;
	if (prot & PROT_WRITE)
		tee_prot |= TEE_MATTR_UW;
	if (prot & PROT_EXEC)
		tee_prot |= TEE_MATTR_UX;

	if (flags & MAP_SHARED)
		goto fin;

	if (ROUNDUP_OVERFLOW(length, SMALL_PAGE_SIZE, &num_rounded_bytes))
		return TEE_ERROR_BAD_PARAMETERS;

	f = fobj_ta_mem_alloc(ROUNDUP_DIV(length, SMALL_PAGE_SIZE));
	if (!f) {
		ret = TEE_ERROR_OUT_OF_MEMORY;
		goto fin;
	}

	mobj = mobj_with_fobj_alloc(f, NULL);
	fobj_put(f);
	if (!mobj) {
		ret = TEE_ERROR_OUT_OF_MEMORY;
		goto fin;
	}
	ret = vm_map_pad(uctx, va, num_rounded_bytes, tee_prot, vm_flags,
			mobj, 0, 0, 0, 0);
	mobj_put(mobj);

fin:
	if (!*va) {
		printk("try mapping a shared ta memory\n");
		core_mmu_get_user_va_range(&ta_map_base, &ta_size);
		ta_map_base += ((16 * 8) << 20) - length;
		ret = map_virt_phys(&ta_map_base, ((long)1 << 32) + ((long)1 << 31), length, 0);
		if (ret) {
			printk("map %lx (%lx) failed\n", ((long)1 << 32) + ((long)1 << 31), length);
		}
		*va = ta_map_base;
	}
	TRACE_RET2(ret, mapped_addr);
}

TEE_Result syscall_mremap(void *old_address, size_t old_size,
            size_t new_size, int flags, void **new_addr) {
	TEE_Result res = TEE_SUCCESS;
	struct ts_session *sess = ts_get_current_session();
	struct user_mode_ctx *uctx = to_user_mode_ctx(sess->ctx);
	uint32_t vm_flags = 0;
	vaddr_t old_va = (vaddr_t)old_address;
	vaddr_t *new_va = (vaddr_t*) new_addr;

	res = vm_get_flags(uctx, old_va, old_size, &vm_flags);
	if (res)
		goto fin;
	if (vm_flags & VM_FLAG_PERMANENT) {
		res = TEE_ERROR_ACCESS_DENIED;
		goto fin;
	}

	res = vm_remap(uctx, new_va, old_va, old_size, 0, 0);

fin:
	TRACE_RET2(res, new_addr);
}

TEE_Result syscall_munmap(void *addr, size_t num_bytes, int *ret_status) {
	TEE_Result res = TEE_SUCCESS;
	struct ts_session *sess = ts_get_current_session();
	struct user_mode_ctx *uctx = to_user_mode_ctx(sess->ctx);
	size_t sz = ROUNDUP(num_bytes, SMALL_PAGE_SIZE);
	uint32_t vm_flags = 0;
	vaddr_t end_va = 0;
	vaddr_t va = addr;

	/*
	 * The vm_get_flags() and vm_unmap() are supposed to detect or handle
	 * overflow directly or indirectly. However, since this function is an
	 * API function it's worth having an extra guard here. If nothing else,
	 * to increase code clarity.
	 */
	if (ADD_OVERFLOW(va, sz, &end_va)) {
		res = TEE_ERROR_BAD_PARAMETERS;
		goto fin;
	}

	// res = vm_get_flags(uctx, va, sz, &vm_flags);
	// if (res)
	// 	goto fin;
	// if (vm_flags & VM_FLAG_PERMANENT) {
	// 	res = TEE_ERROR_ACCESS_DENIED;
	// 	goto fin;
	// }

	res = vm_unmap(uctx, va, sz);

	*ret_status = 0;

fin:
	TRACE_RET2(res, ret_status);
}
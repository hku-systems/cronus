/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

// This is written according to tee_fs_rpc.c

#include <assert.h>
#include <kernel/tee_misc.h>
#include <kernel/thread.h>
#include <mm/core_memprot.h>
#include <optee_rpc_cmd.h>
#include <stdlib.h>
#include <string_ext.h>
#include <string.h>
#include <tee/tee_posix_fs.h>
#include <tee/tee_fs.h>
#include <tee/tee_fs_rpc.h>
#include <tee/tee_pobj.h>
#include <tee/tee_svc_storage.h>
#include <trace.h>
#include <unistd.h>
#include <util.h>

struct ree_fs_rpc_operation {
    struct thread_param params[THREAD_RPC_MAX_NUM_PARAMS];
    size_t num_params;
};

static TEE_Result operation_commit(struct ree_fs_rpc_operation *op)
{
	return thread_rpc_cmd(OPTEE_RPC_CMD_FS, op->num_params, op->params);
}

TEE_Result ree_rpc_open(const char *filename, int flag, int mode, int *fd)
{
	struct mobj *mobj;
	TEE_Result res;
	void *va;

	va = thread_rpc_shm_cache_alloc(THREAD_SHM_CACHE_USER_FS,
					THREAD_SHM_TYPE_APPLICATION,
					TEE_FS_NAME_MAX, &mobj);
	if (!va)
		return TEE_ERROR_OUT_OF_MEMORY;

	if (strlen(filename) >= TEE_FS_NAME_MAX) {
	  return TEE_ERROR_BAD_PARAMETERS;
	}

	strcpy(va, filename);

	struct ree_fs_rpc_operation op = {
		.num_params = 3, .params = {
			[0] = THREAD_PARAM_VALUE(IN, OPTEE_RPC_FS_OPEN, 0, 0),
			[1] = THREAD_PARAM_MEMREF(IN, mobj, 0, TEE_FS_NAME_MAX),
			[2] = THREAD_PARAM_VALUE(OUT, 0, 0, 0),
	} };

	res = operation_commit(&op);
	if (res == TEE_SUCCESS)
		*fd = op.params[2].u.value.a;

	return res;
}

TEE_Result ree_rpc_close(int fd)
{
	struct ree_fs_rpc_operation op = {
		.num_params = 1, .params = {
			[0] = THREAD_PARAM_VALUE(IN, OPTEE_RPC_FS_CLOSE, fd, 0),
		},
	};

	return operation_commit(&op);
}

static TEE_Result ree_rpc_read_init(struct ree_fs_rpc_operation *op,
				int fd, tee_fs_off_t offset,
				size_t data_len, void **out_data)
{
	struct mobj *mobj;
	uint8_t *va;

	if (offset < 0)
		return TEE_ERROR_BAD_PARAMETERS;

	va = thread_rpc_shm_cache_alloc(THREAD_SHM_CACHE_USER_FS,
					THREAD_SHM_TYPE_APPLICATION,
					data_len, &mobj);
	if (!va)
		return TEE_ERROR_OUT_OF_MEMORY;

	*op = (struct ree_fs_rpc_operation){
		.num_params = 2, .params = {
			[0] = THREAD_PARAM_VALUE(IN, OPTEE_RPC_FS_READ, fd,
						 offset),
			[1] = THREAD_PARAM_MEMREF(OUT, mobj, 0, data_len),
		},
	};

	*out_data = va;

	return TEE_SUCCESS;
}

static TEE_Result ree_rpc_read_final(struct ree_fs_rpc_operation *op,
				 size_t *data_len)
{
	TEE_Result res = operation_commit(op);

	if (res == TEE_SUCCESS)
		*data_len = op->params[1].u.memref.size;
	return res;
}

static TEE_Result ree_rpc_write_init(struct ree_fs_rpc_operation *op,
				 int fd, tee_fs_off_t offset,
				 size_t data_len, void **data)
{
	struct mobj *mobj;
	uint8_t *va;

	if (offset < 0)
		return TEE_ERROR_BAD_PARAMETERS;

	va = thread_rpc_shm_cache_alloc(THREAD_SHM_CACHE_USER_FS,
					THREAD_SHM_TYPE_APPLICATION,
					data_len, &mobj);
	if (!va)
		return TEE_ERROR_OUT_OF_MEMORY;

	*op = (struct ree_fs_rpc_operation){
		.num_params = 2, .params = {
			[0] = THREAD_PARAM_VALUE(IN, OPTEE_RPC_FS_WRITE, fd,
						 offset),
			[1] = THREAD_PARAM_MEMREF(IN, mobj, 0, data_len),
		},
	};

	*data = va;

	return TEE_SUCCESS;
}

static TEE_Result ree_rpc_write_final(struct ree_fs_rpc_operation *op)
{
	return operation_commit(op);
}

static TEE_Result ree_rpc_truncate(int fd, size_t len)
{
	struct ree_fs_rpc_operation op = {
		.num_params = 1, .params = {
			[0] = THREAD_PARAM_VALUE(IN, OPTEE_RPC_FS_TRUNCATE, fd,
						 len),
		}
	};

	return operation_commit(&op);
}

TEE_Result ree_rpc_read(int fd, void *buf, size_t count, size_t off, ssize_t *rsize) {
  struct ree_fs_rpc_operation op;
  TEE_Result ret = TEE_SUCCESS;
  void *share_buf;
  ret = ree_rpc_read_init(&op, fd, off, count, (void**)&share_buf);
  if (ret) {
	EMSG("RPC read -> %lx", ret);
    return TEE_ERROR_BAD_PARAMETERS;
  }

  ret = ree_rpc_read_final(&op, rsize);

  if (ret) {
	EMSG("RPC read fin -> %lx", ret);
    return TEE_ERROR_BAD_PARAMETERS;
  }

  memcpy(buf, share_buf, *rsize);
  return ret;
}

TEE_Result ree_rpc_write(int fd, const void *buf, size_t count, size_t off, ssize_t *wsize) {
  struct ree_fs_rpc_operation op;
  TEE_Result ret;
  void *share_buf;
  ret = ree_rpc_write_init(&op, fd, off, count, (void**)&share_buf);
  if (ret) {
	EMSG("RPC write -> %lx", ret);
    return TEE_ERROR_BAD_PARAMETERS;
  }

  memcpy(share_buf, buf, count);

  ret = ree_rpc_write_final(&op);
  if (ret) {
	EMSG("RPC write fin -> %lx", ret);
    return TEE_ERROR_BAD_PARAMETERS;
  }

  *wsize = count;
  return ret;
}

const struct posix_fs posix_direct_ree_fs = {
    .open = ree_rpc_open,
    .read = ree_rpc_read,
    .write = ree_rpc_write,
    .close = ree_rpc_close
};



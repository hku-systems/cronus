/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <tee_internal_api_extensions.h>
#include <utee_syscalls.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/uio.h>
#include <pthread.h>
#include <pthread_arch.h>

#include "libos.h"

#define BIT32(nr)		(UINT32_C(1) << (nr))
#define BIT64(nr)		(UINT64_C(1) << (nr))

#define TEE_MAP_FLAG_SHAREABLE	BIT32(0)
#define TEE_MAP_FLAG_WRITEABLE	BIT32(1)
#define TEE_MAP_FLAG_EXECUTABLE	BIT32(2)

#define SMALL_PAGE_SIZE		0x00001000

#define FD_START 3
#define FD_MAX 20
int fd_used[20];

int file_fd_idx = FD_START;

struct fd_struct {
    int cur;
    int handle;
} fd_struct_mapping [FD_MAX];

struct stat {
	unsigned long	st_dev;		/* Device.  */
	unsigned long	st_ino;		/* File serial number.  */
	unsigned int	st_mode;	/* File mode.  */
	unsigned int	st_nlink;	/* Link count.  */
	unsigned int	st_uid;		/* User ID of the file's owner.  */
	unsigned int	st_gid;		/* Group ID of the file's group. */
	unsigned long	st_rdev;	/* Device number, if device.  */
	unsigned long	__pad1;
	long		st_size;	/* Size of file, in bytes.  */
	int		st_blksize;	/* Optimal block size for I/O.  */
	int		__pad2;
	long		st_blocks;	/* Number 512-byte blocks allocated. */
	long		st_atime;	/* Time of last access.  */
	unsigned long	st_atime_nsec;
	long		st_mtime;	/* Time of last modification.  */
	unsigned long	st_mtime_nsec;
	long		st_ctime;	/* Time of last status change.  */
	unsigned long	st_ctime_nsec;
	unsigned int	__unused4;
	unsigned int	__unused5;
};

static int next_fd() {
  int ret = -1;
  if (file_fd_idx >= FD_MAX) {
    file_fd_idx = FD_START;
  }
  for (int i = file_fd_idx;i < FD_MAX;i++) {
    if (!fd_used[i]) {
      fd_used[i] = 1;
      ret = i;
      file_fd_idx = i + 1;
      return ret;
    }
  }
  for (int i = FD_START;i < file_fd_idx;i++) {
    if (!fd_used[i]) {
      fd_used[i] = 1;
      ret = i;
      file_fd_idx = i + 1;
      return ret;
    }
  }
  return ret;
}

int tee_fs_openat(int fd, const char* filename, int flag, int mode) {
  int handle;
  TEE_UUID uuid;
  TEE_Result ret = _utee_fs_open(filename, flag, mode, &handle);
  int ret_fd = -1;
  uintptr_t va;
  int flags = TEE_MAP_FLAG_SHAREABLE;
  if (ret) {
    errno = EACCES;
    goto no_error;
  }

  ret_fd = next_fd();
  if (ret_fd == -1) {
    libos_err("no avaiable fd\n");
    return -1;
  }
  fd_struct_mapping[ret_fd].cur = 0;
  fd_struct_mapping[ret_fd].handle = handle;

no_error:
  libos_debug("open file (%s, %lx, %lx) -> %lx\n", filename, flag, mode, ret_fd);
  return ret_fd;
}

int tee_fs_read(int fd, void *buf, size_t count) {
  struct fd_struct *fds = &fd_struct_mapping[fd];
  TEE_Result ret;
  ssize_t rsize;

  if (fd == STDIN_FILENO)
    return -1;

  ret = _utee_fs_read(fds->handle, buf, count, fds->cur, &rsize);
  if (ret) {
    // handle
    return -1;
  }  
  fds->cur += rsize;
  return rsize;
}

int tee_fs_readv(int fd, const struct iovec *iov, int iovcnt) {
  int total_size = 0;
  int rsize = 0;
  int has_error = 0;
  for (int i = 0;i < iovcnt;i++) {
    if (iov[i].iov_len > 0) {
      rsize = tee_fs_read(fd, iov[i].iov_base, iov[i].iov_len);
      if (rsize > 0)
        total_size += rsize;
      else
        has_error = 1;
    }
  }
  if (has_error) {
    total_size = -1;
    libos_err_ret(total_size);
  }
  
  libos_debug_ret(total_size);
}

int tee_fs_write(int fd, const void *buf, size_t count) {
    struct fd_struct *fds = &fd_struct_mapping[fd];
    TEE_Result ret;
    ssize_t wsize;

    if (fd == STDOUT_FILENO || fd == STDERR_FILENO) {
      _utee_log(buf, count);
      wsize = count;
    } else {
      ret = _utee_fs_write(fds->handle, buf, count, fds->cur, &wsize);
      if (ret) {
        // handle
        return -1;
      }
      fds->cur += wsize;
    }

    return wsize;
}

int tee_fs_writev(int fd, const struct iovec *iov, int iovcnt) {
    int total_size = 0;
    char buf[100];
    int wsize = 0;
    int has_error = 0;

    for (int i = 0;i < iovcnt; i++) {
        total_size += iov[i].iov_len;
        if (iov[i].iov_len) {
            wsize = tee_fs_write(fd, iov[i].iov_base, iov[i].iov_len);
            if (wsize > 0)
              total_size += wsize;
            else
              has_error = 1;
        }
    }

    if (has_error) total_size = -1;

    return total_size;
}

void tee_fs_close(int fd) {
  struct fd_struct *fds = &fd_struct_mapping[fd];
  TEE_Result ret;
  ret = _utee_fs_close(fds->handle);
  if (ret) {
    // errno = ?;
    libos_err("close error");
  }
  fd_used[fd] = 0;
  libos_debug("close fd %d\n", fd);
}

int tee_dev_newfstatat(int dirfd, const char *pathname, struct stat *statbuf, int flags) {
  if (strcmp(pathname, "/dev/dri/card0") == 0) {
    return 0;
  }
  return -1;
}

int tee_fs_newfstatat(int dirfd, const char *pathname, struct stat *statbuf, int flags) {
  // TODO: fs
  return tee_dev_newfstatat(dirfd, pathname, statbuf, flags);
}

int tee_fs_fstat(int dirfd, struct stat *statbuf) {
  int handle = fd_struct_mapping[dirfd].handle;
  statbuf->st_dev = handle;
}

int tee_dev_ioctl(int fd, unsigned long cmd, void *args) {
    struct fd_struct *fds = &fd_struct_mapping[fd];
    TEE_Result ret = TEE_ERROR_BAD_PARAMETERS;
    int ret_val;

    switch (fd) {
      case STDIN_FILENO:  break;
      case STDOUT_FILENO: break;
        break;
      default: ret = _utee_dev_ioctl(fds->handle, cmd, args, &ret_val); break;
    }

    if (ret) {
        ret_val = -1;
        libos_err_ret2(ret_val, ret);
    }
    libos_debug_ret(ret_val);
}
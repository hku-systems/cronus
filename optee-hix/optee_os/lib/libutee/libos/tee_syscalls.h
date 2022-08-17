//
// Created by jianyu on 7/29/2021.
//

#ifndef OPTEE_OS_TEE_H
#define OPTEE_OS_TEE_H

extern struct fd_struct {
    int cur;
    int handle;
} fd_struct_mapping [];

int tee_fs_openat(int fd, const char* filename, int flag, int mode);
int tee_fs_read(int fd, void *buf, size_t count);
int tee_fs_readv(int fd, const struct iovec *iov, int iovcnt);
int tee_fs_write(int fd, const void *buf, size_t count);
int tee_fs_writev(int fd, const struct iovec *iov, int iovcnt);
void tee_fs_close(int fd);

// TODO: check stat
int tee_fs_newfstatat(int dirfd, const char *pathname, struct stat *statbuf, int flags);
int tee_fs_fstat(int dirfd, struct stat * statbuf);

int tee_dev_ioctl(int fd, unsigned long cmd, void *args);

void *tee_mmap(void *addr, size_t length, int prot, int flags,
                int fd, off_t offset);
void *tee_mremap(void *old_address, size_t old_size,
            size_t new_size, int flags);
int tee_munmap(void *addr, size_t len);

// user
int tee_user_gettid();
int tee_user_getpid();

// sched
int tee_sched_yield();

// time
int tee_clock_gettime(int clk, struct timespec *ts);
int tee_clock_nanosleep(int clockid, int flags,
                           const struct timespec *request,
                           struct timespec *remain);
int tee_nanosleep(const struct timespec *request,
                    struct timespec *remain);
#endif //OPTEE_OS_TEE_H

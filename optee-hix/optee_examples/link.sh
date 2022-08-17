rm ./rodinia_nw/ta/gdev-cuda
rm ./mnist_cuda/ta/gdev-cuda
rm ./hello_world_cuda/ta/gdev-cuda
rm ./rodinia_nn/ta/gdev-cuda
rm ./rodinia_lud/ta/gdev-cuda
rm ./rodinia_cuda/ta/gdev-cuda
rm ./rodinia_bp/ta/gdev-cuda
rm ./rodinia_pf/ta/gdev-cuda
rm ./rodinia_gs/ta/gdev-cuda
rm ./rodinia_srad/ta/gdev-cuda
rm ./rodinia_hs/ta/gdev-cuda

ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev rodinia_nw/ta/gdev-cuda
ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev mnist_cuda/ta/gdev-cuda
ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev hello_world_cuda/ta/gdev-cuda
ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev rodinia_nn/ta/gdev-cuda
ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev rodinia_lud/ta/gdev-cuda
ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev rodinia_cuda/ta/gdev-cuda
ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev rodinia_bp/ta/gdev-cuda
ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev rodinia_pf/ta/gdev-cuda
ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev rodinia_gs/ta/gdev-cuda
ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev rodinia_srad/ta/gdev-cuda
ln -s /home/jianyu/optee-naive/optee_os/lib/libgdev/gdev rodinia_hs/ta/gdev-cuda
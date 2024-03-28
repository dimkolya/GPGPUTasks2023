#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define VALUES_PER_WORK_ITEM 64
#define WORK_GROUP_SIZE 128

__kernel void sum_global_atomic(__global unsigned int *as,
                                __global unsigned int *result,
                                int n) {
    int index = get_global_id(0);
    if (index >= n) return;
    atomic_add(result, as[index]);
}

__kernel void sum_loop(__global unsigned int *as,
                       __global unsigned int *result,
                       int n) {
    int index = get_global_id(0);
    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        int id = index * VALUES_PER_WORK_ITEM + i;
        if (id < n) {
            sum += as[id];
        }
    }
    atomic_add(result, sum);
}

__kernel void sum_loop_coalesced(__global unsigned int *as,
                       __global unsigned int *result,
                       int n) {
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int groupId = get_group_id(0);

    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        int id = groupId * localSize * VALUES_PER_WORK_ITEM + localSize * i + localId;
        if (id < n) {
            sum += as[id];
        }
    }
    atomic_add(result, sum);
}

__kernel void sum_local_memory(__global unsigned int *as,
                               __global unsigned int *result,
                               int n) {
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE];
    local_as[localId] = as[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_as[i];
        }
        atomic_add(result, sum);
    }
}

__kernel void sum_tree(__global unsigned int *as,
                       __global unsigned int *result,
                       int n) {
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    __local unsigned int local_as[WORK_GROUP_SIZE];
    local_as[localId] = as[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = WORK_GROUP_SIZE; i > 1; i /= 2) {
        if (2 * localId < i) {
            local_as[localId] += local_as[localId + i / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        atomic_add(result, local_as[localId]);
    }
}
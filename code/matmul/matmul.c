#include "rpi.h"
#include <stddef.h>
#include <string.h>
#include "matmul.h"
#include "mailbox.h"

//TODO: SWAP THESE

#include "matmulshader.h"


int gpu_prepare(
	volatile struct GPU **gpu)
{
	uint32_t handle, vc;
	volatile struct GPU *ptr;

	/* TURN ON THE QPU */
	if (qpu_enable(1))
		return -2;

	/* ALLOCATE MEMORY FOR THE STRUCT QPU */
	handle = mem_alloc(sizeof(struct GPU), 4096, GPU_MEM_FLG);
	if (!handle)
	{
		qpu_enable(0);
		return -3;
	}

	/* CLAIM THE BUS ADDRESS OF THE MEMORY */
	vc = mem_lock(handle);

	/* USE THE Pi 0 GPU OFFSET TO GET AN ADDRESS WE CAN READ/WRITE ON THE CPU */
	ptr = (volatile struct GPU *)(vc - 0x40000000);
	if (ptr == NULL)
	{
		mem_free(handle);
		mem_unlock(handle);
		qpu_enable(0);
		return -4;
	}

	/* INITIALIZE STRUCT QPU FIELDS*/
	ptr->handle = handle;
	ptr->mail[0] = GPU_BASE + (uint32_t)&ptr->code;
	ptr->mail[1] = GPU_BASE + (uint32_t)&ptr->unif;

	*gpu = ptr;
	return 0;
}

/* SEND THE CODE AND UNIFS TO THE GPU (see docs p. 89-91)*/
uint32_t gpu_execute(volatile struct GPU *gpu, struct profiler_data *profiler_data)
{
	return gpu_fft_base_exec_direct(
		(uint32_t)gpu->mail[0],
		(uint32_t *)gpu->unif_ptr,
		NUM_QPUS,
		profiler_data
	);
}

/* RELEASE MEMORY AND TURN OFF QPU */
void gpu_release(volatile struct GPU *gpu)
{
	uint32_t handle = gpu->handle;
	mem_unlock(handle);
	mem_free(handle);
	qpu_enable(0);
}

// TODO: SET UP THE UNIFORMS FOR YOUR KERNEL 
void gpu_init(volatile struct GPU **gpu, int n)
{
	int ret = gpu_prepare(gpu);
	if (ret < 0)
		return;

	volatile struct GPU *ptr = *gpu;
	memcpy((void *)ptr->code, matmulshader, sizeof ptr->code);

	for (int i = 0; i < NUM_QPUS; i++) {
		ptr->unif[i][0] = N;
		ptr->unif[i][1] = GPU_BASE + (uint32_t)&ptr->A;
		ptr->unif[i][2] = GPU_BASE + (uint32_t)&ptr->B;
		ptr->unif[i][3] = GPU_BASE + (uint32_t)&ptr->C;
		ptr->unif[i][4] = NUM_QPUS;
		ptr->unif[i][5] = i;
		ptr->unif_ptr[i] = GPU_BASE + (uint32_t)&ptr->unif[i];
	}
}

int gpu_exec(volatile struct GPU *gpu, struct profiler_data *profiler_data)
{
	int start_time = timer_get_usec();
	int iret = gpu_execute(gpu, profiler_data);
	int end_time = timer_get_usec();

	return end_time - start_time;
}
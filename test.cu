#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
using namespace std;

__device__ unsigned int reduce_sum(unsigned int in)
{
    extern __shared__ unsigned int sdata[];

    // Perform first level of reduction:
    // - Write to shared memory
    unsigned int ltid = threadIdx.x;

    sdata[ltid] = in;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2 ; s > 0 ; s >>= 1)
    {
        if (ltid < s)
        {
            sdata[ltid] += sdata[ltid + s];
        }

        __syncthreads();
    }

    return sdata[0];
}

__global__ void mykernel(int vectorsize, int *count, double *rands) 
{
	int id = blockIdx.x *blockDim.x + threadIdx.x;

	int step = gridDim.x * blockDim.x;

	const double *rand1 = rands + id;
    const double *rand2 = rand1 + vectorsize;

    int tempcount = 0;

	for (int i = 0; i < vectorsize; i += step, rand1 +=step, rand2 += step)
	{
		double x = *rand1;
		double y = *rand2;
		if(((x*x)+(y*y)) < 1 )
			tempcount++;
	}
	tempcount = reduce_sum(tempcount);

	if (threadIdx.x == 0)
    {
        count[blockIdx.x] = tempcount;
    }
}

int main(void) 
{
	auto t_start = std::chrono::high_resolution_clock::now();

	double vectorsize = 67107840;
	// cin >> vectorsize;
	int blocksize = 1024;

	int gridsize = ceil(vectorsize/blocksize);
	size_t sharedmemsize = blocksize * sizeof(double);

    //RANDOM NUMBER GENERATION//
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_MTGP32);
    double *rands = 0;
    cudaMalloc((void **)&rands, 2* vectorsize * sizeof(double));
    curandSetPseudoRandomGeneratorSeed(prng, 1337);
    curandGenerateUniformDouble(prng, (double *)rands, 2 * vectorsize);
    curandDestroyGenerator(prng);
    //RANDOM NUMBER GENERATION//

	int *count, *cuda_count;	

	count = (int *)malloc (gridsize * sizeof(int));
	cudaMalloc((void **)&cuda_count, gridsize *sizeof(int));

		mykernel <<<gridsize, blocksize, sharedmemsize>>>(vectorsize, cuda_count, rands);

	if (cudaMemcpy (count, cuda_count, gridsize *sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
		printf("failed to cpy back\n");

	int totalcount = 0;
	for (int i = 0; i < gridsize; i ++)
	{
		totalcount += count[i];
	}

	printf("count = %d\n", totalcount);
	float ratio = totalcount / vectorsize;
	printf("pi =  %.15f \n", (ratio * 4));

	cudaFree(cuda_count);

	auto t_end = std::chrono::high_resolution_clock::now();

    printf("duration: %f\n", (std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000));

	return 0;
}

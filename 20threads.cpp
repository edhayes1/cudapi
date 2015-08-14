#include <future>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <ctime>
#include <chrono>

using namespace std;

double vectorsize = 67107840;
int numthreads = 20;

void randomints(float* a, int n, int add)
{
	random_device rd;
    mt19937_64 mt(rd());
    uniform_real_distribution<double> dist(0,1);

    for (int i =0; i < n; ++i)
    {
    	a[i] = dist(mt);
    }
}


// void randomints(float* a, int n, int add)
// {
// 	unsigned int seed = 1337 + add;
// 	int i;
// 	for (i = 0; i < n; ++i)
// 	{
// 	  	a[i] = (float)rand_r(&seed) / (float)RAND_MAX;
// 	}
// }

double dotask(int seedadd)
{
	double tempcount = 0;
	float *a, *b;
	int n = ceil(vectorsize/numthreads);
	//printf("hi %d", vectorsize);
	int size = n * sizeof(float);
	a = (float *)malloc(size);
	b = (float *)malloc(size);
	randomints(a, n, seedadd);
	randomints(b, n, seedadd);

	//cout << "rand = " << RAND_MAX << endl;
	for(int i = 0; i < n; i++)
	{
		if(sqrt((a[i]*a[i])+(b[i]*b[i])) < 1)
		{
			tempcount++;
		}
	}
	//cout << "n = " << n << endl; 

	return tempcount;
}

int main()
{
	auto t_start = std::chrono::high_resolution_clock::now();

	double count = 0;
	vector<future<double>> futures;
	for (int i = 0; i < numthreads; ++i)
	{
		futures.push_back (std::async(std::launch::async,dotask, i));
	}
	
	for (int i = 0; i < numthreads; ++i)
	{
		count = count + futures[i].get();
	}
	vectorsize = numthreads * ceil(vectorsize/numthreads);
	printf ("count %f\n", count);

	printf("pi = %.15f\n", 4*(count/vectorsize));

    auto t_end = std::chrono::high_resolution_clock::now();

    printf("duration: %f\n", (std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000));

	return 0;
}
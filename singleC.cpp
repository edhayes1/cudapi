#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>
#include <random> 
#include <chrono>

using namespace std;

void randomints(double* a, int n)
{
	random_device rd;
    mt19937_64 mt(rd());
    uniform_real_distribution<double> dist(0,1);

    for (int i =0; i < n; ++i)
    {
    	a[i] = dist(mt);
    }
}



int main()
{
	auto t_start = std::chrono::high_resolution_clock::now();
	double vectorsize = 67107840;
	//cin >> vectorsize;
	double *a, *b;
	double *d_a, *d_b;//, *d_c;
	int size = vectorsize * sizeof(double);

	a = (double *)malloc(size); randomints(a, vectorsize);
	b = (double *)malloc(size); randomints(b, vectorsize);


	int count = 0;
	for(int i = 0; i < vectorsize; i++)
	{
		if((a[i]*a[i])+(b[i]*b[i]) < 1)
		{
			count++;
		}
	}
	double ratio = (double)count / (double)vectorsize;
	printf("pi = %.15f\n", ratio *4);
	auto t_end = std::chrono::high_resolution_clock::now();

    printf("duration: %.15f\n", (std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000));

	return 0;

}
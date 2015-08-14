#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

vector<double> randomints(vector<double> &a, double n)
{
	int i;
	for (i = 0; i < n; ++i)
	{
	  	a[i] = (double)rand() / (double)RAND_MAX;
	}
	cout << "created random ints " << endl;
	return a;
}


int main()
{
	double vectorsize = 50000000;
	vector<double> a (vectorsize);
	vector<double> b (vectorsize);

	a = randomints(a, vectorsize);
	b = randomints(b, vectorsize);

	cout << "done random ints" << endl;
	int count = 0;
	for(int i = 0; i < a.size(); i++)
	{
		if(sqrt((a[i]*a[i])+(b[i]*b[i])) < 1)
		{
			count++;
		}
	}
	double ratio = count / vectorsize;
	cout << "pi =  " << ratio*4 << endl;
	return 0;
}
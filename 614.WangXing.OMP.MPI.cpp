#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include <fstream>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <string>
#include "mpi.h"
using namespace std;
using namespace chrono;

const static double Ax = 0.0, Ay = 0.0, Bx = 3.0, By = 0.0, Cx = 2.0, Cy = 3.0, Dx = 0.0, Dy = 3.0;
const static double A1 = -1.0, B1 = 4.0, A2 = -1.0, B2 = 4.0;
int M, N, Num_threads, M0 = 0, N0 = 0, Top = 1, Bottom = 1, Left = 1, Right = 1;
int Rank, Size, GlobalI, GlobalJ, subM, subN, subI, subJ;
double h1, h2, eps, seps;
vector<double> X, Y;

int PiontClassific(double x, double y)
{
	if (y < 3 && y > 0 && y < 3 * x + 9 && y < -3 * x + 9)
		
		return 1;
	else if (y == 0 && x >= -3 && x <= 3)
		
		return 0;
	else if (y == 3 && x >= -2 && x <= 2)
		
		return 0;
	else if (y == 3 * x + 9 && y < 3 && y > 0)
		
		return 0;
	else if (y == -3 * x + 9 && y < 3 && y > 0)
		
		return 0;
	else
		
		return -1;

}


double a_ij(const int i, const int j)
{
	double xi_minus = X[i] - 0.5 * h1, yj_minus = Y[j] - 0.5 * h2, yj_plus = Y[j] + 0.5 * h2;
	double l_ij = 0;
	if (PiontClassific(xi_minus, yj_plus) == 1 && PiontClassific(xi_minus, yj_minus) == 1)
	{
		return 1.0;
	}
	else if (PiontClassific(xi_minus, yj_plus) == 1 && PiontClassific(xi_minus, yj_minus) != 1)
	{
		l_ij = yj_plus;
		return l_ij / h2 + (1 - l_ij / h2) / seps;
	}
	else if (PiontClassific(xi_minus, yj_plus) != 1 && PiontClassific(xi_minus, yj_minus) == 1)
	{
		if (xi_minus >= -3 && xi_minus < -2)
		{
			l_ij = 3 * xi_minus + 9 - yj_minus;
		}
		else if (xi_minus >= -2 && xi_minus <= -2)
		{
			l_ij = 3 - yj_minus;
		}
		else if (xi_minus > 2 && xi_minus <= 3)
		{
			l_ij = -3 * xi_minus + 9 - yj_minus;
		}
		return l_ij / h2 + (1 - l_ij / h2) / seps;
	}
	else
	{
		if (xi_minus >= -3 && xi_minus < -2)
		{
			if (yj_minus >= 3 * xi_minus + 9 || yj_plus <= 0)
			{
				return 1.0 / seps;
			}
			else
			{
				l_ij = 3 * xi_minus + 9;
			}
		}
		else if (xi_minus >= -2 && xi_minus <= -2)
		{
			if (yj_minus >= 3 || yj_plus <= 0)
			{
				return 1.0 / seps;
			}
			else
			{
				l_ij = 3;
			}
		}
		else if (xi_minus > 2 && xi_minus <= 3)
		{
			if (yj_minus >= -3 * xi_minus + 9 || yj_plus <= 0)
			{
				return 1.0 / seps;
			}
			else
			{
				l_ij = -3 * xi_minus + 9;
			}
		}
		else
		{
			return 1.0 / seps;
		}
		return l_ij / h2 + (1 - l_ij / h2) / seps;
	}
}

double b_ij(const int i, const int j)
{
	double yj_minus = Y[j] - 0.5 * h2, xi_minus = X[i] - 0.5 * h1, xi_plus = X[i] + 0.5 * h1;
	double l_ij = 0;
	if ((yj_minus >= -1 && yj_minus <= 0) || (yj_minus >= 3 && yj_minus <= 4))
	{
		return 1.0 / seps;
	}
	else if (PiontClassific(xi_minus, yj_minus) == 1 && PiontClassific(xi_plus, yj_minus) == 1)
	{
		return 1.0;
	}
	else if (PiontClassific(xi_minus, yj_minus) != 1 && PiontClassific(xi_plus, yj_minus) == 1)
	{
		l_ij = xi_plus - (yj_minus - 9) / 3;
		return l_ij / h1 + (1 - l_ij / h1) / seps;
	}
	else if (PiontClassific(xi_minus, yj_minus) == 1 && PiontClassific(xi_plus, yj_minus) != 1)
	{
		l_ij = -(yj_minus - 9) / 3 - xi_minus;
		return l_ij / h1 + (1 - l_ij / h1) / seps;
	}
	else
	{
		if (xi_plus <= (yj_minus - 9) / 3 || xi_minus >= -(yj_minus - 9) / 3)
		{
			return 1.0 / seps;
		}
		else
		{
			l_ij = -(yj_minus - 9) / 3 - (yj_minus - 9) / 3;
			return l_ij / h1 + (1 - l_ij / h1) / seps;
		}
	}
}

double F_ij(const int i, const int j)
{
	double xi_minus = X[i] - 0.5 * h1, xi_plus = X[i] + 0.5 * h1, yj_minus = Y[j] - 0.5 * h2, yj_plus = Y[j] + 0.5 * h2;
	if (yj_minus >= 3 || yj_plus <= 0)
	{
		return 0;
	}
	if (PiontClassific(xi_minus, yj_minus) == 1 && PiontClassific(xi_minus, yj_plus) == 1 && PiontClassific(xi_plus, yj_minus) == 1 && PiontClassific(xi_plus, yj_plus) == 1)
	{
		return 1.0; 
	}
	else if (PiontClassific(xi_minus, yj_minus) == 1 && PiontClassific(xi_minus, yj_plus) != 1 && PiontClassific(xi_plus, yj_minus) != 1 && PiontClassific(xi_plus, yj_plus) != 1)
	{
		if (yj_plus < 3)
		{
			return 0.5 * (-(yj_minus - 9) / 3 - xi_minus) * (-3 * xi_minus + 9 - yj_minus);
		}
		else
		{
			return 0.5 * ((2 - xi_minus) + (-(yj_minus - 9) / 3 - xi_minus)) * (3 - yj_minus);
		}
	}
	else if (PiontClassific(xi_minus, yj_minus) != 1 && PiontClassific(xi_minus, yj_plus) != 1 && PiontClassific(xi_plus, yj_minus) == 1 && PiontClassific(xi_plus, yj_plus) != 1)
	{
		if (yj_plus < 3)
		{
			return 0.5 * (xi_plus - (yj_minus - 9) / 3) * (3 * xi_plus + 9 - yj_minus);
		}
		else
		{
			return 0.5 * ((xi_plus + 2) + (xi_plus - (yj_minus - 9) / 3)) * (3 - yj_minus);
		}
	}
	else if (PiontClassific(xi_minus, yj_minus) != 1 && PiontClassific(xi_minus, yj_plus) == 1 && PiontClassific(xi_plus, yj_minus) != 1 && PiontClassific(xi_plus, yj_plus) != 1)
	{
		if (xi_plus > 3)
		{
			return 0.5 * ((-(yj_plus - 9) / 3 - xi_minus) + (3 - xi_minus)) * (yj_plus);
		}
		else
		{
			return h1 * h2 - h1 * (0 - yj_minus) - 0.5 * (xi_plus + (yj_plus - 9) / 3) * (yj_plus - (-3 * xi_plus + 9));
		}
	}
	else if (PiontClassific(xi_minus, yj_minus) != 1 && PiontClassific(xi_minus, yj_plus) != 1 && PiontClassific(xi_plus, yj_minus) != 1 && PiontClassific(xi_plus, yj_plus) == 1)
	{
		if (xi_minus < -3)
		{
			return 0.5 * ((xi_plus - (yj_plus - 9) / 3) + (xi_plus + 3)) * (yj_plus);
		}
		else
		{
			return h1 * h2 - h1 * (0 - yj_minus) - 0.5 * ((yj_plus - 9) / 3 - xi_minus) * (yj_plus - (3 * xi_minus + 9));
		}
	}
	else if (PiontClassific(xi_minus, yj_minus) == 1 && PiontClassific(xi_minus, yj_plus) == 1 && PiontClassific(xi_plus, yj_minus) != 1 && PiontClassific(xi_plus, yj_plus) != 1)
	{
		return 0.5 * ((-(yj_plus - 9) / 3 - xi_minus) + (-(yj_minus - 9) / 3 - xi_minus)) * h2;
	}
	else if (PiontClassific(xi_minus, yj_minus) != 1 && PiontClassific(xi_minus, yj_plus) != 1 && PiontClassific(xi_plus, yj_minus) == 1 && PiontClassific(xi_plus, yj_plus) == 1)
	{
		return 0.5 * ((xi_plus - (yj_plus - 9) / 3) + (xi_plus - (yj_minus - 9) / 3)) * h2;
	}
	else if (PiontClassific(xi_minus, yj_minus) == 1 && PiontClassific(xi_minus, yj_plus) != 1 && PiontClassific(xi_plus, yj_minus) == 1 && PiontClassific(xi_plus, yj_plus) != 1)
	{
		return h1 * (3 - yj_minus);
	}
	else if (PiontClassific(xi_minus, yj_minus) != 1 && PiontClassific(xi_minus, yj_plus) == 1 && PiontClassific(xi_plus, yj_minus) != 1 && PiontClassific(xi_plus, yj_plus) == 1)
	{
		return h1 * yj_plus;
	}
	else if (PiontClassific(xi_minus, yj_minus) == 1 && PiontClassific(xi_minus, yj_plus) == 1 && PiontClassific(xi_plus, yj_minus) == 1 && PiontClassific(xi_plus, yj_plus) != 1)
	{
		return h1 * h2 - 0.5 * (xi_plus + (yj_plus - 9) / 3) * (yj_plus - (-3 * xi_plus + 9));
	}
	else if (PiontClassific(xi_minus, yj_minus) == 1 && PiontClassific(xi_minus, yj_plus) != 1 && PiontClassific(xi_plus, yj_minus) == 1 && PiontClassific(xi_plus, yj_plus) == 1)
	{
		return h1 * h2 - 0.5 * ((yj_plus - 9) / 3 - xi_minus) * (yj_plus - (3 * xi_minus + 9));
	}
	else
	{
		if (xi_minus < 3 && xi_plus >= 3 && yj_minus <= 0 && yj_plus >= -3 * xi_minus + 9)
		{
			return 0.5 * (3 - xi_minus) * (-3 * xi_minus + 9);
		}
		else if (xi_minus <= -3 && xi_plus > -3 && yj_minus <= 0 && yj_plus >= 3 * xi_plus + 9)
		{
			return 0.5 * (xi_plus - 3) * (3 * xi_plus + 9);
		}
		else
		{
			return 0;
		}
	}
}

double NormMax(const vector<vector<double> >& array)
{
	double maxElement = fabs(array[0][0]);
	for (int i = 0; i < array.size(); i++)
	{
		for (int j = 0; j < array[i].size(); j++)
		{
			maxElement = max(array[i][j], maxElement);
		}
	}
	return maxElement;
}

void printMat(const vector<vector<double> >& matrix)
{
	cout << "Rank:" << Rank << endl;
	cout << "Matrix:" << subN << "*" << subM << endl;
	for (int j = subN - 1; j >= 0; j--)
	{
		for (int i = 0; i < subM; i++)
		{
			cout.width(9);
			cout.precision(3);
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
	return;
}

vector<vector<double> > Rk(const vector<vector<double> >& w, const int count)
{
	vector<vector<double> > result(subM, vector<double>(subN, 0.0));
	vector<double> top_w_send, bottom_w_send, left_w_send = w[0], right_w_send = w[subM - 1];
	top_w_send.resize(subM), bottom_w_send.resize(subM);
	vector<double> top_w, bottom_w, left_w, right_w;
	top_w.resize(subM), bottom_w.resize(subM), left_w.resize(subN), right_w.resize(subN);
	for (int i = 0; i < subM; i++)
	{
		top_w_send[i] = w[i][subN - 1];
		bottom_w_send[i] = w[i][0];
	}
	MPI_Request send_request, recv_request;
	
	if (Left)
	{
		MPI_Isend(left_w_send.data(), subN, MPI_DOUBLE, Rank - 1, count, MPI_COMM_WORLD, &send_request);
	}
	if (Right)
	{
		MPI_Isend(right_w_send.data(), subN, MPI_DOUBLE, Rank + 1, count, MPI_COMM_WORLD, &send_request);
	}
	if (Top)
	{
		MPI_Isend(top_w_send.data(), subM, MPI_DOUBLE, Rank + M0, count, MPI_COMM_WORLD, &send_request);
	}
	if (Bottom)
	{
		MPI_Isend(bottom_w_send.data(), subM, MPI_DOUBLE, Rank - M0, count, MPI_COMM_WORLD, &send_request);
	}
	
	#pragma omp parallel for collapse(2) num_threads(Num_threads)
	for (int i = 2; i < subM; i++)
	{
		for (int j = 2; j < subN; j++)
		{
			int global_i = i - 1 + subI;
			int global_j = j - 1 + subJ;
			double w_ij = w[i - 1][j - 1];
			double w_left = w[i - 2][j - 1];
			double w_right = w[i][j - 1];
			double w_up = w[i - 1][j];
			double w_down = w[i - 1][j - 2];
			result[i - 1][j - 1] = -(a_ij(global_i + 1, global_j) * (w_right - w_ij) / h1 - a_ij(global_i, global_j) * (w_ij - w_left) / h1) / h1 - (b_ij(global_i, global_j + 1) * (w_up - w_ij) / h2 - b_ij(global_i, global_j) * (w_ij - w_down) / h2) / h2 - F_ij(global_i, global_j);
		}
	}
	
	if (Left)
	{
		MPI_Irecv(left_w.data(), subN, MPI_DOUBLE, Rank - 1, count, MPI_COMM_WORLD, &recv_request);
		MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
	}
	if (Right)
	{
		MPI_Irecv(right_w.data(), subN, MPI_DOUBLE, Rank + 1, count, MPI_COMM_WORLD, &recv_request);
		MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

	}
	if (Top)
	{
		MPI_Irecv(top_w.data(), subM, MPI_DOUBLE, Rank + M0, count, MPI_COMM_WORLD, &recv_request);
		MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

	}
	if (Bottom)
	{
		MPI_Irecv(bottom_w.data(), subM, MPI_DOUBLE, Rank - M0, count, MPI_COMM_WORLD, &recv_request);
		MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
	}

	#pragma omp parallel for num_threads(Num_threads)
	for (int j = 1; j <= subN; j++)
	{
		//i == 1
		int l_global_i = subI;
		int l_global_j = j - 1 + subJ;
		double l_w_ij = w[0][j - 1];
		double l_w_left = l_global_i == 1 ? 0 : left_w[j - 1];
		double l_w_right = l_global_i == M - 1 ? 0 : w[1][j - 1];
		double l_w_up = l_global_j == N - 1 ? 0 : j == subN ? top_w[0] : w[0][j];
		double l_w_down = l_global_j == 1 ? 0 : j == 1 ? bottom_w[0] : w[0][j - 2];
		result[0][j - 1] = -(a_ij(l_global_i + 1, l_global_j) * (l_w_right - l_w_ij) / h1 - a_ij(l_global_i, l_global_j) * (l_w_ij - l_w_left) / h1) / h1 - (b_ij(l_global_i, l_global_j + 1) * (l_w_up - l_w_ij) / h2 - b_ij(l_global_i, l_global_j) * (l_w_ij - l_w_down) / h2) / h2 - F_ij(l_global_i, l_global_j);
	
		int r_global_i = subM - 1 + subI;
		int r_global_j = j - 1 + subJ;
		double r_w_ij = w[subM - 1][j - 1];
		double r_w_left = r_global_i == 1 ? 0 : w[subM - 2][j - 1];
		double r_w_right = r_global_i == M - 1 ? 0 : right_w[j - 1];
		double r_w_up = r_global_j == N - 1 ? 0 : j == subN ? top_w[subM - 1] : w[subM - 1][j];
		double r_w_down = r_global_j == 1 ? 0 : j == 1 ? bottom_w[subM - 1] : w[subM - 1][j - 2];
		result[subM - 1][j - 1] = -(a_ij(r_global_i + 1, r_global_j) * (r_w_right - r_w_ij) / h1 - a_ij(r_global_i, r_global_j) * (r_w_ij - r_w_left) / h1) / h1 - (b_ij(r_global_i, r_global_j + 1) * (r_w_up - r_w_ij) / h2 - b_ij(r_global_i, r_global_j) * (r_w_ij - r_w_down) / h2) / h2 - F_ij(r_global_i, r_global_j);
	}

	#pragma omp parallel for num_threads(Num_threads)
	for (int i = 1; i <= subM; i++)
	{
		
		int b_global_i = i - 1 + subI;
		int b_global_j = subJ;
		double b_w_ij = w[i - 1][0];
		double b_w_left = b_global_i == 1 ? 0 : i == 1 ? left_w[0] : w[i - 2][0];
		double b_w_right = b_global_i == M - 1 ? 0 : i == subM ? right_w[0] : w[i][0];
		double b_w_up = b_global_j == N - 1 ? 0 : w[i - 1][1];
		double b_w_down = b_global_j == 1 ? 0 : bottom_w[i - 1];
		result[i - 1][0] = -(a_ij(b_global_i + 1, b_global_j) * (b_w_right - b_w_ij) / h1 - a_ij(b_global_i, b_global_j) * (b_w_ij - b_w_left) / h1) / h1 - (b_ij(b_global_i, b_global_j + 1) * (b_w_up - b_w_ij) / h2 - b_ij(b_global_i, b_global_j) * (b_w_ij - b_w_down) / h2) / h2 - F_ij(b_global_i, b_global_j);
	
		int t_global_i = i - 1 + subI;
		int t_global_j = subN - 1 + subJ;
		double t_w_ij = w[i - 1][subN - 1];
		double t_w_left = t_global_i == 1 ? 0 : i == 1 ? left_w[subN - 1] : w[i - 2][subN - 1];
		double t_w_right = t_global_i == M - 1 ? 0 : i == subM ? right_w[subN - 1] : w[i][subN - 1];
		double t_w_up = t_global_j == N - 1 ? 0 : top_w[i - 1];
		double t_w_down = t_global_j == 1 ? 0 : w[i - 1][subN - 2];
		result[i - 1][subN - 1] = -(a_ij(t_global_i + 1, t_global_j) * (t_w_right - t_w_ij) / h1 - a_ij(t_global_i, t_global_j) * (t_w_ij - t_w_left) / h1) / h1 - (b_ij(t_global_i, t_global_j + 1) * (t_w_up - t_w_ij) / h2 - b_ij(t_global_i, t_global_j) * (t_w_ij - t_w_down) / h2) / h2 - F_ij(t_global_i, t_global_j);
	}
	return result;
}

vector<vector<double> > M_minus_TauRk(const vector<vector<double > >& wk, const vector<vector<double> >& rk, double& local_norm_squre, const int count)
{
	vector<vector<double> > result(rk.size(), vector<double>(rk[0].size(), 0.0));
	double local_numerator = 0.0, local_denominator = 0.0;
	vector<double> top_rk_send, bottom_rk_send, left_rk_send = rk[0], right_rk_send = rk[subM - 1];
	top_rk_send.resize(subM), bottom_rk_send.resize(subM);
	vector<double> top_rk, bottom_rk, left_rk, right_rk;
	top_rk.resize(subM), bottom_rk.resize(subM), left_rk.resize(subN), right_rk.resize(subN);
	for (int i = 0; i < subM; i++)
	{
		top_rk_send[i] = rk[i][subN - 1];
		bottom_rk_send[i] = rk[i][0];
	}
	MPI_Request send_request_left, send_request_right, send_request_top, send_request_bot, recv_request_left, recv_request_right, recv_request_top, recv_request_bot;

	if (Left)
	{
		MPI_Isend(left_rk_send.data(), subN, MPI_DOUBLE, Rank - 1, count, MPI_COMM_WORLD, &send_request_left);
	}
	if (Right)
	{
		MPI_Isend(right_rk_send.data(), subN, MPI_DOUBLE, Rank + 1, count, MPI_COMM_WORLD, &send_request_right);
	}
	if (Top)
	{
		MPI_Isend(top_rk_send.data(), subM, MPI_DOUBLE, Rank + M0, count, MPI_COMM_WORLD, &send_request_top);
	}
	if (Bottom)
	{
		MPI_Isend(bottom_rk_send.data(), subM, MPI_DOUBLE, Rank - M0, count, MPI_COMM_WORLD, &send_request_bot);
	}

	#pragma omp parallel for collapse(2) reduction(+ : local_numerator, local_denominator) num_threads(Num_threads)
	for (int i = 2; i < subM; i++)
	{
		for (int j = 2; j < subN; j++)
		{
			int global_i = i - 1 + subI;
			int global_j = j - 1 + subJ;
			double rk_ij = rk[i - 1][j - 1];
			double rk_left = rk[i - 2][j - 1];
			double rk_right = rk[i][j - 1];
			double rk_up = rk[i - 1][j];
			double rk_down = rk[i - 1][j - 2];
			double A_rk = -(a_ij(global_i + 1, global_j) * (rk_right - rk_ij) / h1 - a_ij(global_i, global_j) * (rk_ij - rk_left) / h1) / h1 - (b_ij(global_i, global_j + 1) * (rk_up - rk_ij) / h2 - b_ij(global_i, global_j) * (rk_ij - rk_down) / h2) / h2;
			local_numerator += A_rk * rk_ij;
			local_denominator += A_rk * A_rk;
		}
	}

	

	
	if (Left)
	{
		MPI_Wait(&send_request_left, MPI_STATUS_IGNORE);
		MPI_Irecv(left_rk.data(), subN, MPI_DOUBLE, Rank - 1, count, MPI_COMM_WORLD, &recv_request_left);
		MPI_Wait(&recv_request_left, MPI_STATUS_IGNORE);
	}
	if (Right)
	{
		MPI_Wait(&send_request_right, MPI_STATUS_IGNORE);
		MPI_Irecv(right_rk.data(), subN, MPI_DOUBLE, Rank + 1, count, MPI_COMM_WORLD, &recv_request_right);
		MPI_Wait(&recv_request_right, MPI_STATUS_IGNORE);
	}
	if (Top)
	{
		MPI_Wait(&send_request_top, MPI_STATUS_IGNORE);
		MPI_Irecv(top_rk.data(), subM, MPI_DOUBLE, Rank + M0, count, MPI_COMM_WORLD, &recv_request_top);
		MPI_Wait(&recv_request_top, MPI_STATUS_IGNORE);
	}
	if (Bottom)
	{
		MPI_Wait(&send_request_bot, MPI_STATUS_IGNORE);
		MPI_Irecv(bottom_rk.data(), subM, MPI_DOUBLE, Rank - M0, count, MPI_COMM_WORLD, &recv_request_bot);
		MPI_Wait(&recv_request_bot, MPI_STATUS_IGNORE);
	}
	//Residual calculation
	#pragma omp parallel for reduction(+ : local_numerator, local_denominator) num_threads(Num_threads)
	for (int j = 1; j <= subN; j++)
	{
		
		int l_global_i = subI;
		int l_global_j = j - 1 + subJ;
		double l_rk_ij = rk[0][j - 1];
		double l_rk_left = l_global_i == 1 ? 0 : left_rk[j - 1];
		double l_rk_right = l_global_i == M - 1 ? 0 : rk[1][j - 1];
		double l_rk_up = l_global_j == N - 1 ? 0 : j == subN ? top_rk[0] : rk[0][j];
		double l_rk_down = l_global_j == 1 ? 0 : j == 1 ? bottom_rk[0] : rk[0][j - 2];
		double l_A_rk = -(a_ij(l_global_i + 1, l_global_j) * (l_rk_right - l_rk_ij) / h1 - a_ij(l_global_i, l_global_j) * (l_rk_ij - l_rk_left) / h1) / h1 - (b_ij(l_global_i, l_global_j + 1) * (l_rk_up - l_rk_ij) / h2 - b_ij(l_global_i, l_global_j) * (l_rk_ij - l_rk_down) / h2) / h2;

		
		int r_global_i = subM - 1 + subI;
		int r_global_j = j - 1 + subJ;
		double r_rk_ij = rk[subM - 1][j - 1];
		double r_rk_left = r_global_i == 1 ? 0 : rk[subM - 2][j - 1];
		double r_rk_right = r_global_i == M - 1 ? 0 : right_rk[j - 1];
		double r_rk_up = r_global_j == N - 1 ? 0 : j == subN ? top_rk[subM - 1] : rk[subM - 1][j];
		double r_rk_down = r_global_j == 1 ? 0 : j == 1 ? bottom_rk[subM - 1] : rk[subM - 1][j - 2];
		double r_A_rk = -(a_ij(r_global_i + 1, r_global_j) * (r_rk_right - r_rk_ij) / h1 - a_ij(r_global_i, r_global_j) * (r_rk_ij - r_rk_left) / h1) / h1 - (b_ij(r_global_i, r_global_j + 1) * (r_rk_up - r_rk_ij) / h2 - b_ij(r_global_i, r_global_j) * (r_rk_ij - r_rk_down) / h2) / h2;
		
		local_numerator += l_A_rk * l_rk_ij + r_A_rk * r_rk_ij;
		local_denominator += l_A_rk * l_A_rk + r_A_rk * r_A_rk;
	}
	

	#pragma omp parallel for reduction(+ : local_numerator, local_denominator) num_threads(Num_threads)
	for (int i = 1; i <= subM; i++)
	{
		//j == 1
		int b_global_i = i - 1 + subI;
		int b_global_j = subJ;
		double b_rk_ij = rk[i - 1][0];
		double b_rk_left = b_global_i == 1 ? 0 : i == 1 ? left_rk[0] : rk[i - 2][0];
		double b_rk_right = b_global_i == M - 1 ? 0 : i == subM ? right_rk[0] : rk[i][0];
		double b_rk_up = b_global_j == N - 1 ? 0 : rk[i - 1][1];
		double b_rk_down = b_global_j == 1 ? 0 : bottom_rk[i - 1];
		double b_A_rk = -(a_ij(b_global_i + 1, b_global_j) * (b_rk_right - b_rk_ij) / h1 - a_ij(b_global_i, b_global_j) * (b_rk_ij - b_rk_left) / h1) / h1 - (b_ij(b_global_i, b_global_j + 1) * (b_rk_up - b_rk_ij) / h2 - b_ij(b_global_i, b_global_j) * (b_rk_ij - b_rk_down) / h2) / h2;

		//j == subN
		int t_global_i = i - 1 + subI;
		int t_global_j = subN - 1 + subJ;
		double t_rk_ij = rk[i - 1][subN - 1];
		double t_rk_left = t_global_i == 1 ? 0 : i == 1 ? left_rk[subN - 1] : rk[i - 2][subN - 1];
		double t_rk_right = t_global_i == M - 1 ? 0 : i == subM ? right_rk[subN - 1] : rk[i][subN - 1];
		double t_rk_up = t_global_j == N - 1 ? 0 : top_rk[i - 1];
		double t_rk_down = t_global_j == 1 ? 0 : rk[i - 1][subN - 2];
		double t_A_rk = -(a_ij(t_global_i + 1, t_global_j) * (t_rk_right - t_rk_ij) / h1 - a_ij(t_global_i, t_global_j) * (t_rk_ij - t_rk_left) / h1) / h1 - (b_ij(t_global_i, t_global_j + 1) * (t_rk_up - t_rk_ij) / h2 - b_ij(t_global_i, t_global_j) * (t_rk_ij - t_rk_down) / h2) / h2;
		
		local_numerator += b_A_rk * b_rk_ij + t_A_rk * t_rk_ij;
		local_denominator += b_A_rk * b_A_rk + t_A_rk * t_A_rk;
	}
	
	

	double tau_k1, global_numerator = 0.0, global_denominator = 0.0;
	MPI_Allreduce(&local_numerator, &global_numerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&local_denominator, &global_denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	tau_k1 = global_numerator / global_denominator;

	double Norm_squre = 0.0;
	#pragma omp parallel for collapse(2) reduction(+ : Norm_squre) num_threads(Num_threads)
	for (int i = 0; i < subM; i++)
	{
		for (int j = 0; j < subN; j++)
		{
			double temp = tau_k1 * rk[i][j];
			result[i][j] = wk[i][j] - temp;
			Norm_squre += temp * temp;
		}
	}
	local_norm_squre = Norm_squre;
	return result;
}

void write2DVectorToFile(const vector<vector<double> >& data, const string& filename) {
	ofstream file(filename.c_str());
	if (file.is_open()) {
		for (int j = data.size() - 1; j >= 0; j--) {
			for (int i = 0; i < data[j].size(); i++) {
				file << data[i][j] << " ";
			}
			file << "\n";
		}
		file.close();
		cout << "Data written successfully." << endl;
	}
	else {
		cerr << "Unable to open file!" << endl;
	}
}

void kernel(int M, int N, double eps, int argc, char* argv[])
{
	if (Rank == 0) {
		cout << "Situation M = " << M << ", N = " << N << ", eps = " << eps << ", Number of Threads: " << Num_threads << endl;
		cout << "Size of processes: " << Size << endl;
	}
	switch (Size)
	{
	case 1:
		M0 = 1; N0 = 1; break;
	case 2:
		M0 = 2; N0 = 1; break;
	case 4:
		M0 = 2; N0 = 2; break;
	case 8:
		M0 = 4; N0 = 2; break;
	case 16:
		M0 = 4; N0 = 4; break;
	default:
		M0 = 1; N0 = 1; break;;
	}
	GlobalI = (Rank % M0);
	GlobalJ = (Rank / M0);
	subM = GlobalI == M0 - 1 ? M / M0 - 1 : M / M0;
	subN = GlobalJ == N0 - 1 ? N / N0 - 1 : N / N0;
	subI = GlobalI * M / M0 + 1;
	subJ = GlobalJ * N / N0 + 1;
	if (Rank + M0 >= Size)
		Top = 0;
	if (Rank - M0 < 0)
		Bottom = 0;
	if (Rank % M0 - 1 < 0)
		Left = 0;
	if (Rank % M0 + 1 >= M0)
		Right = 0;
	
	h1 = (B1 - A1) / M;
	h2 = (B2 - A2) / N;
	if (h1 >= h2)
	{
		seps = h1 * h1;
	}
	else
	{
		seps = h2 * h2;
	}
	X.resize(M + 1);
	Y.resize(N + 1);

	for (int i = 0; i <= M; i++)
	{
		X[i] = A1 + i * h1;
		Y[i] = A2 + i * h2;
	}

	int count = 0;
	vector<vector<double> > w_k(subM, vector<double>(subN, 0.0));
	double obtained, local_norm_squre, sum_norm_squre;
	vector<vector<double> > r_k = Rk(w_k, count);
	w_k = M_minus_TauRk(w_k, r_k, local_norm_squre, count);
	MPI_Allreduce(&local_norm_squre, &sum_norm_squre, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	obtained = sqrt(sum_norm_squre);
	count++;
	do
	{
		r_k = Rk(w_k, count);
		w_k = M_minus_TauRk(w_k, r_k, local_norm_squre, count);
		MPI_Allreduce(&local_norm_squre, &sum_norm_squre, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		obtained = sqrt(sum_norm_squre);
		count++;
	} while (obtained > eps);

	double Max_Norm, Max_Rk, NormWk = NormMax(w_k), NormRk = NormMax(Rk(w_k, 0));
	MPI_Reduce(&NormWk, &Max_Norm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&NormRk, &Max_Rk, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	if (Rank == 0)
	{
		cout << "Number of circle:" << count << endl;
		cout << "Obtained:" << obtained << endl;
		cout << "Norm of solution:" << Max_Norm << endl;
		cout << "Norm of residual:" << Max_Rk << endl;
	}

	write2DVectorToFile(w_k, "Data10¡Á10.txt");
	return;
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	Rank = rank; Size = size;

	M = atoi(argv[1]); N = atoi(argv[2]);
	eps = (double)atof(argv[3]);
	Num_threads = atoi(argv[4]);
	
	double start_time, end_time, elapsed_time;
	start_time = MPI_Wtime();
	kernel(M, N, eps, argc, argv);
	X.clear();
	Y.clear();
	end_time = MPI_Wtime();
	elapsed_time = end_time - start_time;
	if(Rank == 0)
		cout << "The run time is: " << elapsed_time << "s" << endl;
	if (Rank == 0) {
		cout << "argc = " << argc << endl;
		for (int i = 0; i < argc; i++) {
			cout << "argv[" << i << "] = " << argv[i] << endl;
		}
	}
	MPI_Finalize();
	return 0;
}

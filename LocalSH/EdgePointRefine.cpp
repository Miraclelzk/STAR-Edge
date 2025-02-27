#include <nanoflann.hpp>

#include "EdgePointRefine.h"
#include <omp.h>
#include <iostream>
#include <random>
#include <vector>
#include <optimization.h>
#include <map>
bool EdgePointRefine::edgePointRe(std::vector<bool>& flag, std::vector<std::vector<int>>& neighboor, Eigen::MatrixX3d& pos, Eigen::MatrixX3d& new_normals, double mu, Eigen::MatrixX3d & NewEdgePoints)
{

	// double mu = 0.01;
	
	//
	std::map<int, bool> flag3;
	// Eigen::VectorXd std::map<int, bool>
	for (int i = 0; i < flag.size(); ++i) {
		flag3[i] = flag[i];  // flag[i] 0 flag3[i] true false
	}
	
	// std::vector<Eigen::Vector3d>
	std::vector<Eigen::Vector3d> Vall;
	for (int i = 0; i < pos.rows(); ++i) {
		Vall.push_back(pos.row(i));
	}
	
	std::vector<Eigen::Vector3d> Nall_new;
	for (int i = 0; i < new_normals.rows(); ++i) {
		Nall_new.push_back(new_normals.row(i)); // 
	}

	bool iw = 0;
	
	std::map<int, Eigen::Vector3d> NewPoints;

	std::cout << "EPR-start" << std::endl;

	for (int iter = 0; iter < pos.rows(); iter++)
	{
		if (!flag3[iter])
		{
			continue;
		}
		int k = neighboor[iter].size();
		if (k < 5)
		{
			continue;
		}

		alglib::real_1d_array x0;
		x0.setlength(3);
		x0[0] = Vall[iter].x();
		x0[1] = Vall[iter].y();
		x0[2] = Vall[iter].z();
		alglib::real_1d_array s0 = "[1,1,1]";

		std::function<void(const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr)> fop_z_lambda
			= [&](const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr) -> void
		{
			int k = neighboor[iter].size();
			int r = Vall.size();


			Eigen::Vector3d z(x[0], x[1], x[2]);
			
			func = 0;
			for (int i = 0; i < k; i++)
			{
				auto zp = Vall[neighboor[iter][i]] - z;
				func += pow(zp.dot(Nall_new[neighboor[iter][i]]), 2);
				//func += x[i] * w[i] * (Nall[neighboor[iter][i]] - n1).squaredNorm()  + (1 - x[i]) * w[i] * (Nall[neighboor[iter][i]] - n2).squaredNorm();
			}
			func = func + mu * (Vall[iter] - z).squaredNorm();

			Eigen::Vector3d g(0, 0, 0);
			g = 2 * mu * (z - Vall[iter]);
			for (int i = 0; i < k; i++)
			{
				//double a = abs(cos(x[k]))* abs(cos(x[k]));
				auto Pj = Vall[neighboor[iter][i]];
				auto nj = Nall_new[neighboor[iter][i]];
				g = g + 2 * (z - Pj).dot(nj) * nj;
			}

			grad[0] = g.x();
			grad[1] = g.y();
			grad[2] = g.z();
		};

		alglib::minbleicstate state;
		double epsg = 0;
		double epsf = 0;
		double epsx = 0;
		alglib::ae_int_t maxits = 0;
		alglib::minbleiccreate(x0, state);
		alglib::minbleicsetscale(state, s0);
		alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);

		alglib::minbleicoptguardsmoothness(state);
		alglib::minbleicoptguardgradient(state, 0.000001);

		alglib::minbleicreport rep2;
		//minbleicoptimize(state, fop, nullptr, nullptr, alglib::parallel);
		alglib::minbleicoptimize(state, fop_z_lambda);
		alglib::minbleicresults(state, x0, rep2);
		//cout << rep2.debugff << endl;
		double mn = 0;
		alglib::real_1d_array G_tmp;
		G_tmp.setlength(3);
		fop_z_lambda(x0, mn, G_tmp, nullptr);
		
		NewPoints[iter] = Eigen::Vector3d(x0[0], x0[1], x0[2]);

	}

	int rowIndex = 0;
	//NewEdgePoints.setZero();
	for (const auto& pair : flag3) {
		if (pair.second != 0) {
			NewEdgePoints.row(rowIndex++) = NewPoints[pair.first];
		}
	}
	NewEdgePoints.conservativeResize(rowIndex, Eigen::NoChange);

	std::cout << "EPR-end" << std::endl;

	return true;
}

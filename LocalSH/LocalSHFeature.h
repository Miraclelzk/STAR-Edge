#pragma once

#include <vector>
#include <Eigen/Dense>


class LocalSHFeature
{
public:

    /**
    * @brief ->
    *
    * @param pcc 
    * @param pos
    * @param bw
    * @param R
    * @param Descs
    *
    * @return
    *
    */
	bool LocalSH(Eigen::MatrixX3d& pcc, int _bw, std::vector<double> &Descriptor);

    /**
    * @brief -> (radius)
    *
    * @param pc
    * @param pos
    * @param bw
    * @param radius
    * @param Descs
    *
    * @return
    *
    */
	bool ComLocalSHFeatures(Eigen::MatrixX3d& pc, Eigen::MatrixX3d& pos, int _bw, float radius, Eigen::MatrixXd& Descs);
	bool ComLocalSHFeatures_standard(Eigen::MatrixX3d pcc, int _bw, Eigen::VectorXd & oneDescs);
	bool ComLocalSHFeatures_knn_upsample(Eigen::MatrixX3d & pc,
		Eigen::MatrixX3d & pos, 
		int _bw,
		int k, 
		int numSamples, 
		Eigen::MatrixXd & Descs,
		Eigen::MatrixX3d & normals, 
		std::vector<std::vector<int>> & neighboor);


	bool ComLocalSHFeatures_knn_nosample(Eigen::MatrixX3d & pc, 
		Eigen::MatrixX3d & pos, 
		int _bw, 
		int k, 
		int numSamples, 
		Eigen::MatrixXd & Descs, 
		Eigen::MatrixX3d & normals, 
		std::vector<std::vector<int>>& neighboor);
	/**
	* @brief -> (knn)
	*
	* @param pc
	* @param pos
	* @param bw
	* @param k kn
	* @param Descs
	*
	* @return
	*
	*/

	

	/**
	* @brief (radius)
	*
	* @param pc
	* @param pos
	* @param bw
	* @param radius
	* @param pcc
	* @param data
	* @param coeffs
	*
	* @return
	*
	*/
	bool ComOneLocalSHFeatures(Eigen::MatrixX3d& pc, Eigen::Vector3d pos, int _bw, float radius, Eigen::MatrixX3d& pcc, double*& data, std::vector<std::complex<double>>& coeffs);

	/**
	* @brief (radius)
	*
	* @param pc
	* @param pos
	* @param bw
	* @param radius
	* @param pcc
	* @param data
	* @param coeffs
	*
	* @return
	*
	*/
	bool ComOneLocalSHFeatures_knn(Eigen::MatrixX3d& pc, Eigen::Vector3d pos, int _bw, int radius, Eigen::MatrixX3d& pcc, double*& data, std::vector<std::complex<double>>& coeffs);

	/**
	* @brief
	*
	* @param pc
	* @param coeffs
	*
	* @return
	*
	*/
	void ComCoeffsToDesc(std::vector<std::complex<double>>*& coeffs, std::vector<double> &Descriptor);
public:

	
	void setProgressCallback(std::function<void(int)> callback)
	{
		progressCallback = callback;
	}

	std::function<void(int)> progressCallback;
};
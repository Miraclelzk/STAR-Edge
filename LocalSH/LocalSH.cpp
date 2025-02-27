#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <LocalSHFeature.h>
#include <NormalRefine.h>

#include "EdgePointRefine.h"

namespace py = pybind11;

void numpy_to_vector_bool(py::array_t<int> input_array, std::vector<bool>& output_vector) {
    py::buffer_info buf_info = input_array.request();
    
    if (buf_info.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }

    int* ptr = static_cast<int*>(buf_info.ptr);
    
    output_vector.resize(buf_info.size);
    
    for (size_t i = 0; i < buf_info.size; ++i) {
        output_vector[i] = (ptr[i] != 0);  
    }
}

py::list convert_vector_to_pylist(const std::vector<std::vector<int>>& vec) {
    py::list result;
    for (const auto& inner_vec : vec) {
        py::list inner_list;
        for (int item : inner_vec) {
            inner_list.append(item);
        }
        result.append(inner_list);
    }
    return result;
}

std::vector<std::vector<int>> convert_pylist_to_vector(const py::list& py_list) {
    std::vector<std::vector<int>> result;
    
    for (const auto& item : py_list) {
        if (py::isinstance<py::list>(item)) {
            py::list inner_list = item.cast<py::list>();
            std::vector<int> inner_vec;
            for (const auto& value : inner_list) {
                inner_vec.push_back(value.cast<int>());
            }
            result.push_back(inner_vec);
        } else {
            throw std::runtime_error("Element in the Python list is not a list");
        }
    }

    return result;
}

void numpy_to_MatrixX3d(const py::array_t<double>& input, Eigen::MatrixX3d& out) {
    // NumPy
    auto buf = input.unchecked<2>(); // 2D array
    size_t n = buf.shape(0); //
    size_t m = buf.shape(1); //

    // n*3
    if (m != 3) {
        throw std::invalid_argument("Input array must have shape (n, 3).");
    }

    // Eigen
    out.resize(n, 3);

    // NumPy Eigen
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            out(i, j) = buf(i, j);
        }
    }
}

py::array_t<double> MatrixXd_to_numpy(const Eigen::MatrixXd& matrix) {
    //
    size_t n = matrix.rows();
    size_t m = matrix.cols();

    // NumPy
    py::array_t<double> result = py::array_t<double>({n, m});
    auto buf = result.mutable_unchecked<2>(); // 2D

    // Eigen NumPy
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            buf(i, j) = matrix(i, j);
        }
    }

    return result;
}

py::dict ComLocalSHFeatures_knn_upsample(py::array_t<double>& input, int bw, int kk, int sampleNum)
{
	//
	Eigen::MatrixX3d pc;
	numpy_to_MatrixX3d(input, pc);

	//
	Eigen::MatrixX3d pos = pc;

	Eigen::MatrixXd Descs;
	Descs.resize(pos.rows(), bw);
	Descs.setZero();
	Eigen::MatrixX3d normals;
	normals.resize(pos.rows(), 3);
	normals.setZero();
	std::vector<std::vector<int>> neighboor(pos.rows());

	LocalSHFeature LSHF;
	LSHF.ComLocalSHFeatures_knn_upsample(pc, pos, bw, kk, sampleNum, Descs, normals, neighboor);//0.05

	py::dict result;
    result["Descs"] = MatrixXd_to_numpy(Descs);
    result["normals"] = MatrixXd_to_numpy(normals);
    result["neighboor"] = convert_vector_to_pylist(neighboor);

    return result;
}

py::dict ComLocalSHFeatures_knn_nosample(py::array_t<double>& input, int bw, int kk, int sampleNum)
{
	//
	Eigen::MatrixX3d pc;
	numpy_to_MatrixX3d(input, pc);

	//
	Eigen::MatrixX3d pos = pc;

	Eigen::MatrixXd Descs;
	Descs.resize(pos.rows(), bw);
	Descs.setZero();
	Eigen::MatrixX3d normals;
	normals.resize(pos.rows(), 3);
	normals.setZero();
	std::vector<std::vector<int>> neighboor(pos.rows());

	LocalSHFeature LSHF;
	LSHF.ComLocalSHFeatures_knn_nosample(pc, pos, bw, kk, sampleNum, Descs, normals, neighboor);//0.05

	py::dict result;
    result["Descs"] = MatrixXd_to_numpy(Descs);
    result["normals"] = MatrixXd_to_numpy(normals);
    result["neighboor"] = convert_vector_to_pylist(neighboor);

    return result;
}

py::array_t<double> checkNormalDirection(py::array_t<double>& normal_origin_, py::array_t<double>& normals_)
{
    Eigen::MatrixX3d normal_origin;
	numpy_to_MatrixX3d(normal_origin_, normal_origin);

    Eigen::MatrixX3d normals;
	numpy_to_MatrixX3d(normals_, normals);

    Eigen::MatrixX3d new_normals;
    new_normals.resize(normals.rows(), 3);
    new_normals.setZero();

    NormalRefine NormRe;
    NormRe.checkNormalDirection(normal_origin, normals, new_normals);

    return MatrixXd_to_numpy(new_normals);
}

py::array_t<double> EdgePointRe(py::array_t<double>& flag_, py::list& neighboor_, py::array_t<double>& pos_, py::array_t<double>& normals_, double mu)
{
    std::vector<bool> edge_flag;
    numpy_to_vector_bool(flag_, edge_flag);

    std::vector<std::vector<int>> neighboor = convert_pylist_to_vector(neighboor_);
    Eigen::MatrixX3d pos;
	numpy_to_MatrixX3d(pos_, pos);

    Eigen::MatrixX3d normals;
	numpy_to_MatrixX3d(normals_, normals);

    Eigen::MatrixX3d NewEdgePoints(pos.rows(), 3);

    EdgePointRefine EdgeRe;
    EdgeRe.edgePointRe(edge_flag, neighboor, pos, normals, mu, NewEdgePoints);

    return MatrixXd_to_numpy(NewEdgePoints);
}


PYBIND11_MODULE(LocalSH, m) 
{

	m.doc() = "Python binding of LocalSH";

	py::module m_sub1 = m.def_submodule("LocalSHFeature");

	m_sub1.def("ComLSHF_knn_upsample", &ComLocalSHFeatures_knn_upsample);
    m_sub1.def("ComLSHF_knn_nosample", &ComLocalSHFeatures_knn_nosample);

    py::module m_sub2 = m.def_submodule("NormalRefine");
	m_sub2.def("checkNormalDirection", &checkNormalDirection);
	m_sub2.def("EdgePointRefine", &EdgePointRe);

}


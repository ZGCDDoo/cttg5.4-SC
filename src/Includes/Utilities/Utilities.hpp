#pragma once

#include <cmath>
#include <iostream>
#include <fstream>
#include <valarray>
#include <vector>
#include <utility>
#include <set>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <mm_malloc.h>
#include <vector>
#include <string>
#include <ccomplex>

//External Libraries
#include <armadillo>
#include "json.hpp"

using Json = nlohmann::json;

//Inspired by Patrick Sémon

using cd_t = std::complex<double>;
using Sign_t = int;
using Site_t = size_t;
using Tau_t = double;

enum class AuxSpin_t
{
	Up,
	Down,
	Zero
};

enum class FermionSpin_t
{
	Up,
	Down
};

using SiteVectorCD_t = arma::cx_vec;
using SiteRowCD_t = arma::cx_rowvec;
using ClusterSitesCD_t = std::vector<arma::cx_vec>;
using ClusterMatrixCD_t = arma::cx_mat;
using ClusterCubeCD_t = arma::cx_cube;

using SiteVector_t = arma::vec;
using SiteRow_t = arma::rowvec;
using ClusterSites_t = std::vector<arma::vec>;
using ClusterMatrix_t = arma::mat;
using ClusterCube_t = arma::cube;

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

namespace Utilities
{

typedef boost::mt19937 EngineTypeMt19937_t;
typedef boost::lagged_fibonacci3217 EngineTypeFibonacci3217_t;
typedef boost::uniform_real<double> UniformDistribution_t;
typedef boost::variate_generator<EngineTypeMt19937_t &, UniformDistribution_t> UniformRngMt19937_t;
typedef boost::variate_generator<EngineTypeFibonacci3217_t &, UniformDistribution_t> UniformRngFibonacci3217_t;

std::string GetSpinName(const FermionSpin_t &spin)
{
	return (spin == FermionSpin_t::Up ? "Up" : "Down");
}

class Vertex
{

  public:
	Vertex(){};
	Vertex(const Tau_t &tau, const Site_t &site, const AuxSpin_t &aux) : tau_(tau),
																		 site_(site),
																		 aux_(aux) {}

	const Vertex &operator=(const Vertex &vertex)
	{
		if (this == &vertex)
			return *this; //évite les boucles infinies
		tau_ = vertex.tau_;
		site_ = vertex.site_;
		aux_ = vertex.aux_;

		return *this;
	}

	// Getters
	Tau_t tau() const { return tau_; };
	Site_t site() const { return site_; };
	AuxSpin_t aux() const { return aux_; };

	//Setters
	void SetAux(AuxSpin_t aux)
	{
		aux_ = aux;
	}

	void FlipAux() { aux_ == AuxSpin_t::Up ? aux_ = AuxSpin_t::Down : aux_ = AuxSpin_t::Up; };

	double Ising()
	{
		if (aux_ == AuxSpin_t::Zero)
		{
			return 0.0;
		}
		return (aux_ == AuxSpin_t::Up ? 1.0 : -1.0);
	};

  private:
	Tau_t tau_;
	Site_t site_;
	AuxSpin_t aux_;
};

class VertexBuilder
{
};

} // namespace Utilities
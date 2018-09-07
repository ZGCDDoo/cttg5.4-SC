#pragma once

#include "../Utilities/Utilities.hpp"
#include "../Utilities/GreenTau.hpp"
#include "../Utilities/Matrix.hpp"

// #if defined(SUBMATRIX)
// #include "ABC_MarkovChainSubMatrix.hpp"
// #else
// #include "ABC_MarkovChain.hpp"
// #endif

namespace Markov
{

#if defined SUBMATRIX
template <typename TIOModel, typename TModel>
class ABC_MarkovChainSubMatrix;
#else
template <typename TIOModel, typename TModel>
class ABC_MarkovChain;
#endif

namespace Obs
{

// Observables that need to befriend ISDataCT
template <typename TIOModel, typename TModel>
class Observables;

template <typename TIOModel, typename TModel>
class GreenBinning;

template <typename TIOModel, typename TModel>
class FillingAndDocc;

template <typename TIOModel, typename TModel>
class GreenTauMesure;
// end of Obs befriend

using Vertex = Utilities::Vertex;
using namespace LinAlg;

template <typename TIOModel, typename TModel>
class ISDataCT
{
  using GreenTau_t = GreenTau::GreenCluster0Tau<TIOModel>;

public:
  ISDataCT(const double &beta, TModel model, const size_t &NTau) : MupPtr_(new Matrix_t()),
                                                                   MdownPtr_(new Matrix_t()),
                                                                   vertices_(),
                                                                   beta_(beta),
                                                                   sign_(1)

  {
#ifdef AFM
    green0CachedUp_ = GreenTau::GreenCluster0Tau<TIOModel>(model.greenCluster0MatUp(), NTau);
    green0CachedDown_ = GreenTau::GreenCluster0Tau<TIOModel>(model.greenCluster0MatDown(), NTau);
#endif
#ifndef AFM
    green0CachedUp_ = GreenTau::GreenCluster0Tau<TIOModel>(model.greenCluster0MatUp(), NTau);
#endif
  }

  double beta() const { return beta_; };

private:
  friend class Markov::Obs::Observables<TIOModel, TModel>;
  friend class Markov::Obs::GreenBinning<TIOModel, TModel>;
  friend class Markov::Obs::FillingAndDocc<TIOModel, TModel>;
  friend class Markov::Obs::GreenTauMesure<TIOModel, TModel>;

#if defined SUBMATRIX
  friend class Markov::ABC_MarkovChainSubMatrix<TIOModel, TModel>;
#else
  friend class Markov::ABC_MarkovChain<TIOModel, TModel>;
#endif

  GreenTau_t green0CachedUp_; //what the hell, this should be const. Better, Moved to Model Shit Fuck
  GreenTau_t green0CachedDown_;
  std::shared_ptr<Matrix_t> MupPtr_;
  std::shared_ptr<Matrix_t> MdownPtr_;
  std::vector<Vertex> vertices_;
  const double beta_;
  Sign_t sign_;
};
} // namespace Obs
} // namespace Markov
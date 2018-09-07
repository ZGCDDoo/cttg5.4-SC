#include "../../Utilities/Utilities.hpp"
#include "../../Utilities/Fourier.hpp"

namespace Markov
{
namespace Obs
{

template <typename TModel, typename TIOModel>
class KineticEnergy
{
  public:
    KineticEnergy(const std::shared_ptr<TModel> &modelPtr, const ClusterCubeCD_t &greenMat) : modelPtr_(modelPtr),
                                                                                              ioModel_(),
                                                                                              greenMat_(greenMat){};

    double GetKineticEnergy()
    {
        const size_t Nc = modelPtr_->Nc;
        ClusterMatrix_t greentau0(Nc, Nc); //green_interacting(tau=0-)
        greentau0.zeros();

        const double TAU0 = 1e-12;

        ClusterMatrix_t nUpMatrix;
        assert(nUpMatrix.load("nUpMatrix.dat"));
        ClusterMatrix_t nDownMatrix;
        assert(nDownMatrix.load("nDownMatrix.dat"));
        ClusterMatrixCD_t nMatrix(nUpMatrix + nDownMatrix, ClusterMatrix_t(Nc, Nc).zeros());

        const double U = modelPtr_->U();
        const double mu = modelPtr_->mu();
        const double beta = modelPtr_->beta();
        const ClusterMatrixCD_t tLoc = modelPtr_->tLoc();

        ClusterMatrixCD_t selfEnergyZM = 0.5 * U * nMatrix;
        ClusterMatrixCD_t selfEnergyFM = U * U * nMatrix / 2.0 * (ClusterMatrixCD_t(Nc, Nc).eye() - nMatrix / 2.0);

        ClusterMatrixCD_t hybFM;
        assert(hybFM.load("hybFM.arma"));

        const ClusterMatrixCD_t FM = ClusterMatrixCD_t(Nc, Nc).eye();
        const ClusterMatrixCD_t SM = tLoc + selfEnergyZM - mu * ClusterMatrixCD_t(Nc, Nc).eye();
        const ClusterMatrixCD_t TM = (tLoc + selfEnergyZM - mu * ClusterMatrixCD_t(Nc, Nc).eye()) * (tLoc + selfEnergyZM - mu * ClusterMatrixCD_t(Nc, Nc).eye()) + hybFM + selfEnergyFM;

        //Get G_(tau=beta/2) for each indepsite
        std::ofstream fout("GTauBetaOver2.dat", std::ios_base::out | std::ios_base::app);
        for (size_t ii = 0; ii < ioModel_.indepSites().size(); ii++)
        {
            const Site_t s1 = ioModel_.indepSites().at(ii).first;
            const Site_t s2 = ioModel_.indepSites().at(ii).second;
            const double temp = Fourier::MatToTauAnalytic(greenMat_.tube(s1, s2), beta / 2.0, beta, FM(s1, s2).real(), SM(s1, s2).real(), TM(s1, s2).real());
            fout << temp << " ";
        }
        fout << std::endl;
        fout.close();

        //Get G_(0-) for each site
        // and kinectic energy
        double KEnergy = 0.0;
        for (size_t jj = 0; jj < greenMat_.n_cols; jj++)
        {
            for (size_t ii = 0; ii < greenMat_.n_rows; ii++)
            {
                greentau0(ii, jj) = Fourier::MatToTauAnalytic(greenMat_.tube(ii, jj), TAU0, beta, FM(ii, jj).real(), SM(ii, jj).real(), TM(ii, jj).real());
                KEnergy += tLoc(ii, jj).real() * greentau0(ii, jj);
            }
        }

        //spin degenearacy for paramagnetic case:
        KEnergy *= 2.0 / static_cast<double>(Nc);

#ifdef AFM
        mpiUt::Print("\t Warning !! \n KineticEnergy wrong for AFM. Correction to come. Contact developper.");
#else
        mpiUt::Print("Warning, Kinetic Energy **might** be wrong, needs additionnal verification.");
#endif

        return KEnergy;
    }

  private:
    const std::shared_ptr<TModel> modelPtr_;
    TIOModel ioModel_;
    const ClusterCubeCD_t greenMat_;
};

} // namespace Obs
} // namespace Markov
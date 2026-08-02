#include "VTKWriter.hpp"
#include "LBMSolver.hpp"
#include "IBMParticle.hpp"
#include <fstream>
namespace io {
void writeFluid(const LBMSolver& f,const std::string& name){ std::ofstream out(name); out<<"# vtk DataFile Version 3.0\nLBM fluid\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS "<<f.nx()<<" "<<f.ny()<<" 1\nORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA "<<f.nodes()<<"\nVECTORS velocity double\n"; for(int n=0;n<f.nodes();++n) out<<f.velocityX(n)<<" "<<f.velocityY(n)<<" 0\n"; }
void writeParticle(const IBMParticle& p,const std::string& name){ std::ofstream out(name); out<<"# vtk DataFile Version 3.0\nIBM particle\nASCII\nDATASET POLYDATA\nPOINTS "<<p.markerCount<<" double\n"; for(int k=0;k<p.markerCount;++k) out<<p.markerX[k]<<" "<<p.markerY[k]<<" 0\n"; out<<"LINES "<<p.markerCount<<" "<<3*p.markerCount<<"\n"; for(int k=0;k<p.markerCount;++k) out<<"2 "<<k<<" "<<(k+1)%p.markerCount<<"\n"; }
}

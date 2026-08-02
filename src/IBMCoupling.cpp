#include "IBMCoupling.hpp"
#include "LBMSolver.hpp"
#include "IBMParticle.hpp"
#include <cmath>
namespace ibm {
double deltaKernel(double r){ r=std::abs(r); if(r<1.0) return 0.125*(3-2*r+std::sqrt(1+4*r-4*r*r)); if(r<2.0) return 0.125*(5-2*r-std::sqrt(-7+12*r-4*r*r)); return 0.0; }
void interpolate(const LBMSolver& fluid, IBMParticle& p){
    for(int k=0;k<p.markerCount;++k){ p.fluidUx[k]=p.fluidUy[k]=0.0; int bx=(int)std::floor(p.markerX[k]), by=(int)std::floor(p.markerY[k]);
        for(int j=by-1;j<=by+2;++j) for(int i=bx-1;i<=bx+2;++i){ if(i<0||i>=fluid.nx()||j<0||j>=fluid.ny()) continue; double w=deltaKernel(i-p.markerX[k])*deltaKernel(j-p.markerY[k]); int n=fluid.nodeIndex(i,j); p.fluidUx[k]+=fluid.velocityX(n)*w; p.fluidUy[k]+=fluid.velocityY(n)*w; }}
}
void computeDirectForcing(IBMParticle& p,double dt){
    for(int k=0;k<p.markerCount;++k){ double rx=p.markerX[k]-p.x, ry=p.markerY[k]-p.y; double ubx=p.vx-p.omega*ry, uby=p.vy+p.omega*rx; p.markerForceX[k]=(ubx-p.fluidUx[k])/dt; p.markerForceY[k]=(uby-p.fluidUy[k])/dt; }
}
void spreadToFluid(LBMSolver& fluid,const IBMParticle& p){
    for(int k=0;k<p.markerCount;++k){ int bx=(int)std::floor(p.markerX[k]), by=(int)std::floor(p.markerY[k]);
        for(int j=by-1;j<=by+2;++j) for(int i=bx-1;i<=bx+2;++i){ if(i<0||i>=fluid.nx()||j<0||j>=fluid.ny()) continue; double w=deltaKernel(i-p.markerX[k])*deltaKernel(j-p.markerY[k]); int n=fluid.nodeIndex(i,j); fluid.forceX(n)+=p.markerForceX[k]*w*p.ds; fluid.forceY(n)+=p.markerForceY[k]*w*p.ds; }}
}
void accumulateHydrodynamicLoads(IBMParticle& p){ p.clearLoads();
    for(int k=0;k<p.markerCount;++k){ double fx=-p.markerForceX[k]*p.ds, fy=-p.markerForceY[k]*p.ds; double rx=p.markerX[k]-p.x, ry=p.markerY[k]-p.y; p.forceX+=fx; p.forceY+=fy; p.torque+=rx*fy-ry*fx; }
}
}

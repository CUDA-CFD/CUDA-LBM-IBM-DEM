#pragma once
#include <array>
namespace lbm {
struct D2Q9 {
    static constexpr int Q = 9;
    inline static constexpr std::array<int,Q> cx{0,1,0,-1,0,1,-1,-1,1};
    inline static constexpr std::array<int,Q> cy{0,0,1,0,-1,1,1,-1,-1};
    inline static constexpr std::array<int,Q> opposite{0,3,4,1,2,7,8,5,6};
    inline static constexpr std::array<double,Q> w{4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};
};
}

#include "corpus/lpf.hpp"
#include <algorithm>
#include <stdexcept>
namespace corpus::lpf {
std::int64_t evaluate(const Knots& w, std::int64_t time) {
    if(time<0) throw std::invalid_argument("LPF time must be nonnegative");
    const auto t=time%86400; const auto i=static_cast<std::size_t>(t/900);
    const auto r=t%900;
    if(w[i]<=0 || w[(i+1)%96]<=0) throw std::invalid_argument("LPF knots must be positive");
    // GES/libs/lpf/linearPiecewiseFunction.cpp:401 computeY used
    // y1 + (y2-y1)*(x-x1)/(x2-x1). This equivalent convex form avoids
    // signed subtraction/product overflow; ceil preserves integer FIFO.
    const __int128_t n=static_cast<__int128_t>(w[i])*(900-r)+static_cast<__int128_t>(w[(i+1)%96])*r;
    return static_cast<std::int64_t>((n+899)/900);
}
std::uint64_t violations(const Knots& w) {
    std::uint64_t n=0;
    for(std::size_t i=0;i<96;++i) {
        if(w[i]<=0) throw std::invalid_argument("LPF knots must be positive");
        if(static_cast<__int128_t>(w[i])-w[(i+1)%96]>900) ++n;
    }
    return n;
}
Knots least_fifo_majorant(const Knots& w) {
    if(violations(w)==0) return w;
    Knots r=w;
    for(std::size_t i=0;i<96;++i) for(std::size_t j=0;j<96;++j)
        r[i]=std::max(r[i],w[j]-static_cast<std::int64_t>(900*((i+96-j)%96)));
    return r;
}
}

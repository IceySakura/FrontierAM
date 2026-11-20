#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>

namespace memory {

template<class T>
struct AlignedDelete {
    void operator()(T* p) const { ::operator delete[](p,std::align_val_t(64)); }
};

template<class T> using Array = std::unique_ptr<T[],AlignedDelete<T>>;

template<class T>
Array<T> allocate(std::size_t count) {
    static_assert(std::is_trivial_v<T>);
    return Array<T>(new(std::align_val_t(64)) T[count]);
}

template<class T>
Array<T> zeros(std::size_t count) {
    auto a = allocate<T>(count);
    std::fill_n(a.get(),count,T{});
    return a;
}

} // namespace memory

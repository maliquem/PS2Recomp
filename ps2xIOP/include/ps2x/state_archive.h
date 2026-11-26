#pragma once

// Save state serialization: one symmetric io() per type, used for both saving and loading.
// The format is only valid for the exact same executable (checked by the save state header),
// so trivially copyable types are written as raw bytes.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace ps2x::state
{
    class Archive
    {
    public:
        // Saving archive.
        explicit Archive(std::vector<uint8_t> &out) : m_out(&out) {}
        // Loading archive.
        Archive(const uint8_t *data, size_t size) : m_in(data), m_inSize(size) {}

        bool loading() const { return m_out == nullptr; }
        size_t remaining() const { return m_inSize - m_inPos; }

        void raw(void *data, size_t size)
        {
            if (size == 0)
                return;
            if (m_out)
            {
                const auto *p = static_cast<const uint8_t *>(data);
                m_out->insert(m_out->end(), p, p + size);
                return;
            }
            if (size > m_inSize - m_inPos)
                throw std::runtime_error("save state truncated");
            std::memcpy(data, m_in + m_inPos, size);
            m_inPos += size;
        }

        uint64_t size(uint64_t value)
        {
            raw(&value, sizeof(value));
            if (loading() && value > remaining() + (1ull << 32))
                throw std::runtime_error("save state corrupt (size)");
            return value;
        }

        template <typename... T>
        void operator()(T &...values)
        {
            (io(*this, values), ...);
        }

    private:
        std::vector<uint8_t> *m_out = nullptr;
        const uint8_t *m_in = nullptr;
        size_t m_inSize = 0;
        size_t m_inPos = 0;
    };

    // Types with a member `stateIo(Archive&)` (classes and structs that are not plain data).
    template <typename T, typename = void>
    struct HasStateIo : std::false_type
    {
    };
    template <typename T>
    struct HasStateIo<T, std::void_t<decltype(std::declval<T &>().stateIo(std::declval<Archive &>()))>> : std::true_type
    {
    };

    template <typename T>
    std::enable_if_t<HasStateIo<T>::value> io(Archive &ar, T &value)
    {
        value.stateIo(ar);
    }

    template <typename T>
    std::enable_if_t<std::is_trivially_copyable_v<T> && !std::is_pointer_v<T> && !HasStateIo<T>::value> io(Archive &ar, T &value)
    {
        ar.raw(&value, sizeof(T));
    }

    inline void io(Archive &ar, std::string &s)
    {
        const uint64_t n = ar.size(s.size());
        if (ar.loading())
            s.resize(static_cast<size_t>(n));
        ar.raw(s.data(), s.size());
    }

    template <typename T>
    void io(Archive &ar, std::vector<T> &v)
    {
        const uint64_t n = ar.size(v.size());
        if (ar.loading())
        {
            v.clear();
            v.resize(static_cast<size_t>(n));
        }
        if constexpr (std::is_trivially_copyable_v<T>)
            ar.raw(v.data(), v.size() * sizeof(T));
        else
            for (auto &e : v)
                io(ar, e);
    }

    template <typename T>
    void io(Archive &ar, std::deque<T> &v)
    {
        const uint64_t n = ar.size(v.size());
        if (ar.loading())
        {
            v.clear();
            v.resize(static_cast<size_t>(n));
        }
        for (auto &e : v)
            io(ar, e);
    }

    template <typename T, size_t N>
    std::enable_if_t<!std::is_trivially_copyable_v<std::array<T, N>>> io(Archive &ar, std::array<T, N> &a)
    {
        for (auto &e : a)
            io(ar, e);
    }

    template <typename T>
    void io(Archive &ar, std::optional<T> &o)
    {
        bool has = o.has_value();
        ar.raw(&has, sizeof(has));
        if (ar.loading())
        {
            o.reset();
            if (has)
                o.emplace();
        }
        if (has)
            io(ar, *o);
    }

    template <typename A, typename B>
    std::enable_if_t<!std::is_trivially_copyable_v<std::pair<A, B>>> io(Archive &ar, std::pair<A, B> &p)
    {
        io(ar, p.first);
        io(ar, p.second);
    }

    template <typename T>
    void io(Archive &ar, std::atomic<T> &a)
    {
        T v = a.load();
        io(ar, v);
        if (ar.loading())
            a.store(v);
    }

    namespace detail
    {
        template <typename V, size_t I = 0>
        void emplaceIndex(V &v, size_t index)
        {
            if constexpr (I < std::variant_size_v<V>)
            {
                if (index == I)
                    v.template emplace<I>();
                else
                    emplaceIndex<V, I + 1>(v, index);
            }
            else
                throw std::runtime_error("save state corrupt (variant)");
        }

        // Ordered and multi maps: element order is the key order, so a plain insert rebuilds it.
        template <typename M>
        void ioMap(Archive &ar, M &m)
        {
            const uint64_t n = ar.size(m.size());
            if (!ar.loading())
            {
                for (auto &kv : m)
                {
                    auto key = kv.first;
                    io(ar, key);
                    io(ar, kv.second);
                }
                return;
            }
            m.clear();
            for (uint64_t i = 0; i < n; ++i)
            {
                typename M::key_type key{};
                io(ar, key);
                typename M::mapped_type value{};
                io(ar, value);
                m.emplace(std::move(key), std::move(value));
            }
        }

        // Unordered containers: iteration order matters to the guest when the runtime walks them
        // (e.g. picking the first ready handler), so keep the bucket count and re-insert in the saved
        // order. With the same standard library this reproduces the original iteration order.
        template <typename M>
        void ioUnordered(Archive &ar, M &m)
        {
            uint64_t buckets = m.bucket_count();
            ar.raw(&buckets, sizeof(buckets));
            const uint64_t n = ar.size(m.size());
            if (!ar.loading())
            {
                for (auto &e : m)
                {
                    if constexpr (std::is_same_v<typename M::key_type, typename M::value_type>)
                    {
                        auto key = e;
                        io(ar, key);
                    }
                    else
                    {
                        auto key = e.first;
                        io(ar, key);
                        io(ar, e.second);
                    }
                }
                return;
            }
            M fresh;
            fresh.rehash(static_cast<size_t>(buckets));
            for (uint64_t i = 0; i < n; ++i)
            {
                typename M::key_type key{};
                io(ar, key);
                if constexpr (std::is_same_v<typename M::key_type, typename M::value_type>)
                    fresh.insert(std::move(key));
                else
                {
                    typename M::mapped_type value{};
                    io(ar, value);
                    fresh.emplace(std::move(key), std::move(value));
                }
            }
            m.swap(fresh);
        }
    }

    template <typename... T>
    void io(Archive &ar, std::variant<T...> &v)
    {
        uint64_t index = v.index();
        ar.raw(&index, sizeof(index));
        if (ar.loading())
            detail::emplaceIndex(v, static_cast<size_t>(index));
        std::visit([&ar](auto &alt) { io(ar, alt); }, v);
    }

    inline void io(Archive &, std::monostate &) {}

    template <typename K, typename V, typename C, typename A>
    void io(Archive &ar, std::map<K, V, C, A> &m) { detail::ioMap(ar, m); }

    template <typename K, typename V, typename C, typename A>
    void io(Archive &ar, std::multimap<K, V, C, A> &m) { detail::ioMap(ar, m); }

    template <typename K, typename V, typename H, typename E, typename A>
    void io(Archive &ar, std::unordered_map<K, V, H, E, A> &m) { detail::ioUnordered(ar, m); }

    template <typename K, typename H, typename E, typename A>
    void io(Archive &ar, std::unordered_set<K, H, E, A> &m) { detail::ioUnordered(ar, m); }
}

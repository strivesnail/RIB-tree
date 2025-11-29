#pragma once

#include <immintrin.h>
#include <omp.h>
#include <xmmintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define overflowCapacity 3
#define maxKey 64

volatile int dummy;
using namespace std;

// Forward declare seg namespace for keySegment
namespace seg {
    template<typename KeyType> struct keySegment;
}

// Include segmentation.h for calculateSegments and toStructSegment
// segmentation.h already includes rib_utils.h, so we don't need to include it separately
#include "config_generator/segmentation.h"

namespace ribtree {

void exponential_backoff(int retry_count);

class ThreadIdManager {
private:
    static thread_local int cached_thread_id_;
    static int max_thread_num_;

public:
    static void initialize(int max_threads) {
        max_thread_num_ = max_threads;
    }

    static int get_thread_id() {
        if (cached_thread_id_ == -1) {
            cached_thread_id_ = omp_get_thread_num();
            if (max_thread_num_ > 0 && cached_thread_id_ >= max_thread_num_) {
                cached_thread_id_ = cached_thread_id_ % max_thread_num_;
            }
        }
        return cached_thread_id_;
    }

    static void refresh_cache() {
        cached_thread_id_ = -1;
    }
};

thread_local int ThreadIdManager::cached_thread_id_ = -1;
int ThreadIdManager::max_thread_num_ = 0;

class ThreadLocalCounter {
private:
    struct alignas(64) ThreadCounter {
        std::atomic<uint32_t> count{0};
        char padding[60];
    };

    std::unique_ptr<ThreadCounter[]> thread_counters_;
    int thread_num_;
public:
    explicit ThreadLocalCounter(int thread_num)
        : thread_num_(thread_num) {
        thread_counters_ = std::make_unique<ThreadCounter[]>(thread_num);
    }

    void increment() {
        int slot = ThreadIdManager::get_thread_id();
        thread_counters_[slot].count.fetch_add(1, std::memory_order_relaxed);
    }

    void decrement() {
        int slot = ThreadIdManager::get_thread_id();
        thread_counters_[slot].count.fetch_sub(1, std::memory_order_relaxed);
    }

    bool is_zero() const {
        for (int i = 0; i < thread_num_; i++) {
            if (thread_counters_[i].count.load(std::memory_order_acquire) > 0) {
                return false;
            }
        }
        return true;
    }

    int64_t get_total_count() const {
        int64_t total = 0;
        for (int i = 0; i < thread_num_; i++) {
            total += thread_counters_[i].count.load(std::memory_order_acquire);
        }
        return total;
    }
};

// Thread-local timing utility for measuring wait operations
class ThreadLocalWaitTimingStats {
private:
    struct alignas(64) ThreadTimingStats {
        uint64_t total_wait_count{0};
        uint64_t total_wait_time_ns{0};
        uint64_t max_wait_time_ns{0};
        char padding[40]; // Ensure 64-byte alignment
    };

    std::unique_ptr<ThreadTimingStats[]> thread_stats_;
    int thread_num_;
    std::string name_;

public:
    explicit ThreadLocalWaitTimingStats(int thread_num, const std::string& name = "")
        : thread_num_(thread_num), name_(name) {
        thread_stats_ = std::make_unique<ThreadTimingStats[]>(thread_num);
    }

    void record_wait(uint64_t wait_time_ns) {
        int slot = ThreadIdManager::get_thread_id();
        assert(slot < thread_num_);
        auto& stats = thread_stats_[slot];
        stats.total_wait_count++;
        stats.total_wait_time_ns += wait_time_ns;
        if (wait_time_ns > stats.max_wait_time_ns) {
            stats.max_wait_time_ns = wait_time_ns;
        }
    }

    void print_stats() {
        uint64_t total_count = 0;
        uint64_t total_time = 0;
        uint64_t max_time = 0;
        int active_threads = 0;

        for (int i = 0; i < thread_num_; i++) {
            const auto& stats = thread_stats_[i];
            if (stats.total_wait_count > 0) {
                active_threads++;
            }
            total_count += stats.total_wait_count;
            total_time += stats.total_wait_time_ns;
            if (stats.max_wait_time_ns > max_time) {
                max_time = stats.max_wait_time_ns;
            }
        }

        if (total_count > 0) {
            double avg_time_us = static_cast<double>(total_time) / total_count / 1000.0;
            double max_time_us = static_cast<double>(max_time) / 1000.0;
            double total_time_us = static_cast<double>(total_time) / 1000.0;

            std::cout << "[" << name_ << "] Wait Stats: "
                      << "count=" << total_count << ", "
                      << "active_threads=" << active_threads << "/" << thread_num_ << ", "
                      << "avg_time=" << std::fixed << std::setprecision(2) << avg_time_us << "us, "
                      << "max_time=" << max_time_us << "us, "
                      << "total_time=" << total_time_us << "us" << std::endl;
        } else {
            std::cout << "[" << name_ << "] Wait Stats: count=0 (no waits recorded)" << std::endl;
        }
    }
};

ThreadLocalWaitTimingStats exponential_backoff_stats(84, "Exponential backoff");

inline void exponential_backoff(int retry_count) {
    auto backoff_start = std::chrono::high_resolution_clock::now();
    if (retry_count > 10) retry_count = 10;
    int backoff = (1 << retry_count);
    std::this_thread::sleep_for(std::chrono::microseconds(backoff));
    //std::this_thread::yield();
    auto backoff_end = std::chrono::high_resolution_clock::now();
    auto backoff_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(backoff_end - backoff_start).count();
    exponential_backoff_stats.record_wait(backoff_duration);
}

enum class InsertStatus {
    SUCCESS,
    FULL,
    SPLIT,
    OUT_OF_RANGE,
    ERROR
};

struct InsertResult {
    InsertStatus status;
    int box_index;
};

enum class DeleteStatus {
    SUCCESS,
    NOT_FOUND,
    SPLIT,
    OUT_OF_RANGE,
    ERROR
};

struct DeleteResult {
    DeleteStatus status;
    bool found;
};

enum class SearchStatus {
    SUCCESS,
    NOT_FOUND,
    SPLIT,
    OUT_OF_RANGE,
    ERROR
};

template <typename KeyType, typename ValueType>
struct SearchResult {
    SearchStatus status;
    ValueType value;
};

static constexpr int32_t BELOW_LOWER_BOUND = -1;
static constexpr int32_t ABOVE_UPPER_BOUND = -2;

struct tas_lock {
    std::atomic<bool> lock_ = {false};

    void lock() {
        for (;;) {
            if (!lock_.exchange(true, std::memory_order_acquire)) {
              break;
            }
            while (lock_.load(std::memory_order_relaxed)) {
              __builtin_ia32_pause();
            }
        }
    }

    void unlock() { lock_.store(false); }
};

template <typename KeyType, typename ValueType>
class Box {
   private:
    size_t maxSize = 0;
    size_t validSize = 0;
    size_t nearestEmptySlot = 0;
    bitset<maxKey> valid_flags;

#ifdef LOCK_SEARCH_SPIN_LOCK
    tas_lock lock_;
#endif

    mutable std::atomic<uint32_t> version_lock_{0};
    static constexpr uint32_t WRITE_LOCK_BIT = 0x80000000;
    static constexpr uint32_t VERSION_MASK   = 0x7FFFFFFF;

    alignas(64) array<KeyType, maxKey> keys;
    alignas(64) array<uint8_t, maxKey> keys_low;
    alignas(64) array<ValueType, maxKey> values;

    struct BoxSearchResult {
        uint8_t isUpdate; // 0 for update, 1 for insert, 2 for full
        size_t level;
        size_t slot;
    };

    enum BoxInsertResult : uint8_t {
        UPDATE = 0,
        INSERT = 1,
        FULL = 2
    };

    inline bool test_write_lock(uint32_t &version) const {
        version = version_lock_.load(std::memory_order_acquire);
        return (version & WRITE_LOCK_BIT) != 0;
    }

    inline bool version_changed(uint32_t old_version) const {
        uint32_t current = version_lock_.load(std::memory_order_acquire);
        return old_version != current;
    }

    inline bool try_acquire_write_lock() {
        uint32_t expected = version_lock_.load(std::memory_order_acquire);
        if (expected & WRITE_LOCK_BIT) {
            return false;
        }
        uint32_t desired = expected | WRITE_LOCK_BIT;
        return version_lock_.compare_exchange_strong(expected, desired,
                                                    std::memory_order_acq_rel,
                                                    std::memory_order_acquire);
    }

    inline void release_write_lock() {
        uint32_t current = version_lock_.load(std::memory_order_relaxed);
        uint32_t new_version = current + 1 - WRITE_LOCK_BIT;
        version_lock_.store(new_version, std::memory_order_release);
    }

    size_t findKeyIndex(KeyType key) const {
        uint8_t key_low = key & 0xFF;
        uint8_t target_low = ((key_low * 251) % 255) + 1;

        __m512i v_target_low = _mm512_set1_epi8(target_low);
        __m512i v_keys_low = _mm512_load_si512(reinterpret_cast<const __m512i*>(keys_low.data()));
        __mmask64 mask_low = _mm512_cmpeq_epi8_mask(v_keys_low, v_target_low);

        while (mask_low) {
            size_t candidate = __builtin_ctzll(mask_low);
            mask_low &= mask_low - 1;
            if (keys[candidate] == key && valid_flags[candidate] && candidate < maxSize) {
                return candidate;
            }
        }
        return maxKey;
    }

    void updateNearestEmptySlot() {
        for (size_t i = nearestEmptySlot; i < maxKey; i++) {
            if (!valid_flags[i]) {
                nearestEmptySlot = i;
                return;
            }
        }

        if (maxSize < maxKey) {
            nearestEmptySlot = maxSize;
        } else {
            nearestEmptySlot = maxKey;
        }
    }

    size_t avx512_filter_main_keys_optimized(KeyType key_low_bound,
                                             pair<KeyType, ValueType>* result_buffer,
                                             size_t max_results) const {
        size_t collected = 0;
        uint8_t target_low = static_cast<uint8_t>(key_low_bound & 0xFF);
        __m512i v_threshold = _mm512_set1_epi8(target_low);

        __m512i v_keys = _mm512_load_si512(reinterpret_cast<const __m512i*>(keys_low.data()));
        __mmask64 mask = _mm512_cmpge_epi8_mask(v_keys, v_threshold);

        while (mask && collected < max_results) {
            int pos = __builtin_ctzll(mask);
            result_buffer[collected++] = {keys[pos], values[pos]};
            mask &= mask - 1;
        }
        return collected;
    }

    size_t copyMainKeysWithLimit(pair<KeyType, ValueType>* result, size_t max_count) const {
        size_t to_copy = std::min(maxSize, max_count);
        for (size_t i = 0; i < to_copy; i++) {
            result[i] = {keys[i], values[i]};
        }
        return to_copy;
    }

   public:
    Box() {}

    Box(const Box& other)
        : maxSize(other.maxSize),
          validSize(other.validSize),
          nearestEmptySlot(other.nearestEmptySlot),
          valid_flags(other.valid_flags) {
        while (true) {
            uint32_t version_start;
            if (other.test_write_lock(version_start)) {
                std::this_thread::yield();
                continue;
            }
            keys = other.keys;
            keys_low = other.keys_low;
            values = other.values;
            if (!other.version_changed(version_start)) {
                break;
            }
        }
    }

    Box& operator=(const Box& other) {
        if (this != &other) {
            while (!try_acquire_write_lock()) {
                std::this_thread::yield();
            }
            while (true) {
                uint32_t version_start;
                if (other.test_write_lock(version_start)) {
                    std::this_thread::yield();
                    continue;
                }
                maxSize = other.maxSize;
                validSize = other.validSize;
                nearestEmptySlot = other.nearestEmptySlot;
                valid_flags = other.valid_flags;
                keys = other.keys;
                keys_low = other.keys_low;
                values = other.values;
                if (!other.version_changed(version_start)) {
                    break;
                }
            }
            release_write_lock();
        }
        return *this;
    }

    Box(Box&& other) noexcept
        : maxSize(other.maxSize),
        validSize(other.validSize),
        nearestEmptySlot(other.nearestEmptySlot),
        valid_flags(other.valid_flags),
        keys(std::move(other.keys)),
        keys_low(std::move(other.keys_low)),
        values(std::move(other.values)) {

        version_lock_.store(other.version_lock_.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
    }

    Box& operator=(Box&& other) noexcept {
        if (this != &other) {
            while (!try_acquire_write_lock()) {
                std::this_thread::yield();
            }

            maxSize = other.maxSize;
            validSize = other.validSize;
            nearestEmptySlot = other.nearestEmptySlot;
            valid_flags = other.valid_flags;
            keys = std::move(other.keys);
            keys_low = std::move(other.keys_low);
            values = std::move(other.values);

            release_write_lock();
        }
        return *this;
    }
    ~Box() = default;

    bool hasEmptySlots() const {
        while (true) {
            uint32_t version_start;
            if (test_write_lock(version_start)) {
                std::this_thread::yield();
                continue;
            }
            bool result = nearestEmptySlot < maxKey;
            if (!version_changed(version_start)) {
                return result;
            }
        }
    }

    size_t getTotalCount() const {
        return maxSize;
    }

    DeleteResult deleteKey(KeyType key) {
        int retry_count = 0;

    retry_delete:
        uint32_t expected = version_lock_.load(std::memory_order_acquire);
        if (expected & WRITE_LOCK_BIT) {
            exponential_backoff(retry_count++);
            goto retry_delete;
        }

        uint32_t desired = expected | WRITE_LOCK_BIT;
        if (!version_lock_.compare_exchange_strong(expected, desired,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_delete;
        }

        size_t index = findKeyIndex(key);
        bool found = false;
        if (index != maxKey) {
            valid_flags[index] = 0;
            validSize--;
            nearestEmptySlot = index < nearestEmptySlot ? index : nearestEmptySlot;
            found = true;
        }

        uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
        version_lock_.store(new_version, std::memory_order_release);
        return {found ? DeleteStatus::SUCCESS : DeleteStatus::NOT_FOUND, found};
    }

    InsertResult updateValue(size_t index, ValueType value) {
        int retry_count = 0;

    retry_write:
        uint32_t expected = version_lock_.load(std::memory_order_acquire);
        if (expected & WRITE_LOCK_BIT) {
            exponential_backoff(retry_count++);
            goto retry_write;
        }

        uint32_t desired = expected | WRITE_LOCK_BIT;
        if (!version_lock_.compare_exchange_strong(expected, desired,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_write;
        }
        
        values[index] = value;
        uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
        version_lock_.store(new_version, std::memory_order_release);
        return {InsertStatus::SUCCESS, -1};
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        int retry_count = 0;

    retry_write:
        uint32_t expected = version_lock_.load(std::memory_order_acquire);
        if (expected & WRITE_LOCK_BIT) {
            exponential_backoff(retry_count++);
            goto retry_write;
        }

        uint32_t desired = expected | WRITE_LOCK_BIT;
        if (!version_lock_.compare_exchange_strong(expected, desired,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_write;
        }

        if (nearestEmptySlot >= maxKey) {
            uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
            version_lock_.store(new_version, std::memory_order_release);
            return {InsertStatus::FULL, -1};
        }

        keys[nearestEmptySlot] = key;
        values[nearestEmptySlot] = value;
        uint8_t key_low = key & 0xFF;
        keys_low[nearestEmptySlot] = ((key_low * 251) % 255) + 1;
        valid_flags[nearestEmptySlot] = true;
        validSize++;
        if (nearestEmptySlot == maxSize) {
            maxSize++;
        }
        updateNearestEmptySlot();

        uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
        version_lock_.store(new_version, std::memory_order_release);
        return {InsertStatus::SUCCESS, -1};
    }

    size_t searchUpdateKey(KeyType key) {
        int retry_count = 0;

    retry_read:
        uint32_t start_version = version_lock_.load(std::memory_order_acquire);
        if (start_version & WRITE_LOCK_BIT) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }

        size_t index = findKeyIndex(key);
        if (start_version != version_lock_.load(std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }

        if (index != maxKey) {
            return index;
        }
        return maxKey;
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        // int retry_count = 0;  // Unused, commented out to avoid warning
#ifdef LOCK_SEARCH
    retry_read:
        uint32_t start_version = version_lock_.load(std::memory_order_acquire);
        if (start_version & WRITE_LOCK_BIT) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }
#endif

#ifdef LOCK_SEARCH_SPIN_LOCK
        lock_.lock();
#endif

        size_t index = findKeyIndex(key);
        ValueType result_value = -1;
        SearchStatus status = SearchStatus::NOT_FOUND;

        if (index != maxKey) {
            result_value = values[index];
            status = SearchStatus::SUCCESS;
        }
#ifdef LOCK_SEARCH
        if (start_version != version_lock_.load(std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }
#endif
#ifdef LOCK_SEARCH_SPIN_LOCK
        lock_.unlock();
#endif
        return {status, result_value};
    }

    size_t getmaxSize() const {
        return maxSize;
    }

    vector<pair<KeyType, ValueType>> getEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        for (size_t i = 0; i < maxSize; i++) {
            entries.push_back({keys[i], values[i]});
        }
        return entries;
    }

    void getEntriesInPlace(vector<pair<KeyType, ValueType>>* entries) const {
        size_t start_size = entries->size();
        entries->resize(start_size + maxSize);
        for (size_t i = 0; i < maxSize; i++) {
            (*entries)[start_size + i] = {keys[i], values[i]};
        }
    }

    void getEntriesInPlace(vector<pair<KeyType, ValueType>>* entries, size_t start_pos) const {
        // Don't resize - assume the vector is already large enough
        // Assign main entries directly
        for (size_t i = 0; i < maxSize; i++) {
            (*entries)[start_pos + i] = {keys[i], values[i]};
        }
    }
};


template <typename KeyType, typename ValueType>
class Segment {
private:
    size_t box_key_range;
    size_t num_threads;

    mutable ThreadLocalCounter operation_counter_;
    std::atomic_flag splitting_flag_ = ATOMIC_FLAG_INIT;

    size_t logical_box_count;
    size_t physical_box_count;

    Box<KeyType, ValueType>* first_box_ptr;
    size_t active_box_count;

public:
    KeyType lower_bound;
    KeyType upper_bound;
    std::deque<std::atomic<uint8_t>> logical_box_write_positions;
    static constexpr size_t PHYSICAL_BOXES_PER_LOGICAL = 1 + overflowCapacity;
    int numBoxes;
    mutable std::atomic<bool> splitting_{false};
    std::atomic<bool> is_splitting_{false};

    Segment(KeyType lower, KeyType upper, size_t box_range, int thread_num)
        : box_key_range(box_range), num_threads(thread_num),
          operation_counter_(thread_num), lower_bound(lower), upper_bound(upper) {
        
        size_t total = upper - lower + 1;
        logical_box_count = total / box_range;
        if (total % box_range != 0) logical_box_count++;
        
        physical_box_count = logical_box_count * PHYSICAL_BOXES_PER_LOGICAL;
        numBoxes = logical_box_count;
        active_box_count = logical_box_count;
        
        first_box_ptr = new Box<KeyType, ValueType>[physical_box_count];
        
        for (size_t i = 0; i < physical_box_count; i++) {
            new (&first_box_ptr[i]) Box<KeyType, ValueType>();
        }
        
        logical_box_write_positions.resize(logical_box_count);
        for (size_t i = 0; i < logical_box_count; i++) {
            logical_box_write_positions[i].store(0, std::memory_order_relaxed);
        }
    }

    Segment(KeyType lower, KeyType upper, size_t box_range, int thread_num,
        Box<KeyType, ValueType>* existing_boxes, size_t logical_count, bool /* take_ownership */ = false)
        : box_key_range(box_range), num_threads(thread_num),
        operation_counter_(thread_num), lower_bound(lower), upper_bound(upper) {
        
        logical_box_count = logical_count;
        physical_box_count = logical_count * PHYSICAL_BOXES_PER_LOGICAL;
        numBoxes = logical_count;
        active_box_count = logical_count;
        first_box_ptr = existing_boxes;
        
        logical_box_write_positions.resize(logical_box_count);
        for (size_t i = 0; i < logical_box_count; i++) {
            logical_box_write_positions[i].store(0, std::memory_order_relaxed);
        }
    }

    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;

    Segment(Segment&& other) noexcept
        : lower_bound(other.lower_bound),
          upper_bound(other.upper_bound),
          box_key_range(other.box_key_range),
          logical_box_count(other.logical_box_count),
          physical_box_count(other.physical_box_count),
          numBoxes(other.numBoxes),
          active_box_count(other.active_box_count),
          operation_counter_(std::move(other.operation_counter_)),
          first_box_ptr(other.first_box_ptr),
          logical_box_write_positions(std::move(other.logical_box_write_positions)) {
    }

    Segment& operator=(Segment&& other) noexcept {
        if (this != &other) {
            lower_bound = other.lower_bound;
            upper_bound = other.upper_bound;
            box_key_range = other.box_key_range;
            logical_box_count = other.logical_box_count;
            physical_box_count = other.physical_box_count;
            numBoxes = other.numBoxes;
            active_box_count = other.active_box_count;
            operation_counter_ = std::move(other.operation_counter_);
            first_box_ptr = other.first_box_ptr;
            logical_box_write_positions = std::move(other.logical_box_write_positions);
        }
        return *this;
    }

    ~Segment() {}

    size_t getLogicalBoxIndex(KeyType key) const {
        return (key - lower_bound) / box_key_range;
    }

    size_t getPhysicalBoxIndex(size_t logical_box_index, uint8_t position_offset) const {
        return logical_box_index * PHYSICAL_BOXES_PER_LOGICAL + position_offset;
    }


    bool try_mark_for_splitting() {
        bool expected = false;
        return is_splitting_.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
    }

    void unmark_splitting() {
        is_splitting_.store(false, std::memory_order_release);
        is_splitting_.notify_all();
    }

    bool is_currently_splitting() const {
        return is_splitting_.load(std::memory_order_acquire);
    }

    void wait_for_split_completion() const {
        is_splitting_.wait(true, std::memory_order_acquire);
    }

    bool enter() {
        if (splitting_.load(std::memory_order_acquire)) {
            return false;
        }
        operation_counter_.increment();
        if (splitting_.load(std::memory_order_acquire)) {
            operation_counter_.decrement();
            return false;
        }
        return true;
    }

    void leave() {
        operation_counter_.decrement();
    }

    void wait_for_operations() {
        splitting_.store(true, std::memory_order_release);
        while (!operation_counter_.is_zero()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        if (!enter()) {
            return {InsertStatus::SPLIT, -1};
        }
        
        if (key < lower_bound || key >= upper_bound) {
            leave();
            return {InsertStatus::OUT_OF_RANGE, -1};
        }

        size_t logical_box_index = getLogicalBoxIndex(key);
        uint8_t current_position = logical_box_write_positions[logical_box_index].load(std::memory_order_acquire);

        while (current_position < PHYSICAL_BOXES_PER_LOGICAL) {
            size_t physical_box_index = getPhysicalBoxIndex(logical_box_index, current_position);

            InsertResult result = (first_box_ptr + physical_box_index)->insertKeyValue(key, value);

            if (result.status == InsertStatus::SUCCESS) {
                leave();
                return {InsertStatus::SUCCESS, static_cast<int>(logical_box_index)};
            }

            if (result.status == InsertStatus::FULL) {
                uint8_t expected = current_position;
                if (logical_box_write_positions[logical_box_index].compare_exchange_weak(
                    expected, current_position + 1,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                    current_position++;
                } else {
                    current_position = logical_box_write_positions[logical_box_index].load(std::memory_order_acquire);
                }
            } else {
                leave();
                return {result.status, static_cast<int>(logical_box_index)};
            }
        }

        leave();
        return {InsertStatus::FULL, static_cast<int>(logical_box_index)};
    }

    DeleteResult deleteKey(KeyType key) {
        enter();
        if (key < lower_bound || key > upper_bound) {
            leave();
            return {DeleteStatus::NOT_FOUND, false};
        }

        size_t logical_box_index = getLogicalBoxIndex(key);
        uint8_t max_position = logical_box_write_positions[logical_box_index].load(std::memory_order_acquire);
        
        for (uint8_t pos = 0; pos <= max_position && pos < PHYSICAL_BOXES_PER_LOGICAL; pos++) {
            size_t physical_box_index = getPhysicalBoxIndex(logical_box_index, pos);
            DeleteResult result = (first_box_ptr + physical_box_index)->deleteKey(key);
            if (result.found) {
                leave();
                return {DeleteStatus::SUCCESS, true};
            }
        }
        
        leave();
        return {DeleteStatus::NOT_FOUND, false};
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        size_t logical_box_index = getLogicalBoxIndex(key);
        uint8_t max_position = logical_box_write_positions[logical_box_index].load(std::memory_order_acquire);
        
        for (uint8_t pos = 0; pos <= max_position && pos < PHYSICAL_BOXES_PER_LOGICAL; pos++) {
            size_t physical_box_index = getPhysicalBoxIndex(logical_box_index, pos);
            
            SearchResult<KeyType, ValueType> result = (first_box_ptr + physical_box_index)->searchKey(key);
            if (result.status == SearchStatus::SUCCESS) {
                return result;
            }
        }
        
        return {SearchStatus::NOT_FOUND, -1};
    }


    KeyType getBoxLower(int box_index) const {
        return lower_bound + box_index * box_key_range;
    }

    KeyType getBoxUpper(int box_index) const {
        KeyType candidate = lower_bound + (box_index + 1) * box_key_range;
        return candidate > upper_bound ? upper_bound : candidate;
    }

    KeyType getLowerBound() const { return lower_bound; }
    KeyType getUpperBound() const { return upper_bound; }
    size_t getBoxKeyRange() const { return box_key_range; }
    size_t getBoxCount() const { return logical_box_count; }

    vector<pair<KeyType, ValueType>> getAllEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        for (size_t logical_idx = 0; logical_idx < logical_box_count; logical_idx++) {
            uint8_t max_position = logical_box_write_positions[logical_idx].load(std::memory_order_acquire);
            for (uint8_t pos = 0; pos <= max_position && pos < PHYSICAL_BOXES_PER_LOGICAL; pos++) {
                size_t physical_box_index = getPhysicalBoxIndex(logical_idx, pos);
                vector<pair<KeyType, ValueType>> box_entries = (first_box_ptr + physical_box_index)->getEntries();
                entries.insert(entries.end(), box_entries.begin(), box_entries.end());
            }
        }
        return entries;
    }

    Box<KeyType, ValueType>* getBoxPtr(size_t index) {
        size_t physical_index = getPhysicalBoxIndex(index, 0);
        return first_box_ptr + physical_index;
    }

    vector<pair<KeyType, ValueType>> prepare_for_split_stage1(int32_t merge_start, int32_t merge_end) {
        vector<pair<KeyType, ValueType>> mergedEntries;

        int32_t start_logical_box = std::max(0, merge_start);
        int32_t end_logical_box = std::min(merge_end, static_cast<int32_t>(logical_box_count) - 1);
        if (start_logical_box > end_logical_box) return mergedEntries;

        std::vector<size_t> box_start_positions(static_cast<size_t>(end_logical_box - start_logical_box + 2), 0);
        size_t total_size = 0;

        // Calculate size of each logical box
        for (int logical_idx = start_logical_box; logical_idx <= end_logical_box; logical_idx++) {
            size_t logical_box_size = 0;
            uint8_t max_position = logical_box_write_positions[logical_idx].load(std::memory_order_acquire);

            for (uint8_t pos = 0; pos <= max_position && pos < PHYSICAL_BOXES_PER_LOGICAL; pos++) {
                size_t physical_box_index = getPhysicalBoxIndex(logical_idx, pos);
                logical_box_size += (first_box_ptr + physical_box_index)->getTotalCount();
            }

            box_start_positions[(logical_idx - start_logical_box) + 1] =
                box_start_positions[(logical_idx - start_logical_box)] + logical_box_size;
            total_size += logical_box_size;
        }

        mergedEntries.resize(total_size);

        // Extract data from logical boxes
        for (int logical_idx = start_logical_box; logical_idx <= end_logical_box; logical_idx++) {
            size_t start_pos = box_start_positions[logical_idx - start_logical_box];
            size_t current_pos = start_pos;
            uint8_t max_position = logical_box_write_positions[logical_idx].load(std::memory_order_acquire);

            for (uint8_t pos = 0; pos <= max_position && pos < PHYSICAL_BOXES_PER_LOGICAL; pos++) {
                size_t physical_box_index = getPhysicalBoxIndex(logical_idx, pos);
                (first_box_ptr + physical_box_index)->getEntriesInPlace(&mergedEntries, current_pos);
                current_pos += (first_box_ptr + physical_box_index)->getTotalCount();
            }
        }

        return mergedEntries;
    }

    void resetBoxes(Box<KeyType, ValueType>* new_first_box, size_t new_logical_count) {
        first_box_ptr = new_first_box;
        logical_box_count = new_logical_count;
        physical_box_count = new_logical_count * PHYSICAL_BOXES_PER_LOGICAL;
        active_box_count = new_logical_count;
        numBoxes = new_logical_count;

        logical_box_write_positions.resize(logical_box_count);
        for (size_t i = 0; i < logical_box_count; i++) {
            logical_box_write_positions[i].store(0, std::memory_order_relaxed);
        }
    }

    void populateSegmentsSerial(const vector<pair<KeyType, ValueType>>& mergedEntries,
                        vector<Segment<KeyType, ValueType>*>& new_segments) {
        size_t current_seg = 0;
        for (const auto& entry : mergedEntries) {
            KeyType key = entry.first;
            ValueType value = entry.second;
            while (current_seg < new_segments.size() && key >= new_segments[current_seg]->getUpperBound()) {
                current_seg++;
            }
            if (current_seg < new_segments.size()) {
                new_segments[current_seg]->insertKeyValue(key, value);
            }
        }
    }

    std::vector<std::pair<KeyType, Segment<KeyType, ValueType>*>> splitSegment(int box_index, double overflowThreshold, double underflowThreshold) {
        // Wait for all ongoing operations to complete
        wait_for_operations();

        // Save original write positions before split
        size_t numLogicalBoxes = logical_box_count;
        std::vector<uint8_t> original_write_positions(numLogicalBoxes);
        for (size_t i = 0; i < numLogicalBoxes; i++) {
            original_write_positions[i] = logical_box_write_positions[i].load(std::memory_order_acquire);
        }

        // Calculate the range of logical boxes to process
        const int NUM_BOXES_TO_LOOK = 3;
        int left_count = NUM_BOXES_TO_LOOK;
        int right_count = NUM_BOXES_TO_LOOK;
        int merge_start = std::max(0, box_index - left_count);
        int merge_end = std::min(static_cast<int>(numLogicalBoxes) - 1, box_index + right_count);

        // Extract entries from the logical boxes to be merged
        auto mergedEntries = prepare_for_split_stage1(merge_start, merge_end);

        // Early return if no entries to process
        if (mergedEntries.empty()) {
            unmark_splitting();
            return {};
        }

        // Extract keys for segmentation calculation
        vector<KeyType> keys;
        keys.reserve(mergedEntries.size());
        for (const auto& entry : mergedEntries) {
            keys.push_back(entry.first);
        }

        // Sort keys before calling calculateSegments
        std::sort(keys.begin(), keys.end());

        // Calculate new segment boundaries
        KeyType merged_lower = getBoxLower(merge_start);
        KeyType merged_upper = getBoxUpper(merge_end);

        std::vector<seg::keySegment<KeyType>> keysegments =
            calculateSegments(keys, overflowThreshold, underflowThreshold, 15, merged_lower, merged_upper);

        std::vector<ribtree::StructSegment<KeyType>> final_segments = toStructSegment(keysegments);

        // Create the middle merged segments
        std::vector<Segment<KeyType, ValueType>*> merged_segments;
        merged_segments.reserve(final_segments.size());
        for (const auto& struct_seg : final_segments) {
            auto* new_seg = new Segment<KeyType, ValueType>(
                struct_seg.seg_lower,
                struct_seg.seg_upper,
                struct_seg.box_range,
                num_threads
            );
            merged_segments.push_back(new_seg);
        }

        // Populate merged segments with data
        populateSegmentsSerial(mergedEntries, merged_segments);

        // Prepare containers for final segment arrangement
        std::vector<Segment<KeyType, ValueType>*> new_segments;
        std::vector<KeyType> new_segment_start_keys;

        // Determine if we need left and right segments
        bool has_left = (merge_start > 0);
        bool has_right = (merge_end < static_cast<int>(numLogicalBoxes) - 1);

        Box<KeyType, ValueType>* original_boxes = getBoxPtr(0);
        KeyType original_upper_bound = upper_bound;

        Segment<KeyType, ValueType>* left_segment = nullptr;
        Segment<KeyType, ValueType>* right_segment = nullptr;

        if (has_left) {
            left_segment = this;
            left_segment->upper_bound = getBoxUpper(merge_start - 1);
            left_segment->resetBoxes(original_boxes, merge_start);

            for (int i = 0; i < merge_start; i++) {
                left_segment->logical_box_write_positions[i].store(
                    original_write_positions[i], std::memory_order_relaxed);
            }
        }

        if (has_right) {
            if (has_left) {
                KeyType right_lower = getBoxLower(merge_end + 1);
                Box<KeyType, ValueType>* right_boxes = original_boxes +
                    ((merge_end + 1) * PHYSICAL_BOXES_PER_LOGICAL);
                size_t right_logical_box_count = numLogicalBoxes - (merge_end + 1);

                right_segment = new Segment<KeyType, ValueType>(
                    right_lower, original_upper_bound, box_key_range, num_threads,
                    right_boxes, right_logical_box_count
                );

                for (size_t i = 0; i < right_logical_box_count; i++) {
                    right_segment->logical_box_write_positions[i].store(
                        original_write_positions[merge_end + 1 + i], std::memory_order_relaxed);
                }
            } else {
                right_segment = this;
                KeyType right_lower = getBoxLower(merge_end + 1);
                Box<KeyType, ValueType>* right_boxes = original_boxes +
                    ((merge_end + 1) * PHYSICAL_BOXES_PER_LOGICAL);
                size_t right_logical_box_count = numLogicalBoxes - (merge_end + 1);

                right_segment->lower_bound = right_lower;
                right_segment->resetBoxes(right_boxes, right_logical_box_count);

                for (size_t i = 0; i < right_logical_box_count; i++) {
                    right_segment->logical_box_write_positions[i].store(
                        original_write_positions[merge_end + 1 + i], std::memory_order_relaxed);
                }
            }
        }

        // Assemble new_segments in logical order: left → merged → right
        if (has_left) {
            new_segments.push_back(left_segment);
            new_segment_start_keys.push_back(left_segment->getLowerBound());
        }

        for (size_t i = 0; i < merged_segments.size(); i++) {
            new_segments.push_back(merged_segments[i]);
            new_segment_start_keys.push_back(merged_segments[i]->getLowerBound());
        }

        if (has_right) {
            new_segments.push_back(right_segment);
            new_segment_start_keys.push_back(right_segment->getLowerBound());
        }

        // Return pairs of (start_key, segment_ptr)
        std::vector<std::pair<KeyType, Segment<KeyType, ValueType>*>> result;
        for (size_t i = 0; i < new_segments.size(); i++) {
            result.push_back({new_segment_start_keys[i], new_segments[i]});
        }

        return result;
    }

};

} // namespace ribtree


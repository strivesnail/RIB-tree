#pragma once

#include <cassert>
#include <cstring>
#include <atomic>
#include <immintrin.h>
#include <sched.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <omp.h>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <functional>
#include "rib_segment.h"

// B+ Tree split threshold configuration
// This threshold applies to all levels (leaf nodes and inner nodes)
// When a node's count reaches this threshold, it will be split
#ifndef SPLIT_THRESHOLD
#define SPLIT_THRESHOLD 32
#endif

namespace ribtree {

enum class PageType : uint8_t { BTreeInner=1, BTreeLeaf=2 };

static const uint64_t pageSize=4*1024;

struct OptLock {
  std::atomic<uint64_t> typeVersionLockObsolete{0b100};

  bool isLocked(uint64_t version) {
    return ((version & 0b10) == 0b10);
  }

  uint64_t readLockOrRestart(bool &needRestart) {
    uint64_t version;
    version = typeVersionLockObsolete.load();
    if (isLocked(version) || isObsolete(version)) {
      _mm_pause();
      needRestart = true;
    }
    return version;
  }

  void writeLockOrRestart(bool &needRestart) {
    uint64_t version;
    version = readLockOrRestart(needRestart);
    if (needRestart) return;

    upgradeToWriteLockOrRestart(version, needRestart);
    if (needRestart) return;
  }

  void upgradeToWriteLockOrRestart(uint64_t &version, bool &needRestart) {
    if (typeVersionLockObsolete.compare_exchange_strong(version, version + 0b10)) {
      version = version + 0b10;
    } else {
      _mm_pause();
      needRestart = true;
    }
  }

  void writeUnlock() {
    typeVersionLockObsolete.fetch_add(0b10);
  }

  bool isObsolete(uint64_t version) {
    return (version & 1) == 1;
  }

  void checkOrRestart(uint64_t startRead, bool &needRestart) const {
    readUnlockOrRestart(startRead, needRestart);
  }

  void readUnlockOrRestart(uint64_t startRead, bool &needRestart) const {
    needRestart = (startRead != typeVersionLockObsolete.load());
  }

  void writeUnlockObsolete() {
    typeVersionLockObsolete.fetch_add(0b11);
  }
};

struct NodeBase : public OptLock{
  PageType type;
  uint16_t count;
};

struct BTreeLeafBase : public NodeBase {
   static const PageType typeMarker=PageType::BTreeLeaf;
};

template<class Key,class Value>
struct BTreeLeaf : public BTreeLeafBase {
   static const uint64_t maxEntries = 128;  // Maximum capacity: 128 segments (increased from 32 to prevent overflow)
   // Note: bulkLoadMax is not a threshold, but a building standard for bulk load
   // During bulk load, we build nodes with exactly 16 segments per leaf
   uint64_t bulkLoadMax = 16;  // Building standard: 16 segments per leaf during bulk load
   bool isBulkLoading = false;  // Flag to indicate bulk load mode (for tracking purposes)

   Key keys[maxEntries];  // Keys for locating segments (segment start keys)
   ribtree::Segment<Key, Value>* segments[maxEntries];  // Segment pointers

   BTreeLeaf() {
      count=0;
      type=typeMarker;
      isBulkLoading = false;
      for (size_t i = 0; i < maxEntries; i++) {
         segments[i] = nullptr;
      }
   }

   bool isFull() { 
      // Note: During bulk load, nodes are built with exactly 16 segments per leaf
      // So isFull() should never be true during bulk load
      if (isBulkLoading) {
        return false;  // Never full during bulk load
      }
      // Normal mode: split when reaching SPLIT_THRESHOLD entries
      return count >= SPLIT_THRESHOLD;
   };

   unsigned lowerBound(Key k) {
      unsigned lower=0;
      unsigned upper=count;
      do {
         unsigned mid=((upper-lower)/2)+lower;
         if (k<keys[mid]) {
            upper=mid;
         } else if (k>keys[mid]) {
            lower=mid+1;
         } else {
            return mid;
         }
      } while (lower<upper);
      return lower;
   }

   unsigned lowerBoundBF(Key k) {
      auto base=keys;
      unsigned n=count;
      while (n>1) {
         const unsigned half=n/2;
         base=(base[half]<k)?(base+half):base;
         n-=half;
      }
      return (*base<k)+base-keys;
   }

   // Find segment containing key k using binary search
   unsigned findSegment(Key k) {
      if (count == 0) return count;
      
      unsigned pos = lowerBound(k);
      
      // Case 1: pos < count
      if (pos < count && segments[pos]) {
         if (k >= segments[pos]->getLowerBound() && 
             k < segments[pos]->getUpperBound()) {
            return pos;
         }
      }
      
      // Case 2: Check previous segment
      if (pos > 0) {
         unsigned prevPos = pos - 1;
         if (segments[prevPos] && 
             k >= segments[prevPos]->getLowerBound() && 
             k < segments[prevPos]->getUpperBound()) {
            return prevPos;
         }
      }
      
      // Case 3: pos == count
      if (pos == count && count > 0) {
         unsigned lastPos = count - 1;
         if (segments[lastPos] && 
             k >= segments[lastPos]->getLowerBound() && 
             k < segments[lastPos]->getUpperBound()) {
            return lastPos;
         }
      }
      
      // Case 4: pos == 0
      if (pos == 0 && count > 0 && segments[0]) {
         if (k >= segments[0]->getLowerBound() && 
             k < segments[0]->getUpperBound()) {
            return 0;
         }
      }
      
      return count;
   }

  void insert(Key k, ribtree::Segment<Key, Value>* seg) {
    assert(seg != nullptr);
    // Note: In normal mode, node splitting will handle overflow
    // During bulk load, nodes are built with exactly 16 segments, so this should not overflow
    if (count) {
      unsigned pos=lowerBound(k);
      if ((pos<count) && (keys[pos]==k)) {
        segments[pos] = seg;
	return;
      }
      memmove(keys+pos+1,keys+pos,sizeof(Key)*(count-pos));
      memmove(segments+pos+1,segments+pos,sizeof(ribtree::Segment<Key, Value>*)*(count-pos));
      keys[pos]=k;
      segments[pos]=seg;
    } else {
      keys[0]=k;
      segments[0]=seg;
    }
    count++;
  }

   BTreeLeaf* split(Key& sep) {
      BTreeLeaf* newLeaf = new BTreeLeaf();
      newLeaf->count = count-(count/2);
      count = count-newLeaf->count;
      memcpy(newLeaf->keys, keys+count, sizeof(Key)*newLeaf->count);
      memcpy(newLeaf->segments, segments+count, sizeof(ribtree::Segment<Key, Value>*)*newLeaf->count);
      // Separator is the first key of the new node
      sep = keys[count];
      // Clear pointers in old node to prevent dangling pointers
      for (unsigned i = 0; i < newLeaf->count; i++) {
         segments[count + i] = nullptr;
         keys[count + i] = Key{};
      }
      return newLeaf;
   }
};

struct BTreeInnerBase : public NodeBase {
   static const PageType typeMarker=PageType::BTreeInner;
};

template<class Key>
struct BTreeInner : public BTreeInnerBase {
   static const uint64_t maxEntries = 128;  // Maximum capacity: 128 children (increased from 32 to prevent overflow)
   // Note: bulkLoadMax is not a threshold, but a building standard for bulk load
   // During bulk load, we build nodes with exactly 16 children per inner node
   uint64_t bulkLoadMax = 16;  // Building standard: 16 children per inner node during bulk load
   bool isBulkLoading = false;  // Flag to indicate bulk load mode (for tracking purposes)

   NodeBase* children[maxEntries];
   Key keys[maxEntries];

   BTreeInner() {
      count=0;
      type=typeMarker;
      isBulkLoading = false;
   }

   bool isFull() { 
      // Note: During bulk load, nodes are built with exactly 16 children per inner node
      // So isFull() should never be true during bulk load
      if (isBulkLoading) {
        return false;  // Never full during bulk load
      }
      // Normal mode: split when reaching SPLIT_THRESHOLD keys (SPLIT_THRESHOLD+1 children)
      return count >= SPLIT_THRESHOLD;
   };

   unsigned lowerBoundBF(Key k) {
      auto base=keys;
      unsigned n=count;
      while (n>1) {
         const unsigned half=n/2;
         base=(base[half]<k)?(base+half):base;
         n-=half;
      }
      return (*base<k)+base-keys;
   }

   unsigned lowerBound(Key k) {
      unsigned lower=0;
      unsigned upper=count;
      do {
         unsigned mid=((upper-lower)/2)+lower;
         if (k<keys[mid]) {
            upper=mid;
         } else if (k>keys[mid]) {
            lower=mid+1;
         } else {
            return mid;
         }
      } while (lower<upper);
      return lower;
   }

   BTreeInner* split(Key& sep) {
      BTreeInner* newInner=new BTreeInner();
      newInner->count=count-(count/2);
      count=count-newInner->count-1;
      sep=keys[count];
      memcpy(newInner->keys,keys+count+1,sizeof(Key)*(newInner->count+1));
      memcpy(newInner->children,children+count+1,sizeof(NodeBase*)*(newInner->count+1));
      // Clear pointers in old node (keys[count] is separator, clear from count+1)
      for (unsigned i = 0; i <= newInner->count; i++) {
         children[count + 1 + i] = nullptr;
         keys[count + 1 + i] = Key{};  // Clear key as well for safety
      }
      return newInner;
   }

   void insert(Key k,NodeBase* child) {
      assert(count<maxEntries-1);
      unsigned pos=lowerBound(k);
      memmove(keys+pos+1,keys+pos,sizeof(Key)*(count-pos+1));
      memmove(children+pos+1,children+pos,sizeof(NodeBase*)*(count-pos+1));
      keys[pos]=k;
      children[pos]=child;
      // Swap to maintain correct B+ tree structure: left subtree < k, right subtree >= k
      std::swap(children[pos],children[pos+1]);
      count++;
   }

};


template<class Key,class Value>
struct BTree {
  std::atomic<NodeBase*> root;
  std::atomic<bool> isBulkLoading{true};  // Automatically true during bulk load, false after

   int thread_num;  // Number of threads for Segment creation
   double underflowThreshold;  // Threshold for segment underflow
   double overflowThreshold;   // Threshold for segment overflow

   // PathEntry for tracking path from root to leaf during insertion
   struct PathEntry {
       NodeBase* node;
       uint64_t version;
       bool is_inner;
   };

   // Promoted segments to root: data structures and lock
   struct PromotedSegmentsLock {
     std::atomic<uint64_t> typeVersionLockObsolete{0b100};
     
     uint64_t readLockOrRestart(bool &needRestart) {
       uint64_t version = typeVersionLockObsolete.load();
       if (((version & 0b10) == 0b10) || ((version & 1) == 1)) {
         _mm_pause();
         needRestart = true;
       }
       return version;
     }
     
     void writeLockOrRestart(bool &needRestart) {
       uint64_t version = readLockOrRestart(needRestart);
       if (needRestart) return;
       if (typeVersionLockObsolete.compare_exchange_strong(version, version + 0b10)) {
         version = version + 0b10;
       } else {
         _mm_pause();
         needRestart = true;
       }
     }
     
     void writeUnlock() {
       typeVersionLockObsolete.fetch_add(0b10);
     }
     
     void readUnlockOrRestart(uint64_t startRead, bool &needRestart) const {
       needRestart = (startRead != typeVersionLockObsolete.load());
     }
   };
   
   struct SegmentIndexRange {
     size_t start;
     size_t end;
   };
   
   // Promoted segments data structures
   static const size_t MAX_PROMOTED_SEGMENTS = SPLIT_THRESHOLD;  // Maximum 32 promoted segments
   PromotedSegmentsLock promoted_lock;  // Read-write version lock for promoted segments arrays
   std::vector<uint8_t> promoted_bitmap;  // Bitmap: 1 if child has promoted segments, 0 otherwise
   std::vector<ribtree::Segment<Key, Value>*> promoted_segments;  // Array of promoted segments
   std::vector<SegmentIndexRange> promoted_index;  // Index array: range for each child node
   size_t promoted_segment_count;  // Current count of promoted segments (max index + 1)
   size_t promoted_child_count;  // Number of child nodes (leaf nodes at root level)
   
   // Map to store keys_count for each segment (from config file)
   std::unordered_map<ribtree::Segment<Key, Value>*, size_t> segment_keys_count_map;

   BTree(int num_threads = 1, double uThreshold = 0.5, double oThreshold = 0.1) 
      : thread_num(num_threads), underflowThreshold(uThreshold), overflowThreshold(oThreshold),
        promoted_segment_count(0), promoted_child_count(0) {
      // Initialize ThreadIdManager for Segment operations
      ribtree::ThreadIdManager::initialize(num_threads);
      
      auto* leaf = new BTreeLeaf<Key,Value>();
      leaf->isBulkLoading = false;  // Start in normal mode (SPLIT_THRESHOLD threshold)
      root = leaf;
      isBulkLoading.store(false);  // Initially in normal mode
      
      // Initialize promoted segments arrays
      promoted_segments.resize(MAX_PROMOTED_SEGMENTS, nullptr);
   }

  // Bulk load function: load segments from config file, build tree, then insert key-value pairs
  void bulkLoad(const std::string& config_file, const std::vector<std::pair<Key, Value>>& data) {
      std::cout << "[BULK LOAD] ========== Starting bulkLoad ==========" << std::endl;
      std::cout << "[BULK LOAD] Config file: " << config_file << ", data size: " << data.size() << std::endl;
      std::cout.flush();
      
      // Step 1: Load segments and build tree (16 segments per leaf, 16 children per inner node)
      std::cout << "[BULK LOAD] Loading segments from config file..." << std::endl;
      std::cout.flush();
      loadConfigByFile(config_file);
      std::cout << "[BULK LOAD] Tree structure built. Starting data insertion..." << std::endl;
      
      // Step 2: Insert key-value pairs into segments using OpenMP
      omp_set_num_threads(thread_num);
      size_t total = data.size();
      size_t progress_interval = std::max(static_cast<size_t>(1), total / 100);  // Update every 1%
      
      auto start_time = std::chrono::high_resolution_clock::now();
      std::atomic<size_t> global_inserted(0);
      std::atomic<size_t> last_printed_count(0);
      
      #pragma omp parallel for
      for (size_t i = 0; i < data.size(); i++) {
         insert(data[i].first, data[i].second);
         size_t current_count = global_inserted.fetch_add(1) + 1;
         
         size_t last_printed = last_printed_count.load();
         size_t next_print_threshold = ((last_printed / progress_interval) + 1) * progress_interval;
         if (current_count >= next_print_threshold || current_count == total) {
            size_t expected = last_printed;
            if (last_printed_count.compare_exchange_strong(expected, current_count)) {
               #pragma omp critical
               {
                  auto current_time = std::chrono::high_resolution_clock::now();
                  long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
                  double progress = 100.0 * current_count / total;
                  long long elapsed_ms = std::max(elapsed, 1LL);
                  double rate = current_count * 1000.0 / elapsed_ms;  // keys per second
                  
                  std::cout << "\r[BULK LOAD] Progress: " << std::fixed << std::setprecision(1) << progress 
                            << "% (" << current_count << "/" << total << " keys) | "
                            << "Rate: " << std::setprecision(0) << rate / 1000.0 << " K keys/s"
                            << std::flush;
               }
            }
         }
      }
      
      size_t inserted = global_inserted.load();
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      long long total_time_ms = std::max(static_cast<long long>(total_time), 1LL);
      std::cout << "\n[BULK LOAD] Completed! Inserted " << inserted << " keys in " 
                << total_time / 1000.0 << " seconds (" 
                << inserted * 1000.0 / total_time_ms << " K keys/s)" << std::endl;
      
      // Step 3: Print tree structure
      // Note: This prints the tree structure after segments are loaded but before keys are inserted
      // If file cannot be opened, continue without printing (non-critical)
      // Skip tree structure printing for large datasets as it can be very slow
      std::cout << "[BULK LOAD] Skipping tree structure printing (can be slow for large datasets)" << std::endl;
      /*
      try {
         std::cout << "[BULK LOAD] Printing tree structure..." << std::endl;
         printTreeStructure("tree_structure.txt");
         std::cout << "[BULK LOAD] Tree structure printed." << std::endl;
      } catch (const std::exception& e) {
         // Silently ignore file writing errors - tree structure printing is for debugging only
         std::cout << "[BULK LOAD] Failed to print tree structure: " << e.what() << std::endl;
      }
      */
      
      // Step 4: Disable bulk load mode (switch to normal mode for future operations)
      // Note: Normal mode allows up to 128 segments/children, but node splitting is disabled
      // Update all nodes to normal mode
      std::cout << "[BULK LOAD] Switching to normal mode..." << std::endl;
      std::function<void(NodeBase*)> setNormalMode = [&](NodeBase* node) {
         if (node->type == PageType::BTreeLeaf) {
            static_cast<BTreeLeaf<Key,Value>*>(node)->isBulkLoading = false;
         } else if (node->type == PageType::BTreeInner) {
            static_cast<BTreeInner<Key>*>(node)->isBulkLoading = false;
            auto inner = static_cast<BTreeInner<Key>*>(node);
            for (unsigned i = 0; i <= inner->count; i++) {
               setNormalMode(inner->children[i]);
            }
         }
      };
      setNormalMode(root.load());
      std::cout << "[BULK LOAD] Normal mode enabled." << std::endl;
   }
   
   // Overload for key-value pairs without config file
   void bulkLoad(const std::vector<std::pair<Key, Value>>& data) {
      // Enable bulk load mode
      isBulkLoading.store(true);
      
      // Update root node
      NodeBase* current_root = root.load();
      if (current_root->type == PageType::BTreeLeaf) {
         static_cast<BTreeLeaf<Key,Value>*>(current_root)->isBulkLoading = true;
      } else if (current_root->type == PageType::BTreeInner) {
         static_cast<BTreeInner<Key>*>(current_root)->isBulkLoading = true;
      }
      
      // Insert all key-value pairs in bulk load mode using OpenMP
      omp_set_num_threads(thread_num);
      size_t inserted = 0;
      #pragma omp parallel for reduction(+:inserted)
      for (size_t i = 0; i < data.size(); i++) {
         insert(data[i].first, data[i].second);
         inserted++;
      }
      
      // Disable bulk load mode
      isBulkLoading.store(false);
   }
   
   // Overload for keys only (creates segments automatically)
   void bulkLoad(const std::vector<Key>& keys) {
      // Enable bulk load mode
      isBulkLoading.store(true);
      
      // Update root node
      NodeBase* current_root = root.load();
      if (current_root->type == PageType::BTreeLeaf) {
         static_cast<BTreeLeaf<Key,Value>*>(current_root)->isBulkLoading = true;
      } else if (current_root->type == PageType::BTreeInner) {
         static_cast<BTreeInner<Key>*>(current_root)->isBulkLoading = true;
      }
      
      // Insert all keys (value = key) in bulk load mode using OpenMP
      omp_set_num_threads(thread_num);
      size_t inserted = 0;
      #pragma omp parallel for reduction(+:inserted)
      for (size_t i = 0; i < keys.size(); i++) {
         insert(keys[i], keys[i]);
         inserted++;
      }
      
      // Disable bulk load mode
      isBulkLoading.store(false);
   }

   void makeRoot(Key k,NodeBase* leftChild,NodeBase* rightChild) {
      auto inner = new BTreeInner<Key>();
      inner->count = 1;
      inner->keys[0] = k;
      inner->children[0] = leftChild;
      inner->children[1] = rightChild;
      inner->isBulkLoading = isBulkLoading.load();  // Use current bulk load mode
      root = inner;
   }

  void yield(int count) {
    if (count>3)
      sched_yield();
    else
      _mm_pause();
  }

  void insert(Key k, Value v) {
    int restartCount = 0;
  restart:
    if (restartCount++) {
      yield(restartCount);
    }
    bool needRestart = false;

    // Track path from root to leaf for recursive splitting
    std::vector<PathEntry> path;

    // Current node
    NodeBase* node = root;
    uint64_t versionNode = node->readLockOrRestart(needRestart);
    if (needRestart || (node!=root)) {
      goto restart;
    }

    // Traverse to leaf, tracking path
    while (node->type==PageType::BTreeInner) {
      auto inner = static_cast<BTreeInner<Key>*>(node);

      // Add to path
      PathEntry entry;
      entry.node = node;
      entry.version = versionNode;
      entry.is_inner = true;
      path.push_back(entry);

      // Route to child: if k == keys[i], go to children[i+1] (because keys[i] means children[i+1] >= keys[i])
      unsigned pos = inner->lowerBound(k);
      if (pos < inner->count && inner->keys[pos] == k) {
         // Key equals separator, go to right child (children[i+1])
         node = inner->children[pos + 1];
      } else {
         // Key < separator, go to left child (children[pos])
         node = inner->children[pos];
      }
      inner->checkOrRestart(versionNode, needRestart);
	if (needRestart) {
	  goto restart;
	}
      versionNode = node->readLockOrRestart(needRestart);
      if (needRestart) {
	  goto restart;
	}
    }

    auto leaf = static_cast<BTreeLeaf<Key,Value>*>(node);
    
    // Add leaf to path
    PathEntry leafEntry;
    leafEntry.node = node;
    leafEntry.version = versionNode;
    leafEntry.is_inner = false;
    path.push_back(leafEntry);

    // Check promoted segments first (if root is inner node, we need to find which child this leaf belongs to)
    // For now, we'll check after we have the leaf - we need to determine child_index
    // This is a simplified approach - in full implementation, we'd track child_index during traversal
    // size_t child_index = SIZE_MAX;  // Will be determined if needed - currently unused
    
    // Try to find in promoted segments (only if we have a way to determine child_index)
    // For now, we'll skip this check and go directly to leaf segments
    // TODO: Implement proper child_index tracking during traversal

    // Check if any segment in this leaf is currently splitting
    for (unsigned i = 0; i < leaf->count; i++) {
      if (leaf->segments[i] && leaf->segments[i]->is_currently_splitting()) {
        // Release all locks
        for (auto& entry : path) {
          if (entry.is_inner) {
            auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
            inner->readUnlockOrRestart(entry.version, needRestart);
          } else {
            auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
            leaf_node->readUnlockOrRestart(entry.version, needRestart);
          }
        }
        // Wait for split to complete
        leaf->segments[i]->wait_for_split_completion();
	goto restart;
      }
    }

    // Find segment containing this key
    ribtree::Segment<Key, Value>* seg = nullptr;
    unsigned segPos = leaf->findSegment(k);
    
    if (segPos < leaf->count) {
      seg = leaf->segments[segPos];
    }
    
    // If no segment found, this is a routing error - retry from root
    if (!seg) {
      for (auto& entry : path) {
        if (entry.is_inner) {
          auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
          inner->readUnlockOrRestart(entry.version, needRestart);
        } else {
          auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
          leaf_node->readUnlockOrRestart(entry.version, needRestart);
        }
      }
      goto restart;
    }

    // Check if segment is currently splitting
    if (seg->is_currently_splitting()) {
      // Release all locks
      for (auto& entry : path) {
        if (entry.is_inner) {
          auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
          inner->readUnlockOrRestart(entry.version, needRestart);
        } else {
          auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
          leaf_node->readUnlockOrRestart(entry.version, needRestart);
        }
      }
      // Wait for split to complete
      seg->wait_for_split_completion();
      goto restart;
    }

    // Try to insert into segment
    ribtree::InsertResult result = seg->insertKeyValue(k, v);
    
    if (result.status == ribtree::InsertStatus::SUCCESS) {
      // Release all locks
      for (auto& entry : path) {
        if (entry.is_inner) {
          auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
          inner->readUnlockOrRestart(entry.version, needRestart);
        } else {
          auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
          leaf_node->readUnlockOrRestart(entry.version, needRestart);
        }
      }
	if (needRestart) goto restart;
      return; // success
    } else if (result.status == ribtree::InsertStatus::OUT_OF_RANGE) {
      // OUT_OF_RANGE is expected during splits, only log if retry count is high
      if (restartCount > 10) {
        std::cout << "[INSERT] Key=" << k << " OUT_OF_RANGE (retry " << restartCount << ")" << std::endl;
      }
      // Key is out of range - segment may have split, retry from root
      for (auto& entry : path) {
        if (entry.is_inner) {
          auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
          inner->readUnlockOrRestart(entry.version, needRestart);
        } else {
          auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
          leaf_node->readUnlockOrRestart(entry.version, needRestart);
        }
      }
      goto restart;
    } else if (result.status == ribtree::InsertStatus::SPLIT) {
      // Segment splitting is expected, only log if retry count is high
      if (restartCount > 10) {
        std::cout << "[INSERT] Key=" << k << " Segment splitting (retry " << restartCount << ")" << std::endl;
      }
      // Segment is currently splitting - wait and retry from root
      for (auto& entry : path) {
        if (entry.is_inner) {
          auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
          inner->readUnlockOrRestart(entry.version, needRestart);
        } else {
          auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
          leaf_node->readUnlockOrRestart(entry.version, needRestart);
        }
      }
      seg->wait_for_split_completion();
      goto restart;
    } else if (result.status == ribtree::InsertStatus::FULL) {
      // Segment is full - need to split
      // Try to mark segment for splitting
      if (!seg->try_mark_for_splitting()) {
        // Another thread is already splitting - wait and retry
        for (auto& entry : path) {
          if (entry.is_inner) {
            auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
            inner->readUnlockOrRestart(entry.version, needRestart);
          } else {
            auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
            leaf_node->readUnlockOrRestart(entry.version, needRestart);
          }
        }
        seg->wait_for_split_completion();
        goto restart;
      }

      // Acquire write lock BEFORE splitting segment
      // This prevents other threads from modifying the leaf during split
      // Other threads' read operations will fail and retry (optimistic locking)
      // Other threads' write operations will also fail
      leaf->upgradeToWriteLockOrRestart(versionNode, needRestart);
      if (needRestart) {
        if (restartCount > 10) {
          std::cout << "[INSERT] Key=" << k << " Failed to upgrade write lock (retry " << restartCount << ")" << std::endl;
        }
        // Failed to upgrade - unmark and retry from root
        seg->unmark_splitting();
        seg->splitting_.store(false, std::memory_order_release);
        for (auto& entry : path) {
          if (entry.is_inner) {
            auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
            inner->readUnlockOrRestart(entry.version, needRestart);
          } else {
            auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
            leaf_node->readUnlockOrRestart(entry.version, needRestart);
          }
        }
	goto restart;
      }

      // Verify segment is still at the same position
      if (segPos >= leaf->count || leaf->segments[segPos] != seg) {
        std::cout << "[INSERT] WARNING: Key=" << k << " Segment position changed! segPos=" << segPos 
                  << ", leaf->count=" << leaf->count << std::endl;
        leaf->writeUnlock();
        seg->unmark_splitting();
        seg->splitting_.store(false, std::memory_order_release);
        for (auto& entry : path) {
          if (entry.is_inner) {
            auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
            inner->readUnlockOrRestart(entry.version, needRestart);
          } else {
            auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
            leaf_node->readUnlockOrRestart(entry.version, needRestart);
          }
        }
	goto restart;
      }

      // Check if this segment is promoted to root - if so, remove it first
      // We need to find child_index for this leaf
      size_t child_index = findChildIndexForLeaf(leaf);
      if (child_index != SIZE_MAX) {
        // Check if this segment is in promoted segments
        bool needRestart_promoted = false;
        uint64_t version_promoted = promoted_lock.readLockOrRestart(needRestart_promoted);
        if (!needRestart_promoted && child_index < promoted_child_count && 
            promoted_bitmap[child_index] == 1) {
          // Check if seg is in promoted segments for this child
          SegmentIndexRange range = promoted_index[child_index];
          bool is_promoted = false;
          for (size_t i = range.start; i <= range.end; i++) {
            if (promoted_segments[i] == seg) {
              is_promoted = true;
              break;
            }
          }
          promoted_lock.readUnlockOrRestart(version_promoted, needRestart_promoted);
          
          if (is_promoted) {
            // Remove from promoted segments before splitting
            removePromotedSegment(seg, child_index);
          }
        } else if (!needRestart_promoted) {
          promoted_lock.readUnlockOrRestart(version_promoted, needRestart_promoted);
        }
      }

      // Now perform segment split (with write lock held, leaf cannot be modified by others)
      auto new_segments = seg->splitSegment(result.box_index, overflowThreshold, underflowThreshold);
      
      if (new_segments.empty()) {
        std::cerr << "[INSERT] ERROR: Key=" << k << " Segment split returned empty" << std::endl;
        // Split failed or no entries - unmark and retry
        leaf->writeUnlock();
        seg->unmark_splitting();
        seg->splitting_.store(false, std::memory_order_release);
        for (auto& entry : path) {
          if (entry.is_inner) {
            auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
            inner->readUnlockOrRestart(entry.version, needRestart);
          } else {
            auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
            leaf_node->readUnlockOrRestart(entry.version, needRestart);
          }
        }
      goto restart;
      }


      // Replace segments in leaf (write lock already held)
      replaceSegmentsInLeaf(leaf, segPos, new_segments, seg, path, versionNode);

      // Release all locks
      // Note: leaf node has write lock, others have read locks
      // Some inner nodes may have been upgraded to write lock in replaceSegmentsInLeaf
      for (auto& entry : path) {
        if (entry.is_inner) {
          // Check if this lock has already been released (marked with UINT64_MAX)
          if (entry.version == UINT64_MAX) {
            continue;  // Already released, skip
          }
          // Check if this node was upgraded to write lock (version has write lock bit set)
          // If version & 0b10 == 0b10, it means write lock was acquired
          if ((entry.version & 0b10) == 0b10) {
            // This node has write lock, use writeUnlock()
            auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
            inner->writeUnlock();
    } else {
            // This node has read lock, use readUnlockOrRestart()
            auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
            inner->readUnlockOrRestart(entry.version, needRestart);
          }
        } else {
          // Leaf node has write lock, use writeUnlock() instead of readUnlockOrRestart()
          auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
          leaf_node->writeUnlock();
        }
      }
      if (needRestart) goto restart;

      // Retry insertion from root after segment split
	  goto restart;
    } else {
      // Other error - release locks and throw
      std::cout << "[INSERT] ERROR: Key=" << k << " Unexpected status=" << static_cast<int>(result.status) 
                << " (0=SUCCESS,1=FULL,2=SPLIT,3=OUT_OF_RANGE,4=ERROR)" << std::endl;
      std::cout << "[INSERT] ERROR: Segment range=[" << seg->getLowerBound() << "," << seg->getUpperBound() << ")" << std::endl;
      std::cout << "[INSERT] ERROR: Leaf segments count=" << leaf->count << std::endl;
      for (auto& entry : path) {
        if (entry.is_inner) {
          auto* inner = static_cast<BTreeInner<Key>*>(entry.node);
          inner->readUnlockOrRestart(entry.version, needRestart);
        } else {
          auto* leaf_node = static_cast<BTreeLeaf<Key,Value>*>(entry.node);
          leaf_node->readUnlockOrRestart(entry.version, needRestart);
        }
      }
      std::string error_msg = "Insert failed: Segment returned status " + 
                              std::to_string(static_cast<int>(result.status)) +
                              " for key " + std::to_string(k);
      throw std::runtime_error(error_msg);
    }
  }

  void replaceSegmentsInLeaf(BTreeLeaf<Key, Value>*& leaf, unsigned oldSegPos,
                            const std::vector<std::pair<Key, ribtree::Segment<Key, Value>*>>& new_segments,
                            ribtree::Segment<Key, Value>* old_segment,
                            std::vector<PathEntry>& path,
                            [[maybe_unused]] uint64_t& versionNode) {
    // Verify segment is still at the same position
    if (oldSegPos >= leaf->count || leaf->segments[oldSegPos] != old_segment) {
      std::cout << "[REPLACE_SEGMENTS] ERROR: Segment position changed! oldSegPos=" << oldSegPos 
                << ", leaf->count=" << leaf->count 
                << ", segment_ptr_match=" << (oldSegPos < leaf->count ? (leaf->segments[oldSegPos] == old_segment ? "yes" : "no") : "out_of_range")
                << std::endl;
      // Don't release leaf's write lock here - let insert function handle it
      // This prevents double unlock when insert function tries to release locks
      old_segment->unmark_splitting();
      old_segment->splitting_.store(false, std::memory_order_release);
      return;
    }
    
    // Check if old segment is reused in new_segments
    bool segment_reused = false;
    for (const auto& pair : new_segments) {
      if (pair.second == old_segment) {
        segment_reused = true;
        break;
      }
    }

    if (new_segments.empty()) {
      // Should not happen, but handle gracefully
      std::cerr << "[REPLACE_SEGMENTS] ERROR: new_segments is empty!" << std::endl;
      // Don't release leaf's write lock here - let insert function handle it
      // This prevents double unlock when insert function tries to release locks
      old_segment->unmark_splitting();
      old_segment->splitting_.store(false, std::memory_order_release);
      return;
    }
    
    // Step 1: Remove old segment
    for (unsigned i = oldSegPos; i < static_cast<unsigned>(leaf->count) - 1; i++) {
      leaf->keys[i] = leaf->keys[i + 1];
      leaf->segments[i] = leaf->segments[i + 1];
    }
    leaf->count--;
    
    // Step 2: Insert all new segments (may temporarily exceed SPLIT_THRESHOLD)
    unsigned elements_after = leaf->count - oldSegPos;
    if (new_segments.size() == 1) {
      // Simple case: replace 1 segment with 1 segment
      leaf->keys[oldSegPos] = new_segments[0].first;
      leaf->segments[oldSegPos] = new_segments[0].second;
      leaf->count++;
    } else {
      // Multiple segments: need to make room
      unsigned shift_by = new_segments.size();
      if (elements_after > 0) {
        unsigned new_pos = oldSegPos + shift_by;
        // Check bounds (but allow temporary overflow for splitting)
        if (new_pos + elements_after > leaf->maxEntries) {
          std::cerr << "[REPLACE_SEGMENTS] WARNING: Leaf would exceed maxEntries! count=" << leaf->count 
                    << ", new_segments.size()=" << new_segments.size() 
                    << ", maxEntries=" << leaf->maxEntries << std::endl;
          // Truncate to fit
          elements_after = (new_pos < leaf->maxEntries) ? (leaf->maxEntries - new_pos) : 0;
        }
        if (elements_after > 0) {
          memmove(leaf->keys + new_pos, leaf->keys + oldSegPos, sizeof(Key) * elements_after);
          memmove(leaf->segments + new_pos, leaf->segments + oldSegPos, 
                  sizeof(ribtree::Segment<Key, Value>*) * elements_after);
        }
      }
      
      // Insert new segments at oldSegPos
      for (size_t i = 0; i < new_segments.size(); i++) {
        leaf->keys[oldSegPos + i] = new_segments[i].first;
        leaf->segments[oldSegPos + i] = new_segments[i].second;
      }
      
      // Update count
      leaf->count += new_segments.size();
    }
    
    // Step 3: Check if leaf needs to split (only in normal mode, not during bulk load)
    if (!leaf->isBulkLoading && leaf->count >= SPLIT_THRESHOLD) {
      // Leaf overflowed - need to split
      Key sep;
      BTreeLeaf<Key, Value>* new_leaf = leaf->split(sep);
      
      // Acquire write lock for new leaf node immediately after creation
      // This ensures consistency: we hold write locks on both old and new leaf nodes
      bool needRestart_new = false;
      new_leaf->writeLockOrRestart(needRestart_new);
      if (needRestart_new) {
        // This should never happen for a newly created node, but handle gracefully
        std::cerr << "[REPLACE_SEGMENTS] ERROR: Failed to acquire write lock on new leaf!" << std::endl;
        // Release leaf's write lock before returning (caller will handle path locks)
        leaf->writeUnlock();
        delete new_leaf;
        return;
      }
      
      // After split, leaf and new_leaf both have approximately half the segments
      
      old_segment->unmark_splitting();
      old_segment->splitting_.store(false, std::memory_order_release);
      // Unmark splitting on old segment if not reused
      if (!segment_reused) {
        delete old_segment;
      }

      // Insert new_leaf into parent
      if (path.size() > 1) {
        // We have a parent - insert new_leaf
        PathEntry& parentEntry = path[path.size() - 2];
        BTreeInner<Key>* parent = static_cast<BTreeInner<Key>*>(parentEntry.node);
        
        // Upgrade parent to write lock
        bool needRestart = false;
        parent->upgradeToWriteLockOrRestart(parentEntry.version, needRestart);
        if (needRestart) {
          // If we can't upgrade, we need to retry from root
          // Release locks and let caller retry
          leaf->writeUnlock();
          new_leaf->writeUnlock();
          delete new_leaf;
          return;
        }
        
        // Mark that parent has been upgraded to write lock
        // Update version to write lock version (original version + 0b10)
        parentEntry.version = parentEntry.version + 0b10;

        // Insert new_leaf into parent
        parent->insert(sep, new_leaf);

        // Check if parent overflows
        if (parent->isFull()) {
          // Recursively split parent
          // Note: parent's write lock will be handled by splitInnerNodeRecursive
          new_leaf->writeUnlock();
          splitInnerNodeRecursive(path, path.size() - 2);
        } else {
          // Release parent's write lock and mark as released
          parent->writeUnlock();
          // Mark as released by setting version to UINT64_MAX
          // This tells insert() that this lock has already been released
          parentEntry.version = UINT64_MAX;
          new_leaf->writeUnlock();
        }
      } else {
        // No parent - leaf was root, need to create new root
        BTreeInner<Key>* new_root = new BTreeInner<Key>();
        new_root->count = 2;
        new_root->children[0] = leaf;
        new_root->children[1] = new_leaf;
        new_root->keys[0] = sep;
        root = new_root;
        new_leaf->writeUnlock();
      }
      return;
    }
    
    // Leaf doesn't need to split - we're done
    old_segment->unmark_splitting();
    old_segment->splitting_.store(false, std::memory_order_release);
    // Unmark splitting on old segment if not reused
    if (!segment_reused) {
      delete old_segment;
    }

  }

  // Recursively split inner nodes when they overflow
  void splitInnerNodeRecursive(std::vector<PathEntry>& path, size_t nodeIndex) {
    if (nodeIndex >= path.size()) {
      return;
    }

    PathEntry& entry = path[nodeIndex];
    BTreeInner<Key>* inner = static_cast<BTreeInner<Key>*>(entry.node);

    // Split inner node
    Key sep;
    BTreeInner<Key>* new_inner = inner->split(sep);
    
    // Acquire write lock for new inner node immediately after creation
    // This ensures consistency: we hold write locks on both old and new inner nodes
    bool needRestart_new = false;
    new_inner->writeLockOrRestart(needRestart_new);
    if (needRestart_new) {
      // This should never happen for a newly created node, but handle gracefully
      std::cerr << "[SPLIT_INNER] ERROR: Failed to acquire write lock on new inner node!" << std::endl;
      inner->writeUnlock();
      delete new_inner;
      return;
    }

    if (nodeIndex == 0) {
      // This is the root - create new root
      BTreeInner<Key>* new_root = new BTreeInner<Key>();
      new_root->count = 2;
      new_root->children[0] = inner;
      new_root->children[1] = new_inner;
      new_root->keys[0] = sep;
      root = new_root;
      inner->writeUnlock();
      new_inner->writeUnlock();
      
      // Update path[0] to point to new root
      path[0].node = new_root;
      path[0].is_inner = true;
    } else {
      // Insert new_inner into parent
      PathEntry& parentEntry = path[nodeIndex - 1];
      BTreeInner<Key>* parent = static_cast<BTreeInner<Key>*>(parentEntry.node);
      
      // Upgrade parent to write lock
      bool needRestart = false;
      parent->upgradeToWriteLockOrRestart(parentEntry.version, needRestart);
      if (needRestart) {
        // Can't upgrade - caller will retry
        inner->writeUnlock();
        new_inner->writeUnlock();
        return;
      }
      
      // Mark that parent has been upgraded to write lock
      // Update version to write lock version (original version + 0b10)
      parentEntry.version = parentEntry.version + 0b10;

      // Insert new_inner into parent
      parent->insert(sep, new_inner);

      // Check if parent overflows
      if (parent->isFull()) {
        // Recursively split parent
        // Note: parent's write lock will be handled by recursive call
        inner->writeUnlock();
        new_inner->writeUnlock();
        splitInnerNodeRecursive(path, nodeIndex - 1);
      } else {
        inner->writeUnlock();
        new_inner->writeUnlock();
        // Release parent's write lock and mark as released
        parent->writeUnlock();
        // Mark as released by setting version to UINT64_MAX
        parentEntry.version = UINT64_MAX;
      }
    }
  }

  bool lookup(Key k, Value& result) {
    int restartCount = 0;
  restart:
    if (restartCount++)
      yield(restartCount);
    bool needRestart = false;

    NodeBase* node = root;
    uint64_t versionNode = node->readLockOrRestart(needRestart);
    if (needRestart || (node!=root)) goto restart;

    // Parent of current node
    BTreeInner<Key>* parent = nullptr;
    uint64_t versionParent;

    while (node->type==PageType::BTreeInner) {
      auto inner = static_cast<BTreeInner<Key>*>(node);

      if (parent) {
	parent->readUnlockOrRestart(versionParent, needRestart);
	if (needRestart) goto restart;
      }

      parent = inner;
      versionParent = versionNode;

      // Route to child: if k == keys[i], go to children[i+1] (because keys[i] means children[i+1] >= keys[i])
      unsigned pos = inner->lowerBound(k);
      if (pos < inner->count && inner->keys[pos] == k) {
         // Key equals separator, go to right child (children[i+1])
         node = inner->children[pos + 1];
      } else {
         // Key < separator, go to left child (children[pos])
         node = inner->children[pos];
      }
      inner->checkOrRestart(versionNode, needRestart);
      if (needRestart) goto restart;
      versionNode = node->readLockOrRestart(needRestart);
      if (needRestart) goto restart;
    }

    BTreeLeaf<Key,Value>* leaf = static_cast<BTreeLeaf<Key,Value>*>(node);
    
    // Find segment containing this key
    // keys[i] stores the lower_bound of segment i
    ribtree::Segment<Key, Value>* seg = nullptr;
    unsigned segPos = leaf->findSegment(k);
    
    if (segPos < leaf->count) {
      seg = leaf->segments[segPos];
    }

    bool success = false;
    if (seg) {
      // Verify key is within segment range before searching
      // Segment should have been found correctly by findSegment, but double-check
      if (k < seg->getLowerBound() || k >= seg->getUpperBound()) {
        std::string error_msg = "Routing error: Key " + 
                                std::to_string(k) + 
                                " is out of range for segment [" +
                                std::to_string(seg->getLowerBound()) + ", " +
                                std::to_string(seg->getUpperBound()) +
                                ") during lookup. This indicates a problem with the routing system.";
        throw std::runtime_error(error_msg);
      }
      
      // Search in segment's box
      ribtree::SearchResult<Key, Value> searchResult = seg->searchKey(k);
      if (searchResult.status == ribtree::SearchStatus::SUCCESS) {
      success = true;
        result = searchResult.value;
      }
      // If NOT_FOUND, that's normal - key doesn't exist in the segment
    } else {
      // Cannot find segment for key - this is a routing error
      std::string error_msg = "Routing error: Cannot find segment for key " + 
                              std::to_string(k) + 
                              " in leaf node during lookup. This indicates a problem with the routing system.";
      throw std::runtime_error(error_msg);
    }

    if (parent) {
      parent->readUnlockOrRestart(versionParent, needRestart);
      if (needRestart) goto restart;
    }
    node->readUnlockOrRestart(versionNode, needRestart);
    if (needRestart) goto restart;

    return success;
  }

  uint64_t scan(Key k, int range, Value* output) {
    int restartCount = 0;
  restart:
    if (restartCount++)
      yield(restartCount);
    bool needRestart = false;

    NodeBase* node = root;
    uint64_t versionNode = node->readLockOrRestart(needRestart);
    if (needRestart || (node!=root)) goto restart;

    // Parent of current node
    BTreeInner<Key>* parent = nullptr;
    uint64_t versionParent;

    while (node->type==PageType::BTreeInner) {
      auto inner = static_cast<BTreeInner<Key>*>(node);

      if (parent) {
	parent->readUnlockOrRestart(versionParent, needRestart);
	if (needRestart) goto restart;
      }

      parent = inner;
      versionParent = versionNode;

      // Route to child: if k == keys[i], go to children[i+1] (because keys[i] means children[i+1] >= keys[i])
      unsigned pos = inner->lowerBound(k);
      if (pos < inner->count && inner->keys[pos] == k) {
         // Key equals separator, go to right child (children[i+1])
         node = inner->children[pos + 1];
      } else {
         // Key < separator, go to left child (children[pos])
         node = inner->children[pos];
      }
      inner->checkOrRestart(versionNode, needRestart);
      if (needRestart) goto restart;
      versionNode = node->readLockOrRestart(needRestart);
      if (needRestart) goto restart;
    }

    BTreeLeaf<Key,Value>* leaf = static_cast<BTreeLeaf<Key,Value>*>(node);
    
    // Find segment containing this key
    // keys[i] stores the lower_bound of segment i
    ribtree::Segment<Key, Value>* seg = nullptr;
    unsigned segPos = leaf->findSegment(k);
    
    if (segPos < leaf->count) {
      seg = leaf->segments[segPos];
    }

    int count = 0;
    if (seg) {
      // Verify key is within segment range before scanning
      // Segment should have been found correctly by findSegment, but double-check
      if (k < seg->getLowerBound() || k >= seg->getUpperBound()) {
        std::string error_msg = "Routing error: Key " + 
                                std::to_string(k) + 
                                " is out of range for segment [" +
                                std::to_string(seg->getLowerBound()) + ", " +
                                std::to_string(seg->getUpperBound()) +
                                ") during scan. This indicates a problem with the routing system.";
        throw std::runtime_error(error_msg);
      }
      
      // Scan from segment - get all entries starting from k
      // Note: This is a simplified version - actual scan would need to handle multiple segments
      std::vector<std::pair<Key, Value>> entries = seg->getAllEntries();
      for (const auto& entry : entries) {
        if (entry.first >= k && count < range) {
          output[count++] = entry.second;
        }
        if (count >= range) break;
      }
    } else {
      // Cannot find segment for key - this is a routing error
      std::string error_msg = "Routing error: Cannot find segment for key " + 
                              std::to_string(k) + 
                              " in leaf node during scan. This indicates a problem with the routing system.";
      throw std::runtime_error(error_msg);
    }

    if (parent) {
      parent->readUnlockOrRestart(versionParent, needRestart);
      if (needRestart) goto restart;
    }
    node->readUnlockOrRestart(versionNode, needRestart);
    if (needRestart) goto restart;

    return count;
  }

  // Insert a segment directly into the tree (used by loadConfigByFile)
  void insertSegment(Key k, ribtree::Segment<Key, Value>* seg) {
    int restartCount = 0;
  restart:
    if (restartCount++)
      yield(restartCount);
    bool needRestart = false;

    NodeBase* node = root;
    uint64_t versionNode = node->readLockOrRestart(needRestart);
    if (needRestart || (node!=root)) goto restart;

    BTreeInner<Key>* parent = nullptr;
    uint64_t versionParent;

    while (node->type==PageType::BTreeInner) {
      auto inner = static_cast<BTreeInner<Key>*>(node);

      // Note: insertSegment is only used during bulk load (buildTreeFromSegments)
      // All segments are pre-built, so no new children will be inserted
      // Therefore, we don't need to check for splits here

      if (parent) {
        parent->readUnlockOrRestart(versionParent, needRestart);
        if (needRestart) goto restart;
      }

      parent = inner;
      versionParent = versionNode;

      // Route to child: if k == keys[i], go to children[i+1] (because keys[i] means children[i+1] >= keys[i])
      unsigned pos = inner->lowerBound(k);
      if (pos < inner->count && inner->keys[pos] == k) {
         // Key equals separator, go to right child (children[i+1])
         node = inner->children[pos + 1];
      } else {
         // Key < separator, go to left child (children[pos])
         node = inner->children[pos];
      }
      inner->checkOrRestart(versionNode, needRestart);
      if (needRestart) goto restart;
      versionNode = node->readLockOrRestart(needRestart);
      if (needRestart) goto restart;
    }

    auto leaf = static_cast<BTreeLeaf<Key,Value>*>(node);

    // Need write lock to insert segment
    if (parent) {
      parent->upgradeToWriteLockOrRestart(versionParent, needRestart);
      if (needRestart) goto restart;
    }
    node->upgradeToWriteLockOrRestart(versionNode, needRestart);
    if (needRestart) {
      if (parent) parent->writeUnlock();
      goto restart;
    }
    if (!parent && (node != root)) {
      node->writeUnlock();
      goto restart;
    }

    // Note: insertSegment is only used during bulk load (buildTreeFromSegments)
    // All segments are pre-built and inserted in order, so leaves should never be full
    // The split check is removed as it's unnecessary

    // Insert segment into leaf
    leaf->insert(k, seg);

    if (parent) {
      parent->readUnlockOrRestart(versionParent, needRestart);
      if (needRestart) {
        node->writeUnlock();
        goto restart;
      }
    }
    node->writeUnlock();
  }

  // Build tree structure from segments in bulk
  // Building standard: 16 segments per leaf, 16 children per inner node
  // This is not a threshold - nodes are built by grouping segments/children according to this standard
  void buildTreeFromSegments(const std::vector<std::pair<Key, ribtree::Segment<Key, Value>*>>& segment_list) {
    if (segment_list.empty()) return;
    
    const size_t BULK_LOAD_MAX = 16;  // Building standard: 16 segments per leaf, 16 children per inner
    
    std::cout << "[BUILD_TREE] Starting to build tree from " << segment_list.size() << " segments" << std::endl;
    
    // Step 1: Build leaf nodes (each leaf contains up to 16 segments)
    std::vector<BTreeLeaf<Key,Value>*> leaf_nodes;
    size_t total_segments = segment_list.size();
    size_t total_leaves = (total_segments + BULK_LOAD_MAX - 1) / BULK_LOAD_MAX;
    size_t progress_interval = std::max(static_cast<size_t>(1), total_leaves / 100);  // Update every 1%
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "[BUILD_TREE] Step 1: Building leaf nodes..." << std::endl;
    for (size_t i = 0; i < segment_list.size(); i += BULK_LOAD_MAX) {
      size_t leaf_index = i / BULK_LOAD_MAX;
      
      // Print progress every 1%
      if (leaf_index % progress_interval == 0 || leaf_index == total_leaves - 1) {
        auto current_time = std::chrono::high_resolution_clock::now();
        long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
        double progress = 100.0 * (leaf_index + 1) / total_leaves;
        long long elapsed_ms = std::max(elapsed, 1LL);
        double rate = (leaf_index + 1) * 1000.0 / elapsed_ms;  // leaves per second
        
        std::cout << "\r[BUILD_TREE] Progress: " << std::fixed << std::setprecision(1) << progress 
                  << "% (" << (leaf_index + 1) << "/" << total_leaves << " leaf nodes) | "
                  << "Rate: " << std::setprecision(0) << rate << " leaves/s"
                  << std::flush;
      }
      auto* leaf = new BTreeLeaf<Key,Value>();
      leaf->isBulkLoading = true;  // Set bulk load mode
      
      size_t end = std::min(i + BULK_LOAD_MAX, segment_list.size());
      for (size_t j = i; j < end; j++) {
        Key lower_bound = segment_list[j].first;
        ribtree::Segment<Key, Value>* seg = segment_list[j].second;
        leaf->insert(lower_bound, seg);
      }
      leaf_nodes.push_back(leaf);
    }
    std::cout << std::endl;  // New line after progress bar
    std::cout << "[BUILD_TREE] Step 1 complete: Created " << leaf_nodes.size() << " leaf nodes" << std::endl;
    
    // Step 1.5: Select and promote top segments to root
    promoteTopSegmentsToRoot(leaf_nodes);
    
    // Step 2: Build inner nodes level by level (16 children per inner node)
    std::cout << "[BUILD_TREE] Step 2: Building inner nodes level by level..." << std::endl;
    std::vector<NodeBase*> current_level;
    for (auto* leaf : leaf_nodes) {
      current_level.push_back(leaf);
    }
    
    int level = 0;
    while (current_level.size() > 1) {
      size_t total_nodes_at_level = current_level.size();
      size_t total_inner_nodes = (total_nodes_at_level + BULK_LOAD_MAX - 1) / BULK_LOAD_MAX;
      size_t progress_interval = std::max(static_cast<size_t>(1), total_inner_nodes / 100);  // Update every 1%
      auto level_start_time = std::chrono::high_resolution_clock::now();
      
      std::vector<NodeBase*> next_level;
      
      for (size_t i = 0; i < current_level.size(); i += BULK_LOAD_MAX) {
        size_t inner_index = i / BULK_LOAD_MAX;
        
        // Print progress every 1%
        if (inner_index % progress_interval == 0 || inner_index == total_inner_nodes - 1) {
          auto current_time = std::chrono::high_resolution_clock::now();
          long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - level_start_time).count();
          double progress = 100.0 * (inner_index + 1) / total_inner_nodes;
          long long elapsed_ms = std::max(elapsed, 1LL);
          double rate = (inner_index + 1) * 1000.0 / elapsed_ms;  // nodes per second
          
          std::cout << "\r[BUILD_TREE] Level " << level << " Progress: " << std::fixed << std::setprecision(1) << progress 
                    << "% (" << (inner_index + 1) << "/" << total_inner_nodes << " inner nodes) | "
                    << "Rate: " << std::setprecision(0) << rate << " nodes/s"
                    << std::flush;
        }
        auto* inner = new BTreeInner<Key>();
        inner->isBulkLoading = true;  // Set bulk load mode
        
        size_t end = std::min(i + BULK_LOAD_MAX, current_level.size());
        
        // First child (no key)
        inner->children[0] = current_level[i];
        inner->count = 0;
        
        // Remaining children with keys
        // For B+ tree, separator key should be the first key of the next child
        // This ensures: all keys in children[i] < keys[i], all keys in children[i+1] >= keys[i]
        for (size_t j = i + 1; j < end; j++) {
          Key sep_key;
          if (current_level[j]->type == PageType::BTreeLeaf) {
            auto* curr_leaf = static_cast<BTreeLeaf<Key,Value>*>(current_level[j]);
            if (curr_leaf->count == 0) {
              throw std::runtime_error("Internal error: Leaf node has zero segments during tree construction");
            }
            sep_key = curr_leaf->keys[0];  // First key (first segment's lower_bound) in current leaf
          } else {
            auto* curr_inner = static_cast<BTreeInner<Key>*>(current_level[j]);
            if (curr_inner->count == 0) {
              throw std::runtime_error("Internal error: Inner node has zero keys during tree construction");
            }
            // For inner node, we need to get the minimum key from its leftmost path
            // But for simplicity, we can use the first separator key
            // Actually, we should get the first key from the leftmost leaf
            // For now, let's use a recursive approach to get the minimum key
            NodeBase* leftmost = curr_inner->children[0];
            while (leftmost->type == PageType::BTreeInner) {
              leftmost = static_cast<BTreeInner<Key>*>(leftmost)->children[0];
            }
            auto* leftmost_leaf = static_cast<BTreeLeaf<Key,Value>*>(leftmost);
            if (leftmost_leaf->count == 0) {
              throw std::runtime_error("Internal error: Leftmost leaf has zero segments during tree construction");
            }
            sep_key = leftmost_leaf->keys[0];  // First key in leftmost leaf
          }
          
          inner->keys[inner->count] = sep_key;
          inner->children[inner->count + 1] = current_level[j];
          inner->count++;
        }
        
        // Note: We do NOT swap children[0] and children[1] here
        // The children are already in the correct order from segment_list
        // children[0] contains the first child (smallest keys)
        // children[i+1] contains keys >= keys[i]
        // This matches the B+ tree structure where:
        // - children[0] contains all keys < keys[0]
        // - children[i+1] contains all keys >= keys[i] and < keys[i+1]
        
        next_level.push_back(inner);
      }
      
      std::cout << std::endl;  // New line after progress bar
      std::cout << "[BUILD_TREE] Level " << level << " complete: Created " << next_level.size() << " inner nodes" << std::endl;
      current_level = next_level;
      level++;
    }
    
    // Step 3: Set root
    std::cout << "[BUILD_TREE] Step 3: Setting root (level " << level << ")" << std::endl;
    root = current_level[0];
    std::cout << "[BUILD_TREE] Tree building complete!" << std::endl;
  }

  // Select and promote top segments to root based on keys_count
  void promoteTopSegmentsToRoot(const std::vector<BTreeLeaf<Key,Value>*>& leaf_nodes) {
    if (leaf_nodes.empty()) return;
    
    std::cout << "[PROMOTE] Selecting top " << MAX_PROMOTED_SEGMENTS << " segments to promote to root..." << std::endl;
    
    // Step 1: Collect all segments with their keys_count and child index
    struct SegmentInfo {
      ribtree::Segment<Key, Value>* seg;
      size_t child_index;  // Which leaf node (child) this segment belongs to
      size_t keys_count;   // Number of keys in this segment
      Key lower_bound;      // Segment lower bound for sorting
    };
    
    std::vector<SegmentInfo> all_segments;
    for (size_t child_idx = 0; child_idx < leaf_nodes.size(); child_idx++) {
      auto* leaf = leaf_nodes[child_idx];
      for (unsigned i = 0; i < leaf->count; i++) {
        if (leaf->segments[i] != nullptr) {
          SegmentInfo info;
          info.seg = leaf->segments[i];
          info.child_index = child_idx;
          info.lower_bound = leaf->keys[i];
          // Get keys_count from map (stored during loadConfigByFile)
          auto it = segment_keys_count_map.find(leaf->segments[i]);
          info.keys_count = (it != segment_keys_count_map.end()) ? it->second : 0;
          all_segments.push_back(info);
        }
      }
    }
    
    // Step 2: Sort by keys_count (descending), then select top MAX_PROMOTED_SEGMENTS
    // Note: We need keys_count from config file, but for now we'll use segment size as proxy
    // In real implementation, we should have stored keys_count during loadConfigByFile
    std::sort(all_segments.begin(), all_segments.end(), 
              [](const SegmentInfo& a, const SegmentInfo& b) {
                // Sort by keys_count descending, then by lower_bound for stability
                if (a.keys_count != b.keys_count) {
                  return a.keys_count > b.keys_count;
                }
                return a.lower_bound < b.lower_bound;
              });
    
    size_t promote_count = std::min(MAX_PROMOTED_SEGMENTS, all_segments.size());
    
    // Step 3: Initialize promoted segments arrays
    bool needRestart = false;
    promoted_lock.writeLockOrRestart(needRestart);
    if (needRestart) {
      std::cerr << "[PROMOTE] ERROR: Failed to acquire write lock" << std::endl;
      return;
    }
    
    promoted_child_count = leaf_nodes.size();
    promoted_bitmap.resize(promoted_child_count, 0);
    promoted_index.resize(promoted_child_count);
    promoted_segment_count = 0;
    
    // Step 4: Insert promoted segments and update arrays
    std::vector<size_t> child_segment_counts(promoted_child_count, 0);  // Count segments per child
    
    for (size_t i = 0; i < promote_count; i++) {
      const auto& info = all_segments[i];
      size_t pos = promoted_segment_count++;
      
      if (pos >= MAX_PROMOTED_SEGMENTS) {
        std::cerr << "[PROMOTE] WARNING: Exceeded MAX_PROMOTED_SEGMENTS" << std::endl;
        break;
      }
      
      promoted_segments[pos] = info.seg;
      
      // Update bitmap and index
      if (promoted_bitmap[info.child_index] == 0) {
        // First segment for this child
        promoted_bitmap[info.child_index] = 1;
        promoted_index[info.child_index] = {start: pos, end: pos};
        child_segment_counts[info.child_index] = 1;
      } else {
        // Additional segment for this child - extend range
        promoted_index[info.child_index].end = pos;
        child_segment_counts[info.child_index]++;
      }
    }
    
    promoted_lock.writeUnlock();
    
    // Print detailed promotion information
    std::cout << "[PROMOTE] ========== Promotion Summary ==========" << std::endl;
    std::cout << "[PROMOTE] Total segments promoted: " << promoted_segment_count << " / " << MAX_PROMOTED_SEGMENTS << std::endl;
    std::cout << "[PROMOTE] Total child nodes: " << promoted_child_count << std::endl;
    std::cout << "[PROMOTE] Children with promoted segments: ";
    size_t children_with_promoted = 0;
    for (size_t i = 0; i < promoted_child_count; i++) {
      if (promoted_bitmap[i] == 1) {
        children_with_promoted++;
      }
    }
    std::cout << children_with_promoted << std::endl;
    std::cout << "[PROMOTE] -----------------------------------------" << std::endl;
    
    // Print detailed information for each child
    for (size_t i = 0; i < promoted_child_count; i++) {
      if (promoted_bitmap[i] == 1) {
        SegmentIndexRange range = promoted_index[i];
        std::cout << "[PROMOTE] Child[" << i << "]: " << child_segment_counts[i] 
                  << " segment(s) promoted" << std::endl;
        std::cout << "[PROMOTE]   Index range: [" << range.start << ", " << range.end << "]" << std::endl;
        std::cout << "[PROMOTE]   Segments: ";
        
        // Print segment details
        for (size_t j = range.start; j <= range.end; j++) {
          if (promoted_segments[j] != nullptr) {
            auto* seg = promoted_segments[j];
            auto it = segment_keys_count_map.find(seg);
            size_t keys_count = (it != segment_keys_count_map.end()) ? it->second : 0;
            std::cout << "[" << seg->lower_bound << "," << seg->upper_bound << ")"
                      << "(keys=" << keys_count << ") ";
          }
        }
        std::cout << std::endl;
      }
    }
    
    // Print top segments by keys_count
    std::cout << "[PROMOTE] -----------------------------------------" << std::endl;
    std::cout << "[PROMOTE] Top " << std::min(promote_count, size_t(10)) << " segments by keys_count:" << std::endl;
    for (size_t i = 0; i < std::min(promote_count, size_t(10)); i++) {
      const auto& info = all_segments[i];
      std::cout << "[PROMOTE]   #" << (i+1) << ": Child[" << info.child_index 
                << "] segment [" << info.lower_bound << "," << info.seg->upper_bound 
                << ") keys_count=" << info.keys_count << std::endl;
    }
    
    std::cout << "[PROMOTE] =========================================" << std::endl;
  }
  
  // Find segment in promoted segments array
  ribtree::Segment<Key, Value>* findPromotedSegment(Key k, size_t child_index) {
    if (child_index >= promoted_child_count || promoted_bitmap[child_index] == 0) {
      return nullptr;
    }
    
    bool needRestart = false;
    uint64_t version = promoted_lock.readLockOrRestart(needRestart);
    if (needRestart) {
      return nullptr;  // Lock failed, retry later
    }
    
    // Get index range for this child
    SegmentIndexRange range = promoted_index[child_index];
    
    // Binary search in the range (skip nullptr)
    ribtree::Segment<Key, Value>* found = nullptr;
    size_t left = range.start;
    size_t right = range.end;
    
    while (left <= right) {
      size_t mid = left + (right - left) / 2;
      
      // Skip nullptr
      while (mid <= right && promoted_segments[mid] == nullptr) {
        mid++;
      }
      if (mid > right) {
        right = mid - 1;
        continue;
      }
      
      auto* seg = promoted_segments[mid];
      if (k >= seg->lower_bound && k < seg->upper_bound) {
        found = seg;
        break;
      } else if (k < seg->lower_bound) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    
    promoted_lock.readUnlockOrRestart(version, needRestart);
    if (needRestart) {
      return nullptr;  // Version changed, retry
    }
    
    return found;
  }
  
  // Find child index for a leaf node (by traversing from root)
  size_t findChildIndexForLeaf(BTreeLeaf<Key, Value>* target_leaf) {
    // This is a simplified implementation - in practice, we might want to cache this
    // or track it during traversal. For now, we'll search from root.
    if (promoted_child_count == 0) {
      return SIZE_MAX;
    }
    
    NodeBase* node = root;
    size_t child_index = 0;
    
    // Traverse to find which leaf index this is
    // This is inefficient but works for now
    std::function<size_t(NodeBase*, size_t&)> countLeaves = [&](NodeBase* n, size_t& index) -> size_t {
      if (n->type == PageType::BTreeLeaf) {
        if (static_cast<BTreeLeaf<Key,Value>*>(n) == target_leaf) {
          return index++;
        }
        return index++;
      } else {
        auto* inner = static_cast<BTreeInner<Key>*>(n);
        size_t count = 0;
        for (unsigned i = 0; i <= inner->count; i++) {
          count += countLeaves(inner->children[i], index);
          if (index > child_index && static_cast<BTreeLeaf<Key,Value>*>(inner->children[i]) == target_leaf) {
            child_index = index - 1;
            return count;
          }
        }
        return count;
      }
    };
    
    size_t idx = 0;
    countLeaves(node, idx);
    
    if (child_index < promoted_child_count) {
      return child_index;
    }
    return SIZE_MAX;
  }
  
  // Remove segment from promoted segments array
  void removePromotedSegment(ribtree::Segment<Key, Value>* seg, size_t child_index) {
    if (child_index >= promoted_child_count || promoted_bitmap[child_index] == 0) {
      return;
    }
    
    bool needRestart = false;
    promoted_lock.writeLockOrRestart(needRestart);
    if (needRestart) {
      return;  // Lock failed, retry later
    }
    
    // Find and remove the segment
    SegmentIndexRange range = promoted_index[child_index];
    for (size_t i = range.start; i <= range.end; i++) {
      if (promoted_segments[i] == seg) {
        promoted_segments[i] = nullptr;  // Mark as deleted
        
        // Check if this child has any other segments
        bool has_other = false;
        for (size_t j = range.start; j <= range.end; j++) {
          if (promoted_segments[j] != nullptr) {
            has_other = true;
            break;
          }
        }
        
        if (!has_other) {
          promoted_bitmap[child_index] = 0;
        }
        
        break;
      }
    }
    
    promoted_lock.writeUnlock();
  }

  // Load segments from config file and build tree structure
  void loadConfigByFile(const std::string& config_file) {
    std::cout << "[LOAD_CONFIG] ========== Starting loadConfigByFile ==========" << std::endl;
    std::cout << "[LOAD_CONFIG] Opening config file: " << config_file << std::endl;
    std::cout.flush();
    
    // First, count total lines for progress bar
    std::ifstream count_stream(config_file);
    if (!count_stream.is_open()) {
      throw std::runtime_error("Failed to open config file: " + config_file);
    }
    size_t total_lines = 0;
    std::string dummy_line;
    while (getline(count_stream, dummy_line)) {
      total_lines++;
    }
    count_stream.close();
    std::cout << "[LOAD_CONFIG] Config file has " << total_lines << " lines" << std::endl;
    std::cout.flush();
    
    // Now open file for actual reading
    std::ifstream config(config_file);
    if (!config.is_open()) {
      throw std::runtime_error("Failed to open config file: " + config_file);
    }
    std::cout << "[LOAD_CONFIG] Config file opened successfully" << std::endl;
    std::cout.flush();

    // Step 1: Read all segments from config file
    std::vector<std::pair<Key, ribtree::Segment<Key, Value>*>> segment_list;
    std::string line;
    size_t segment_count = 0;
    size_t line_count = 0;
    size_t progress_interval = std::max(static_cast<size_t>(1), total_lines / 100);  // Update every 1%
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "[LOAD_CONFIG] Starting to read segments from config file..." << std::endl;
    std::cout.flush();
    while (getline(config, line)) {
      line_count++;
      
      // Print progress bar every 1%
      if (line_count % progress_interval == 0 || line_count == total_lines) {
        auto current_time = std::chrono::high_resolution_clock::now();
        long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
        double progress = 100.0 * line_count / total_lines;
        long long elapsed_ms = std::max(elapsed, 1LL);
        double rate = line_count * 1000.0 / elapsed_ms;  // lines per second
        
        std::cout << "\r[LOAD_CONFIG] Progress: " << std::fixed << std::setprecision(1) << progress 
                  << "% (" << line_count << "/" << total_lines << " lines, " << segment_count << " segments) | "
                  << "Rate: " << std::setprecision(0) << rate / 1000.0 << " K lines/s"
                  << std::flush;
      }
      if (line.empty()) continue;
      std::istringstream iss(line);
      std::string token = "";
      Key lower = Key{};  // Initialize to default value
      Key upper = Key{};  // Initialize to default value
      size_t box_range = 0;
      size_t keys_count = 0;  // Optional field, default to 0 if not present

      if (getline(iss, token, ',')) {
        if constexpr (std::is_same_v<Key, double>) {
          lower = std::stod(token);
        } else if constexpr (std::is_signed_v<Key>) {
          lower = std::stoll(token);
        } else {
          lower = std::stoull(token);
        }
      }
      if (getline(iss, token, ',')) {
        if constexpr (std::is_same_v<Key, double>) {
          upper = std::stod(token);
        } else if constexpr (std::is_signed_v<Key>) {
          upper = std::stoll(token);
        } else {
          upper = std::stoull(token);
        }
      }
      if (getline(iss, token, ',')) {
        box_range = std::stoul(token);
      } else {
        throw std::runtime_error("Missing box_range in config file. Line: " + line);
      }
      // Read optional keys_count field (for backward compatibility, it's optional)
      if (getline(iss, token)) {
        keys_count = std::stoull(token);
      }
      
      // Validate box_range to prevent division by zero in Segment constructor
      if (box_range == 0) {
        throw std::runtime_error("Invalid box_range: 0 in config file. Line: " + line);
      }
      
      // Validate segment range
      if (lower >= upper) {
        throw std::runtime_error("Invalid segment range: lower >= upper in config file. Line: " + line);
      }

      try {
        auto* seg = new ribtree::Segment<Key, Value>(lower, upper, box_range, thread_num);
        segment_list.push_back({lower, seg});  // Store (lower_bound, segment) pair
        // Store keys_count in map for later use in promotion
        segment_keys_count_map[seg] = keys_count;
        segment_count++;
      } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create segment from config line: " + line + ". Error: " + e.what());
      }
    }
    std::cout << std::endl;  // New line after progress bar
    std::cout << "[LOAD_CONFIG] Finished reading config file. Total segments created: " << segment_count << std::endl;
    
    // Step 2: Build tree structure from segments
    std::cout << "[BULK LOAD] Building tree structure from " << segment_list.size() << " segments..." << std::endl;
    buildTreeFromSegments(segment_list);
    std::cout << "[BULK LOAD] Tree structure built successfully." << std::endl;
    
    // Step 3: Print tree structure for debugging (after tree is built)
    // Skip tree structure printing for large datasets as it can be very slow
    // printTreeStructure("tree_structure.txt");
  }

  // Print tree structure to file for debugging
  void printTreeStructure(const std::string& output_file) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to open output file: " + output_file);
    }
    
    out << "=== B-Tree Structure Analysis ===\n\n";
    
    // Helper function to recursively print nodes
    std::function<void(NodeBase*, int, int&)> printNode = 
      [&](NodeBase* node, int level, int& node_id) {
        int current_id = node_id++;
        std::string indent(level * 2, ' ');
        
        if (node->type == PageType::BTreeInner) {
          auto* inner = static_cast<BTreeInner<Key>*>(node);
          out << indent << "Inner Node #" << current_id << " (level " << level << "):\n";
          out << indent << "  Keys array (count=" << inner->count << "): [";
          for (unsigned i = 0; i < inner->count; i++) {
            out << inner->keys[i];
            if (i < inner->count - 1) out << ", ";
          }
          out << "]\n";
          out << indent << "  Children count: " << (inner->count + 1) << "\n";
          for (unsigned i = 0; i <= inner->count; i++) {
            out << indent << "  -> Child[" << i << "]:\n";
            printNode(inner->children[i], level + 1, node_id);
          }
        } else {
          auto* leaf = static_cast<BTreeLeaf<Key,Value>*>(node);
          out << indent << "Leaf Node #" << current_id << " (level " << level << "):\n";
          out << indent << "  Keys array (count=" << leaf->count << "): [";
          for (unsigned i = 0; i < leaf->count; i++) {
            out << leaf->keys[i];
            if (i < leaf->count - 1) out << ", ";
          }
          out << "]\n";
          out << indent << "  Segments:\n";
          for (unsigned i = 0; i < leaf->count; i++) {
            if (leaf->segments[i]) {
              out << indent << "    [" << i << "] key=" << leaf->keys[i] 
                  << ", segment range=[" << leaf->segments[i]->getLowerBound() 
                  << ", " << leaf->segments[i]->getUpperBound() << ")\n";
            } else {
              out << indent << "    [" << i << "] key=" << leaf->keys[i] 
                  << ", segment=NULL\n";
            }
          }
          out << "\n";
        }
      };
    
    int node_id = 0;
    NodeBase* root_node = root.load();
    printNode(root_node, 0, node_id);
    
            out << "\n=== End of Tree Structure ===\n";
            out.close();
  }


};

}
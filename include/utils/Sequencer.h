// Sequencer.h
#pragma once
#include <map>
#include <mutex>
#include <condition_variable>
#include <cstddef>   // size_t
#include "ThreadSafeQueue.h"
template<typename IndexT = std::size_t,
         typename ValueT = void,
         typename QueueT = ThreadSafeQueue<ValueT>>
class Sequencer
{
public:
    explicit Sequencer(IndexT start_index,
                       QueueT& out_queue)
        : next_(start_index),
          outQ_(out_queue)
    {}

    // non-copyable
    Sequencer(const Sequencer&)            = delete;
    Sequencer& operator=(const Sequencer&) = delete;

    /** Push a (index, value) pair coming from any worker thread. */
    void push(IndexT idx, ValueT&& value)
    {
        std::unique_lock<std::mutex> lk(m_);

        // If the index is greater than next_, add it to pending_
        if (idx < next_)
        {
            std::cerr << "Warning: index " << idx << " is less than next_ " << next_ << std::endl;
        }else{
            pending_.emplace(idx, std::move(value));    
        }

        // Emit every contiguous element that starts at next_
        while (!pending_.empty() &&
               pending_.begin()->first == next_)
        {
            outQ_.enqueue( std::move(pending_.begin()->second) );
            pending_.erase(pending_.begin());
            ++next_;
        }
        cv_.notify_all();  // if a waiter cares
    }

    /** Block the caller until every index ≤ max_index is flushed. */
    void wait_until(IndexT max_index)
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return next_ > max_index; });
    }

    /** Flush all pending results to the output queue without waiting for next_. */
    void flush()
    {
        std::unique_lock<std::mutex> lk(m_);
        while (!pending_.empty())
        {
        auto node = pending_.extract(pending_.begin());   // C++17 node-handle
        lk.unlock();                                      // <-- release lock
        outQ_.enqueue(std::move(node.mapped()));          // may block
        lk.lock();
        ++next_;                                          // advance in sequence
        }

        cv_.notify_all();
    }

private:
    IndexT                     next_;     // next index we must emit
    std::map<IndexT,ValueT>    pending_;  // results that arrived early
    QueueT&                    outQ_;

    std::mutex                 m_;
    std::condition_variable    cv_;
};
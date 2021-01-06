/* *
* Stopwatch for performance measuring purpose.
* 
 * @author  tyouhyou    github.com/tyouhyou
 * @license MIT
 * */

#pragma once

#include <chrono>
#include <functional>

namespace zb
{

    class StopWatch
    {
    public:
        StopWatch &start()
        {
            startpoint = std::chrono::high_resolution_clock::now();
            lastpoint = startpoint;
            return *this;
        }

        long long elaspsed()
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startpoint).count();
        }

        long long wrap()
        {
            auto temppoint = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(temppoint - lastpoint);
            lastpoint = temppoint;
            return dur.count();
        }

        StopWatch &reset()
        {
            return start();
        }

    private:
        std::chrono::high_resolution_clock::time_point startpoint;
        std::chrono::high_resolution_clock::time_point lastpoint;
    };

} // namespace zb
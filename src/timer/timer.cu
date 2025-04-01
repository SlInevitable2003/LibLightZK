#include "timer.cuh"

void Timer::start() { start_time = std::chrono::high_resolution_clock::now(); }

double Timer::stop(const std::string& label) 
{
    double duration_sec = elapsed();
    printf("\033[35m[Timing]\033[0m %s in: %.9f seconds\n", label.c_str(), duration_sec);
    return duration_sec;
}

double Timer::elapsed() 
{
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time);
    return duration.count() / (1e6);
}
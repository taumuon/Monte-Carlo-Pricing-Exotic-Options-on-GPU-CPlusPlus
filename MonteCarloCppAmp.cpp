#define NOMINMAX

#include <amp.h>
#include <iostream>
#include <chrono>
#include <tuple>
#include <random>
#include <algorithm>
#include <numeric>

#define _USE_MATH_DEFINES
#include "math.h"

#include "amp_tinymt_rng.h"
#include "amp_sobol_rng.h"

#include <amp_algorithms.h>
#include <amp_stl_algorithms.h>

#include "amp_math.h"

using namespace amp_algorithms;
using namespace amp_stl_algorithms;

using namespace concurrency;

class NormalDistributedPair
{
public:
	float norm0;
	float norm1;
};

// Box Muller from GPU Gems 3
NormalDistributedPair BoxMuller(float u0, float u1) restrict(amp)
{
	auto rx = fast_math::sqrt(-2.0f * fast_math::log(u0));
	auto theta = 2.0f * static_cast<float>(M_PI) * u1;
	auto norm0 = rx * fast_math::sin(theta);
	auto norm1 = rx * fast_math::cos(theta);

	NormalDistributedPair normalDistributedPair;
	normalDistributedPair.norm0 = norm0;
	normalDistributedPair.norm1 = norm1;
	return normalDistributedPair;
}

float CalculateNextPoint(float S, float r, float sigma, float deltaT, float norm) restrict(amp)
{
	return S * fast_math::exp(((r - (0.5f * sigma * sigma)) * deltaT) + (sigma * fast_math::sqrt(deltaT) * norm));
}

std::tuple<float, float> CalcMeanStdDevCPU(std::vector<float> input)
{
	auto sum = std::accumulate(begin(input), end(input), 0.0f);
	auto mean = sum / input.size();

	auto accum = 0.0f;
	std::for_each(begin(input), end(input), [&](const float d)
	{
		accum += (d - mean) * (d - mean);
	});
	auto stddev = std::sqrt(accum / input.size());

	return std::make_tuple(mean, stddev);
}

void DoubleBarrierCPU()
{
	auto start = std::chrono::steady_clock::now();

	std::random_device rd;
	std::mt19937 e2(rd());
	std::normal_distribution<> dist(0.0, 1.0);

	const int numTrajectories = 1000;
	auto numSamples = 10000;

	auto S0 = 100.0f;
	auto strike = 90.0f;
	auto upperBarrier = 160.0f;
	auto lowerBarrier = 75.0f;
	auto r = 0.05f;
	auto T = 0.5f;
	auto sigma = 0.4f;

	auto payoffs = std::vector<float>(numTrajectories);

	for (int trajectoryIndex = 0; trajectoryIndex < numTrajectories; ++trajectoryIndex)
	{
		auto numSamples = 10000;
		auto knockedOut = false;
		auto deltaT = T / static_cast<float>(numSamples);
		auto S = S0;
		auto last = 0.0f;

		for (int sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex)
		{
			auto norm = dist(e2);

			S = S * std::exp(((r - (0.5f * sigma * sigma)) * deltaT) + (sigma * std::sqrt(deltaT) * static_cast<float>(norm)));
			if (S < lowerBarrier || S > upperBarrier)
			{
				knockedOut = true;
				break;
			}
			last = S;
		}

		auto payoff = last - strike;
		payoffs[trajectoryIndex] = knockedOut ? 0.0f : std::max(payoff, 0.0f);
	}

	double mean, stddev;
	std::tie(mean, stddev) = CalcMeanStdDevCPU(payoffs);

	auto discountFactor = exp(-r * T);
	auto priceMC = discountFactor * mean;
	auto stddevMC = discountFactor * stddev / sqrt(static_cast<double>(payoffs.size()));

	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;

	std::cout << "double barrier CPU. price: " << priceMC << " stddev:" << stddevMC << " time: " << std::chrono::duration<double, std::milli>(diff).count() << std::endl;
}

void DoubleBarrier(int seed)
{
	auto start = std::chrono::steady_clock::now();

    const int rank = 1;
	const int numTrajectories = 1000;
	extent<rank> e_size(numTrajectories);
    tinymt_collection<rank> myrand(e_size, seed);
    array<float, rank> payoffs(e_size);

	auto S0 = 100.0f;
	auto strike = 90.0f;
	auto upperBarrier = 160.0f;
	auto lowerBarrier = 75.0f;
	auto r = 0.05f;
	auto T = 0.5f;
	auto sigma = 0.4f;

    parallel_for_each(e_size, [=, &payoffs] (index<1> idx) restrict(amp)
    {
        auto t = myrand[idx];

		auto numSamples = 10000;
		auto numIterations = numSamples / 2;

		auto knockedOut = false;

		auto deltaT = T / static_cast<float>(numSamples);
		auto S = S0;
		auto last = 0.0f;

		for (int i = 0; i < numIterations; ++i)
		{
			auto u0 = t.next_single();
			auto u1 = t.next_single();
			auto normalDistributedPair = BoxMuller(u0, u1);

			S = CalculateNextPoint(S, r, sigma, deltaT, normalDistributedPair.norm0);
			if (S < lowerBarrier || S > upperBarrier)
			{
				knockedOut = true;
			}
			last = S;
			S = CalculateNextPoint(S, r, sigma, deltaT, normalDistributedPair.norm1);
			if (S < lowerBarrier || S > upperBarrier)
			{
				knockedOut = true;
			}
			last = S;
		}

		auto profit = last - strike;
		payoffs[idx] = knockedOut ? 0.0f : (profit < 0.0f ? 0.0f : profit);
    });

	auto payoffsView = payoffs.view_as(e_size);
	auto total = amp_algorithms::reduce(payoffsView, amp_algorithms::plus<float>());
	auto mean = total / static_cast<float>(numTrajectories);
	auto squaredDifferences = array<float, rank>(e_size);
	parallel_for_each(e_size, [=, &payoffs, &squaredDifferences](index<1> idx) restrict(amp)
	{
		auto val = payoffs[idx];
		squaredDifferences[idx] = (val - mean) * (val - mean);
	});
	auto squaredDifferenceView = squaredDifferences.view_as(e_size);
	auto sumSquaredDifferences = amp_algorithms::reduce(squaredDifferenceView, amp_algorithms::plus<float>());

	auto stddev = std::sqrt(sumSquaredDifferences / static_cast<float>(numTrajectories));

	auto discountFactor = std::exp(-r * T);
	auto priceMC = discountFactor * mean;
	auto stddevMC = discountFactor * stddev / std::sqrt(static_cast<double>(numTrajectories));

	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;

	std::cout << "double barrier price: " << priceMC << " stddev:" << stddevMC << " time: " << std::chrono::duration<double, std::milli>(diff).count() << std::endl;
}

int main()
{
    accelerator default_device;
    std::wcout << L"Using device : " << default_device.get_description() << std::endl;
    if (default_device == accelerator(accelerator::direct3d_ref))
        std::cout << "WARNING!! Running on very slow emulator! Only use this accelerator for debugging." << std::endl;

	DoubleBarrierCPU();

	int seed = 5489;
    DoubleBarrier(seed);
	seed = 2359;
	DoubleBarrier(seed);

    return 0;
}

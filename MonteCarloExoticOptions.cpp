#include "stdafx.h"

#include <iostream>
#include <chrono>
#include <tuple>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>

using namespace std;
using namespace placeholders;

auto callPayoff = [](double strike, double price) { return std::max(price - strike, 0.0); };

auto europeanPayoff = [](std::function<double(double strike, double price)> payoff, vector<double> assetPath)
{
	return std::bind(payoff, _1, assetPath[assetPath.size() - 1]);
};

auto europeanCallPayoff = [](double strike, vector<double> assetPath)
{
	return europeanPayoff(callPayoff, assetPath)(strike);
};

auto asianArithmeticMeanPayoff = [](double strike, vector<double> assetPath, std::function<double(double strike, double price)> payoff)
{
	auto assetPathMean = std::accumulate(begin(assetPath), end(assetPath), 0.0) / assetPath.size();
	return payoff(strike, assetPathMean);
};

auto asianArithmeticMeanCallPayoff = std::bind(asianArithmeticMeanPayoff, _1, _2, callPayoff);

auto doubleBarrierPayoff = [](std::function<double(vector<double> assetPath)> payoff, double upperBarrier, double lowerBarrier, vector<double> assetPath)
{
	auto result = std::find_if(begin(assetPath), end(assetPath), [&](double i)
	{
		return i > upperBarrier || i < lowerBarrier;
	});
	if (result != (end(assetPath))) { return 0.0; }

	return payoff(assetPath);
};

auto doubleBarrierEuropeanCallPayoff = [](double strike, double upperBarrier, double lowerBarrier, vector<double> assetPath)
{
	auto pathPayoffFunction = std::bind(europeanCallPayoff, strike, _1);
	return doubleBarrierPayoff(pathPayoffFunction, upperBarrier, lowerBarrier, assetPath);
};

std::tuple<double, double> CalcMeanStdDev(vector<double> input)
{
	auto sum = std::accumulate(begin(input), end(input), 0.0);
	auto mean = sum / input.size();

	auto accum = 0.0;
	std::for_each(begin(input), end(input), [&](const double d)
	{
		accum += (d - mean) * (d - mean);
	});
	auto stddev = sqrt(accum / input.size());

	return std::make_tuple(mean, stddev);
}

std::tuple<double, double> CalculateOptionValue(double S0, double r, double T, double sigma, int numTrajectories, int numSamples, std::function<double(vector<double> path)> payoffFunction)
{
	std::random_device rd;
	std::mt19937 e2(rd());
	std::normal_distribution<> dist(0.0, 1.0);

	auto deltaT = static_cast<double>(T) / static_cast<double>(numSamples);

	std::vector<double> payoffs;
	for (int trajectoryIndex = 0; trajectoryIndex < numTrajectories; ++trajectoryIndex)
	{
		std::vector<double> assetPath;
		auto S = S0;
		for (int i = 0; i < numSamples; ++i)
		{
			auto norm = dist(e2);
			auto newS = (S * exp(((r - (0.5 * sigma * sigma)) * deltaT) + (sigma * sqrt(deltaT) * norm)));
			S = newS;
			assetPath.push_back(newS);
		}

		auto payoff = payoffFunction(assetPath);
		payoffs.push_back(payoff);
	}

	double mean, stddev;
	std::tie(mean, stddev) = CalcMeanStdDev(payoffs);

	auto discountFactor = exp(-r * T);
	auto priceMC = discountFactor * mean;
	auto stddevMC = discountFactor * stddev / sqrt(static_cast<double>(payoffs.size()));
	return std::make_tuple(priceMC, stddevMC);
}

void RunAsianArithmeticMean()
{
	auto S0 = 100.0;
	auto strike = 90.0;
	auto r = 0.05;
	auto T = 1.0;
	auto sigma = 0.2;
	auto numTrajectories = 100000;
	auto numSamples = 12;

	auto payoffFunction = std::bind(asianArithmeticMeanCallPayoff, strike, _1);
	auto start = chrono::steady_clock::now();
	auto result = CalculateOptionValue(S0, r, T, sigma, numTrajectories, numSamples, payoffFunction);
	auto end = chrono::steady_clock::now();
	auto diff = end - start;

	double price;
	double stddev;
	std::tie(price, stddev) = result;
	cout << "Asian arithmetic mean: " << price << " stddev:" << stddev << " time: " << chrono::duration<double, milli>(diff).count() << endl;
}

void RunDoubleBarrier()
{
	auto S0 = 100.0;
	auto strike = 90.0;
	auto upperBarrier = 160.0;
	auto lowerBarrier = 75.0;
	auto r = 0.05;
	auto T = 0.5;
	auto sigma = 0.4;
	auto numTrajectories = 1000;
	auto numSamples = 10000;

	auto payoffFunction = std::bind(doubleBarrierEuropeanCallPayoff, strike, upperBarrier, lowerBarrier, _1);
	auto start = chrono::steady_clock::now();
	auto result = CalculateOptionValue(S0, r, T, sigma, numTrajectories, numSamples, payoffFunction);
	auto end = chrono::steady_clock::now();
	auto diff = end - start;

	double price;
	double stddev;
	std::tie(price, stddev) = result;
	cout << "Double barrier: " << price << " stddev:" << stddev << " time: " << chrono::duration<double, milli>(diff).count() << endl;
}

int _tmain(int argc, _TCHAR* argv[])
{
	RunAsianArithmeticMean();

	RunDoubleBarrier();

	return 0;
}
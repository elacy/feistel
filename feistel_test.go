package feistel

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

const (
	maxTestRange       = 1000000
	minRoundsForRandom = 7
)

type testSettings struct {
	maxValue   uint64
	rounds     uint8
	startIndex uint64
	endIndex   uint64
	epochs     bool
}

func (s testSettings) String() string {
	return fmt.Sprintf("maxValue %d, range: %d -> %d, rounds: %d, epochs: %t", s.maxValue, s.startIndex, s.endIndex, s.rounds, s.epochs)
}

type testResult struct {
	settings   *testSettings
	forwardMap map[uint64]uint64
	reverseMap map[uint64]uint64
	err        error
}

var maxValuesToTest = []uint64{13, 16, 64, 101, 1000, 1024, 50_000}
var optionsToTest = []testSettings{
	{
		rounds: 1,
	},
	{
		rounds: 3,
	},
	{
		rounds: 5,
	},
	{
		rounds: 6,
	},
	{
		rounds: 8,
	},
	{
		rounds: 30,
	},
}

var cachedResults []*testResult

func buildTestSettings() []*testSettings {
	result := make([]*testSettings, len(maxValuesToTest)*len(optionsToTest)*2)
	i := 0
	for _, epochs := range []bool{false, true} {
		for _, maxValue := range maxValuesToTest {
			for _, settings := range optionsToTest {
				settings.maxValue = maxValue
				settings.epochs = epochs
				result[i] = &settings
				i++
			}
		}
	}

	return result
}

func TestMain(m *testing.M) {
	for _, arg := range os.Args {
		if strings.HasPrefix(arg, "-test.bench") {
			os.Exit(m.Run())
		}
	}

	cachedResults = make([]*testResult, len(maxValuesToTest)*len(optionsToTest)*2)
	resultChan := make(chan *testResult, 200)
	var wg sync.WaitGroup

	go func() {
		semaphoreChannel := make(chan struct{}, runtime.NumCPU())
		for _, settings := range buildTestSettings() {
			currentSettings := settings
			semaphoreChannel <- struct{}{}
			wg.Add(1)
			go func() {
				resultChan <- runTest(currentSettings)
				wg.Done()
				<-semaphoreChannel
			}()
		}
		wg.Wait()
		close(resultChan)
		close(semaphoreChannel)
	}()

	i := 0
	for result := range resultChan {
		cachedResults[i] = result
		i++
	}

	os.Exit(m.Run())
}

func createNetwork(settings *testSettings, seed uint64) (*Network, error) {
	var options []Option

	if settings.epochs {
		options = append(options, WithEpochs())
	}

	if settings.endIndex == 0 {
		settings.endIndex = settings.maxValue
	}

	return NewNetwork(settings.maxValue, seed, settings.rounds, options...)
}

func runTest(settings *testSettings) *testResult {
	result := &testResult{
		settings: settings,
	}

	domainSize := settings.maxValue + 1
	replaceNet := false

	if settings.maxValue > maxTestRange {
		settings.endIndex = maxTestRange
	} else {
		settings.endIndex = maxTestRange - (maxTestRange % domainSize)
		replaceNet = !settings.epochs
	}

	if settings.endIndex == 0 {
		settings.endIndex = settings.maxValue
	}

	var net *Network
	var err error

	result.forwardMap = make(map[uint64]uint64, settings.endIndex+1-settings.startIndex)
	result.reverseMap = make(map[uint64]uint64, settings.endIndex+1-settings.startIndex)

	for i := settings.startIndex; i <= settings.endIndex; i++ {
		if net == nil || (replaceNet && (i%domainSize == 0)) {
			net, err = createNetwork(settings, i/domainSize)

			if err != nil {
				result.err = err
				return result
			}
		}

		value := i
		adjust := uint64(0)

		if !settings.epochs {
			value %= domainSize
			adjust = i - value
		}

		mappedValue, err := net.Map(value)

		if err != nil {
			result.err = fmt.Errorf("Error Mapping %d, %w", i, err)
			return result
		}

		result.forwardMap[i] = mappedValue + adjust
		inverted, err := net.InvertMap(mappedValue)
		result.reverseMap[mappedValue+adjust] = inverted + adjust

		if err != nil {
			result.err = fmt.Errorf("Error inverting %d, %w", i, err)
			return result
		}
	}

	return result
}

func TestValuesWithinRange(t *testing.T) {
	for _, test := range cachedResults {
		t.Run(test.settings.String(), func(t *testing.T) {
			for original, value := range test.forwardMap {
				domainSize := test.settings.maxValue + 1
				start := original - (original % domainSize)

				if value < start {
					t.Errorf("Epoch mapped value %d is below start index %d", value, start)
				} else if value > start+test.settings.maxValue {
					t.Errorf("Epoch mapped value %d is above end index %d", value, start+test.settings.maxValue)
				}
			}
		})
	}
}

func TestInvertible(t *testing.T) {
	for _, test := range cachedResults {
		t.Run(test.settings.String(), func(t *testing.T) {
			for original, mappedValue := range test.forwardMap {
				if inverted, ok := test.reverseMap[mappedValue]; ok {
					if original != inverted {
						t.Errorf("Mapped %d to %d and inversion produced %d", original, mappedValue, inverted)
						return
					}
				} else {
					t.Errorf("Inversion not performed for %d which mapped to %d", original, mappedValue)
					return
				}
			}
		})
	}
}

func TestNoErr(t *testing.T) {
	for _, test := range cachedResults {
		t.Run(test.settings.String(), func(t *testing.T) {
			if test.err != nil {
				t.Errorf("Error occured %v", test.err)
			}
		})
	}
}

func TestAverageFixedPoints(t *testing.T) {
	for _, tt := range cachedResults {
		test := tt
		if test.settings.maxValue < 20 {
			continue
		}

		if test.settings.rounds < minRoundsForRandom {
			continue
		}

		domainSize := test.settings.maxValue + 1
		// If the total number of maps is not a factor of domain size this will be a floating point number
		// which is fine since the average should still hold
		permutations := float64(len(test.forwardMap)) / float64(domainSize)

		if permutations < 100 {
			continue
		}

		t.Run(test.settings.String(), func(t *testing.T) {
			fixedPoints := 0
			zeroFixedPoints := 0

			for original, mapped := range test.forwardMap {
				if original == mapped {
					fixedPoints++

					if original == 0 {
						zeroFixedPoints++
					}
				}
			}

			average := float64(fixedPoints) / permutations

			if math.Abs(average-1) > 0.1 {
				t.Errorf("Average fixed points not in valid range (between 0.9 and 1.1) total %d, fixed points %d, average %f", len(test.forwardMap), fixedPoints, average)
			}

			if 2*float64(fixedPoints)/float64(domainSize) < float64(zeroFixedPoints) {
				t.Errorf("Disproportionate number of fixed points: %d, fixed points: %d, domain size: %d", zeroFixedPoints, fixedPoints, domainSize)
			}
		})
	}
}

func TestFullRangeCovered(t *testing.T) {
	for _, test := range cachedResults {
		if test.settings.endIndex != test.settings.maxValue || test.settings.startIndex != 0 {
			continue
		}
		t.Run(test.settings.String(), func(t *testing.T) {
			seen := make(map[uint64]uint64, test.settings.maxValue+1)

			for original, mapped := range test.forwardMap {
				if previous, ok := seen[mapped]; ok {
					t.Errorf("both %d and %d are mapped to %d in forward map", previous, original, mapped)
					return
				}

				seen[mapped] = original
			}

			for i := range test.settings.maxValue + 1 {
				if _, ok := seen[i]; !ok {
					t.Errorf("%d was not covered in forward map", i)
					return
				}
			}
		})
	}
}

func TestCycleLengths(t *testing.T) {
	for _, test := range cachedResults {
		if test.settings.rounds < 3 || test.settings.maxValue > test.settings.endIndex || test.settings.epochs || test.settings.startIndex != 0 {
			continue
		}
		t.Run(test.settings.String(), func(t *testing.T) {
			var cycles []int
			cycleLength := 0
			seen := make(map[uint64]struct{}, len(test.forwardMap))

			for i := uint64(0); i <= test.settings.maxValue; i++ {
				currentValue := i
				if _, ok := seen[currentValue]; ok {
					continue
				}
				for {
					seen[currentValue] = struct{}{}
					cycleLength++
					currentValue = test.forwardMap[currentValue]

					if _, ok := seen[currentValue]; ok {
						cycles = append(cycles, cycleLength)
						cycleLength = 0
						break
					}
				}
			}

			maxCycles := maxAllowedCycles(len(test.forwardMap), 3)
			if len(cycles) > maxCycles {
				t.Errorf("Number of cycles (%d) exceeds max %d for n %d", len(cycles), maxCycles, len(test.forwardMap))
				println(fmt.Sprintf("Cycles are %v", cycles))
			}
		})
	}
}

// eulerMascheroni is γ ≈ 0.5772156649
const eulerMascheroni = 0.5772156649015329

// MaxAllowedCycles returns ceil( ln(n) + γ + z*sqrt(ln(n)) ).
// Use z=3 for a lenient (~3σ) upper bound. n is the domain size (count of elements), not the max index.
func maxAllowedCycles(n int, z float64) int {
	if n < 2 {
		// For n=0 or n=1, the only possible permutation has exactly 0 or 1 cycle respectively.
		return int(n)
	}
	ln := math.Log(float64(n))
	mean := ln + eulerMascheroni
	sd := math.Sqrt(ln)
	return int(math.Ceil(mean + z*sd))
}

func TestChiSquare(t *testing.T) {
	for _, test := range cachedResults {
		if (test.settings.rounds < 3) || len(test.forwardMap) < 1000 {
			continue
		}
		t.Run(test.settings.String(), func(t *testing.T) {
			mappedValues := mapValues(test.forwardMap)
			hist := histogram(mappedValues[0:len(mappedValues)/2], test.settings.maxValue+1, 64)
			chiSquaredTest(t, hist, 0.1)
		})
	}
}

func histogram(samples []uint64, n uint64, bins uint64) []uint64 {
	hist := make([]uint64, bins)
	for _, v := range samples {
		if v >= n {
			continue // skip invalid
		}
		b := (uint64(bins) * v) / n
		if b >= bins {
			b = bins - 1
		}
		hist[b]++
	}
	return hist
}

func chiSquaredTest(t *testing.T, values []uint64, threshold float64) {
	t.Helper()

	floatValues := castToFloat64(values)
	mean := stat.Mean(floatValues, nil)
	averages := repeat(mean, len(values))
	chi2 := stat.ChiSquare(floatValues, averages)

	p := distuv.ChiSquared{K: float64(len(values) - 1)}.Survival(chi2)

	if threshold > p {
		t.Errorf("Unven Distribution, p is %f, values: %v", p, values)
	}
}

func castToFloat64(values []uint64) []float64 {
	result := make([]float64, len(values))

	for i := range len(values) {
		result[i] = float64(values[i])
	}

	return result
}

func repeat[T any](value T, count int) []T {
	result := make([]T, count)

	for i := range count {
		result[i] = value
	}

	return result
}

func mapValues[K comparable, V any](input map[K]V) []V {
	result := make([]V, len(input))

	i := 0

	for _, value := range input {
		result[i] = value
		i++
	}

	return result
}

func TestSerialCorrelationLag1(t *testing.T) {
	for _, test := range cachedResults {
		if test.settings.startIndex != 0 || test.settings.endIndex != test.settings.maxValue {
			continue
		}

		if test.settings.rounds < 5 {
			continue
		}

		t.Run(test.settings.String(), func(t *testing.T) {
			pairCount := test.settings.maxValue
			current := make([]float64, pairCount)
			next := make([]float64, pairCount)

			for i := range pairCount {
				current[i] = float64(test.forwardMap[i])
				next[i] = float64(test.forwardMap[i+1])
			}

			corr := stat.Correlation(current, next, nil)
			if math.IsNaN(corr) {
				t.Fatal("correlation is NaN")
			}

			// Expected mean correlation for permutations (slight negative).
			expected := -1.0 / float64(pairCount)

			// Fisher z transform: z = atanh(r); SE ≈ 1/sqrt(n-3)
			// Compare (z - z_expected) to 3σ.
			z := 0.5 * math.Log((1+corr)/(1-corr))
			zExp := 0.5 * math.Log((1+expected)/(1-expected))
			se := 1.0 / math.Sqrt(float64(pairCount-2))
			zScore := (z - zExp) / se

			if math.Abs(zScore) > 3.0 {
				t.Errorf("lag-1 correlation out of band: corr=%.5f, z=%.2f (expected≈%.5f, SE≈%.3f)",
					corr, zScore, expected, se)
			}
		})
	}
}

func TestResidueBiasSmallPrimes(t *testing.T) {
	for _, test := range cachedResults {
		// Only meaningful when we've mapped the full domain [0..maxValue].
		if test.settings.startIndex != 0 || test.settings.endIndex < test.settings.maxValue {
			continue
		}

		t.Run(test.settings.String(), func(t *testing.T) {
			outputs := mapValues(test.forwardMap)

			significance := 0.01
			primes := []uint64{3, 5, 7, 11, 13, 17, 19, 23}

			for _, p := range primes {
				// Ensure enough mass so expected count per residue ≥ ~20
				if len(outputs) < int(p*20) {
					continue
				}
				// Count residues of y mod p across all outputs.
				counts := histogram(outputs, p, p) // your helper: n=p, bins=p
				chiSquaredTest(t, counts, significance)
			}
		})
	}
}

func TestUniqueCombinations(t *testing.T) {
	for _, test := range cachedResults {
		// Only meaningful when we've mapped the full domain [0..maxValue].
		if test.settings.endIndex < 2*test.settings.maxValue {
			continue
		}

		if test.settings.rounds < minRoundsForRandom {
			continue
		}
		t.Run(test.settings.String(), func(t *testing.T) {
			combinations := (test.settings.endIndex + 1 - test.settings.startIndex) / (test.settings.maxValue + 1)
			seen := make(map[string]struct{}, combinations)
			domainSize := test.settings.maxValue + 1
			buf := make([]byte, 8*domainSize)

			for i := range uint64(len(test.forwardMap)) {
				position := i % domainSize
				value := test.forwardMap[i] % domainSize

				bufStart := position * 8
				binary.BigEndian.PutUint64(buf[bufStart:bufStart+8], value)

				if position == domainSize-1 {
					seen[string(buf)] = struct{}{}
				}
			}

			outputs := mapValues(test.forwardMap)

			significance := 0.01
			primes := []uint64{3, 5, 7, 11, 13, 17, 19, 23}

			for _, p := range primes {
				// Ensure enough mass so expected count per residue ≥ ~20
				if len(outputs) < int(p*20) {
					continue
				}
				// Count residues of y mod p across all outputs.
				counts := histogram(outputs, p, p) // your helper: n=p, bins=p
				chiSquaredTest(t, counts, significance)
			}

			if len(seen) < int(combinations) {
				t.Errorf("Total combinations found was %d but total epochs was %d", len(seen), combinations)
			}
		})
	}
}

func TestSerialEpochCorrelationLag1(t *testing.T) {
	for _, test := range cachedResults {
		if test.settings.rounds < 5 {
			continue
		}

		if test.settings.endIndex < 100*(test.settings.maxValue+1) {
			continue
		}

		t.Run(test.settings.String(), func(t *testing.T) {
			combinations := (test.settings.endIndex + 1 - test.settings.startIndex) / (test.settings.maxValue + 1)
			domainSize := test.settings.maxValue + 1
			pairCount := combinations - 1
			current := make([]float64, pairCount)
			next := make([]float64, pairCount)

			for i := range pairCount {
				current[i] = float64(test.forwardMap[i*domainSize] % domainSize)
				next[i] = float64(test.forwardMap[(i+1)*domainSize] % domainSize)
			}

			corr := stat.Correlation(current, next, nil)
			if math.IsNaN(corr) {
				t.Fatal("correlation is NaN")
			}

			// Expected mean correlation for permutations (slight negative).
			expected := -1.0 / float64(pairCount)

			// Fisher z transform: z = atanh(r); SE ≈ 1/sqrt(n-3)
			// Compare (z - z_expected) to 3σ.
			z := 0.5 * math.Log((1+corr)/(1-corr))
			zExp := 0.5 * math.Log((1+expected)/(1-expected))
			se := 1.0 / math.Sqrt(float64(pairCount-2))
			zScore := (z - zExp) / se

			if math.Abs(zScore) > 3.0 {
				t.Errorf("lag-1 correlation out of band: corr=%.5f, z=%.2f (expected≈%.5f, SE≈%.3f)",
					corr, zScore, expected, se)
			}
		})
	}
}

func TestEpochDistribution(t *testing.T) {
	for _, test := range cachedResults {
		if test.settings.rounds < minRoundsForRandom {
			continue
		}

		if test.settings.maxValue > 20 {
			continue
		}

		t.Run(test.settings.String(), func(t *testing.T) {
			domainSize := test.settings.maxValue + 1
			counts := make([]uint64, domainSize*domainSize)

			for original, mapped := range test.forwardMap {
				original := original % domainSize
				mapped := mapped % domainSize
				index := (original * domainSize) + mapped
				counts[index]++
			}

			chiSquaredTest(t, counts, 0.01)
		})
	}
}

func BenchmarkAllSetting(b *testing.B) {
	for _, settings := range buildTestSettings() {
		b.Run(settings.String(), func(b *testing.B) {
			net, err := createNetwork(settings, 0)

			if err != nil {
				b.Fatalf("Unable to create network with error: %v", err)
			}
			for i := uint64(0); i < uint64(b.N); i++ {
				if net.epochs || net.maxValue == ^uint64(0) {
					_, err = net.Map(i)
				} else {
					_, err = net.Map(i % (settings.maxValue + 1))
				}

				if err != nil {
					b.Fatalf("Failed mapping with error %v", err)
				}
			}
		})
	}
}

func TestCoverage(t *testing.T) {
	for i := range 1024 {
		t.Run(fmt.Sprintf("check %d is covered", i), func(t *testing.T) {
			n, err := NewNetwork(uint64(i), 42, 5)
			if err != nil {
				t.Fatal(err)
			}

			results := make(map[uint64]uint8, i)

			for j := range i + 1 {
				point, err := n.Map(uint64(j))
				if err != nil {
					t.Fatal(err)
				}

				results[point]++
			}

			for j := range i + 1 {
				if results[uint64(j)] != 1 {
					t.Fatalf("Number %d covered for field %d %d times", j, i, results[uint64(j)])
				}
			}
		})
	}
}

// Package feistel is a non cryptographic implementation of the feistel cipher
// It's useful for creating random permutations of ranges of integers from 0 to n
package feistel

import (
	"errors"
	"fmt"
	"math"
)

// ErrIndexGreatThanMaxValue is returned when your index is greater than the maximum value
var ErrIndexGreatThanMaxValue = errors.New("feistel: index cannot be greater than max value")

// ErrRoundsMustBeSet is returned when you have provided a zero value for rounds
var ErrRoundsMustBeSet = errors.New("feistel: rounds must be a non zero value")

// Option is a function
type Option func(*Network)

// WithEpochs is an option that allows you to create multiple permutations with the same seed
// instead of providing an error when you exceed max value it will start a new sequence each time
// you cross a multiple of max value + 1. The mapped value will have the number at the start of the
// sequence added to it to maintain invertibility. For example with a max value of 9 there are 10 options
// so 9 is in the first sequence, 10 is the first index of the second sequence (or epoch), 21 is in the third
// sequence. If in the second sequence 0 maps to 9, then 10 will return 19. That way if you provide the 19 value
// we know we should consider it the second sequence (or epoch).
func WithEpochs() Option {
	return func(m *Network) {
		m.epochs = true
	}
}

// NewNetwork creates a new Feistel Network, keep in mind this function runs multiple hash functions to generate seeds
// maxValue is the maximum value in the sequence that the Feistel Network will create permutations for
// seed is a hash seed, by providing a different value this will return different permutations of the sequence
// rounds is the number of hash rounds the network will run to generate each value, the more rounds you run the
// better the distribution but it also adds time to running the function
// opts is a list of all the optional parameters you can add, right now that only includes WithEpochs()
func NewNetwork(maxValue, seed uint64, rounds uint8, opts ...Option) (*Network, error) {
	l, r := findFactors(maxValue)

	network := &Network{
		maxValue:   maxValue,
		rounds:     int(rounds),
		leftRadix:  l,
		rightRadix: r,
	}

	for _, opt := range opts {
		opt(network)
	}

	if network.rounds == 0 {
		return nil, ErrRoundsMustBeSet
	}

	network.seeds = make([]uint64, network.rounds)

	currentSeed := seed
	for i := range network.seeds {
		currentSeed = splitmix64(currentSeed)
		network.seeds[i] = currentSeed
	}

	return network, nil
}

// Network is the container information relevant for generating permutations
type Network struct {
	rounds   int
	maxValue uint64
	epochs   bool
	seeds    []uint64

	leftRadix  uint64
	rightRadix uint64
}

// Map takes an index in a sequence and maps it to another index in the same sequence
func (n *Network) Map(index uint64) (uint64, error) {
	return n.encode(index, false)
}

// InvertMap performs an inversion of Map
func (n *Network) InvertMap(index uint64) (uint64, error) {
	return n.encode(index, true)
}

func (n *Network) encode(index uint64, invert bool) (uint64, error) {
	var epochStart uint64
	var epochHash uint64
	domainSize := n.maxValue + 1

	if index > n.maxValue {
		if n.epochs {
			epochStart = index - (index % domainSize)
			epochHash = splitmix64(epochStart)
			index %= domainSize
		} else {
			return 0, fmt.Errorf("%w, index: %d, maxSize: %d", ErrIndexGreatThanMaxValue, index, n.maxValue)
		}
	}

	if n.maxValue == 0 {
		return 0, nil
	}

	a := index % n.leftRadix
	b := index / n.leftRadix

	for {
		start := 0
		adjust := 1

		if invert {
			start = n.rounds - 1
			adjust = -1
		}

		for round := start; round >= 0 && round < n.rounds; round += adjust {
			seed := n.seeds[round] ^ epochHash

			if round%2 == 0 {
				f := splitmix64(b^seed) % n.leftRadix
				if invert {
					a = (a + n.leftRadix - f) % n.leftRadix
				} else {
					a = (a + f) % n.leftRadix
				}
			} else {
				f := splitmix64(a^seed) % n.rightRadix
				if invert {
					b = (b + n.rightRadix - f) % n.rightRadix
				} else {
					b = (b + f) % n.rightRadix
				}
			}
		}

		index = a + b*n.leftRadix

		if index <= n.maxValue {
			return index + epochStart, nil
		}
	}
}

func findFactors(maxValue uint64) (uint64, uint64) {
	if maxValue < 3 {
		return 2, 2
	}

	if maxValue == ^uint64(0) {
		val := uint64(1) << 32
		return val, val
	}

	value := maxValue + 1
	sqrt := uint64Sqrt(value)

	var bestValue uint64
	var bestOther uint64
	var exceeds uint64

	// If value is 8 the following loop condition is true from the start.
	// So we have to cover this special case.
	if value == 8 {
		return sqrt, value / sqrt
	}

	for current := sqrt; (current*2) > (value/current) && current > 1; current-- {
		if value%current == 0 {
			return current, value / current
		}

		other := (value / current) + 1
		diff := (other * current) - value

		if other > current {
			diff += other - current
		} else {
			diff += current - other
		}

		if bestValue == 0 || exceeds > diff {
			exceeds = diff
			bestOther = other
			bestValue = current
		}
	}

	return bestValue, bestOther
}

func uint64Sqrt(n uint64) uint64 {
	if n < 2 {
		return n
	}
	x := uint64(math.Sqrt(float64(n)))

	for x > 0 && x > n/x {
		x--
	}
	for (x + 1) <= n/(x+1) {
		x++
	}
	return x
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

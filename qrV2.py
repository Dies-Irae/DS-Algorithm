"""
Copyright <2021> <https://github.com/Dies-Irae>

Algorithm from: "Using the Difference of Syndromes to Decode Quadratic Residue Codes" , Yong Li et.al.
Implemented By Dies-Irae in Chongqing University, 2021

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from itertools import combinations
from utils import *


class QR:
    def __init__(self, n):
        self.n = n
        if n == 23:
            self.k = 12
            self.t = 3
            self.generatorPolynomial = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1])

        elif n == 47:
            self.k = 24
            self.t = 5
            self.generatorPolynomial = np.array(
                [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])

        elif n == 71:
            self.k = 36
            self.t = 6
            self.generatorPolynomial = np.array(
                [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
                 1, 1])

        elif n == 73:
            self.k = 37
            self.t = 6
            self.generatorPolynomial = np.array(
                [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
                 0, 1, 1])

        elif n == 89:
            self.k = 45
            self.t = 8
            self.generatorPolynomial = np.array(
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
                 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1])

        elif n == 97:
            self.k = 49
            self.t = 7
            self.generatorPolynomial = np.array(
                [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1])

        self.G, self.H = generatorMatrix(self.n, self.k, self.generatorPolynomial)
        self.emTable, self.smTable = errorTable(self.n, self.H)

    def encode(self, message):
        """
        Input: origin message\n
        Return: encoded codeword
        """
        return np.remainder(np.matmul(message, self.G), 2)

    def generateBatch(self, BatchSize):
        """
        return a randomly generated message in NumPy array form, which size is(infomation bits length(k), BatchSize)
        """
        batch = np.random.randint(2, size=(BatchSize, self.k))
        return batch

    def AWGN(self, inputs, SNR):
        """
        inputs: codewords to be send\n
        SNR: Eb/N0 signal noise ratio(dB)
        """
        inputs = -(inputs * 2 - 1)  # BPSK modulation
        sigma = math.sqrt(1 / (2 * self.k / self.n * math.pow(10, SNR / 10)))
        inputs = inputs + np.random.normal(0, sigma, inputs.shape)
        return inputs

    def __DSAlgorithm(self, r):
        """
        Decode the received bit vectors using DS algorithm(from "Using the Difference of Syndromes to Decode
        Quadratic Residue Codes Fig.1")\n inputs: the received bit vectors in (,k) size
        """
        r = r.copy()
        tau = 0  # step 1)
        s = np.remainder(np.matmul(r, self.H.T), 2)  # step 2)

        while weight(s) > self.t:
            # step 4)
            for we in range(1, self.t // 2 + 1):
                invertCombinations = combinations(range(self.k), we)
                for eta in invertCombinations:
                    eta = list(eta)
                    sd = np.logical_xor(s, xorSum(eta, self.smTable))  # invert one group info bits
                    wsd = weight(sd)
                    if wsd <= self.t - we:
                        dc = r.copy()
                        dc[eta] = 1 - dc[eta]
                        dc = np.logical_xor(dc, np.concatenate((np.zeros(self.k), sd)))
                        if tau != 0:  # step 7)
                            dc = np.roll(dc, self.k)
                        return dc  # step 8)
            tau += 1
            if tau == 1:
                r = np.roll(r, -self.k)  # step 5)
            elif tau == 2:
                r[-self.k] = 1 - r[-self.k]  # r = r +(1 << (k-1))n  # step 6)
            else:
                r[-self.k] = 1 - r[-self.k]
                r = np.roll(r, self.k)
                return r
            s = np.remainder(np.matmul(r, self.H.T), 2)  # step 2)

        dc = np.logical_xor(r, np.concatenate((np.zeros(self.k), s)))  # step 3)

        if tau != 0:
            dc = np.roll(dc, self.k)  # step 7)
        return dc  # step 8)

    def DSDecode(self, inputs):
        inputs[inputs > 0] = 0  # step 2
        inputs[inputs < 0] = 1
        decoded = np.zeros((inputs.shape[0], self.n))
        for i in range(inputs.shape[0]):
            decoded[i] = self.__DSAlgorithm(inputs[i])
        return decoded


def test_thread(code_len, batch_size, SNR_start, SNR_stop, step, max_errs):
    """
    Single test thread for DS algorithm
    :param code_len: Code Length
    :param batch_size: Batch size for every loop
    :param SNR_start: the start point snr, start is included
    :param SNR_stop: the end point snr, end is included
    :param step: step size
    :param max_errs: if max errs achieved, end
    :return: BER Array(Numpy array)
    """
    qr = QR(code_len)
    SNRs = np.arange(SNR_start, SNR_stop + step, step)
    res = np.zeros(SNRs.shape[0])
    for i in SNRs:
        errs = 0
        blks = 0
        while errs < max_errs:
            sample = qr.generateBatch(batch_size)
            encodedSample = qr.encode(sample)
            received = qr.AWGN(encodedSample, i)
            decoded = qr.DSDecode(received)
            BER = np.mean(np.logical_xor(encodedSample, decoded))
            errs += BER * code_len * batch_size
            blks += batch_size
        res[i] = errs / code_len / blks
    return res


##TEST###
if __name__ == "__main__":
    import multiprocessing as mp

    num_cores = int(mp.cpu_count())
    print("Total Cores: " + str(num_cores) + " Cores")
    n_workers = 8
    code_len = 47
    batchSize = 100
    maxErrs = 125
    SNR_start = 0
    SNR_stop = 7
    step = 1
    SNRs = np.arange(SNR_start, SNR_stop + step, step)
    res = np.zeros(SNRs.shape[0])
    pool = mp.Pool(n_workers)
    results = []
    for _ in range(n_workers):
        results.append(pool.apply_async(test_thread, args=(code_len, batchSize, SNR_start, SNR_stop, step, maxErrs)))
    for worker in results:
        res = res + worker.get() / n_workers
    for i in range(SNRs.shape[0]):
        print("BER: %e @ %f dB" % (res[i], SNRs[i]))

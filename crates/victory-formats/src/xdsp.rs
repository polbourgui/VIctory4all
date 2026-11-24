//! The audio DSP's block transforms and convolution reverb.
//!
//! A convolution stream takes its impulse response already in the frequency
//! domain, cut into partitions: for blocks of `B` frames, each partition is
//! `B` complex values (`2B` floats), and a response of `L` floats has
//! `L / 2B` of them. That is uniformly partitioned overlap-save convolution
//! with real FFTs of `2B` samples, which is what we compute.
//!
//! A spectrum of `n` real samples is kept in `n` floats, packed: the DC and
//! Nyquist bins are real, so floats 0 and 1 hold them, and floats `2k`,
//! `2k + 1` hold the real and imaginary parts of bin `k` for `0 < k < n/2`.
//! The forward transform is unscaled and the inverse scales by `1/n`, so
//! a response transformed by [`forward`] convolves as its time-domain
//! samples would.
// shortcut: the hardware's own packing, bin order and scaling are inferred,
// not known. A title that transforms its responses with the DSP's own
// forward transform gets matching layouts either way; one that ships
// responses already transformed needs these to match the hardware.

use std::f32::consts::TAU;

/// One radix-2 FFT size: its twiddles and bit-reversal permutation.
#[derive(Clone)]
struct Plan {
    n: usize,
    /// `e^(-2πik/n)` for `k < n/2`.
    twiddles: Vec<(f32, f32)>,
    reversed: Vec<u32>,
}

impl Plan {
    fn new(n: usize) -> Plan {
        assert!(n.is_power_of_two() && n >= 2, "FFT size {n}");
        let bits = n.trailing_zeros();
        Plan {
            n,
            twiddles: (0..n / 2)
                .map(|k| {
                    let angle = -TAU * k as f32 / n as f32;
                    (angle.cos(), angle.sin())
                })
                .collect(),
            reversed: (0..n as u32)
                .map(|i| i.reverse_bits() >> (32 - bits))
                .collect(),
        }
    }

    /// In-place complex FFT of `n` values; `inverse` conjugates the
    /// twiddles and leaves the result unscaled.
    fn complex(&self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        let n = self.n;
        for i in 0..n {
            let j = self.reversed[i] as usize;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let sign = if inverse { -1.0 } else { 1.0 };
        let mut size = 2;
        while size <= n {
            let half = size / 2;
            let step = n / size;
            for start in (0..n).step_by(size) {
                for k in 0..half {
                    let (wr, wi) = self.twiddles[k * step];
                    let wi = wi * sign;
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            size *= 2;
        }
    }
}

/// A reusable real transform of one size.
#[derive(Clone)]
pub struct Transform {
    plan: Plan,
    re: Vec<f32>,
    im: Vec<f32>,
}

impl Transform {
    pub fn new(n: usize) -> Transform {
        Transform {
            plan: Plan::new(n),
            re: vec![0.0; n],
            im: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.plan.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// `n` real samples to their packed spectrum.
    pub fn forward(&mut self, input: &[f32], spectrum: &mut [f32]) {
        let n = self.plan.n;
        self.re.copy_from_slice(&input[..n]);
        self.im.fill(0.0);
        self.plan.complex(&mut self.re, &mut self.im, false);
        spectrum[0] = self.re[0];
        spectrum[1] = self.re[n / 2];
        for k in 1..n / 2 {
            spectrum[2 * k] = self.re[k];
            spectrum[2 * k + 1] = self.im[k];
        }
    }

    /// A packed spectrum back to `n` real samples, scaled by `1/n`.
    pub fn inverse(&mut self, spectrum: &[f32], output: &mut [f32]) {
        let n = self.plan.n;
        self.re[0] = spectrum[0];
        self.im[0] = 0.0;
        self.re[n / 2] = spectrum[1];
        self.im[n / 2] = 0.0;
        for k in 1..n / 2 {
            let (r, i) = (spectrum[2 * k], spectrum[2 * k + 1]);
            (self.re[k], self.im[k]) = (r, i);
            (self.re[n - k], self.im[n - k]) = (r, -i);
        }
        self.plan.complex(&mut self.re, &mut self.im, true);
        let scale = 1.0 / n as f32;
        for (o, r) in output[..n].iter_mut().zip(&self.re) {
            *o = r * scale;
        }
    }
}

/// `n` real samples to their packed spectrum.
pub fn forward(input: &[f32], spectrum: &mut [f32]) {
    Transform::new(input.len()).forward(input, spectrum);
}

/// A packed spectrum back to its real samples.
pub fn inverse(spectrum: &[f32], output: &mut [f32]) {
    Transform::new(spectrum.len()).inverse(spectrum, output);
}

/// `acc += a * b` over packed spectra.
fn multiply_add(acc: &mut [f32], a: &[f32], b: &[f32]) {
    acc[0] += a[0] * b[0];
    acc[1] += a[1] * b[1];
    let pairs = acc.len() / 2;
    for k in 1..pairs {
        let (ar, ai) = (a[2 * k], a[2 * k + 1]);
        let (br, bi) = (b[2 * k], b[2 * k + 1]);
        acc[2 * k] += ar * br - ai * bi;
        acc[2 * k + 1] += ar * bi + ai * br;
    }
}

/// Cuts a time-domain impulse response into the partitions a convolution
/// stream with blocks of `block` frames takes.
pub fn partition(response: &[f32], block: usize) -> Vec<f32> {
    let mut transform = Transform::new(2 * block);
    let mut padded = vec![0.0; 2 * block];
    let mut out = Vec::new();
    for chunk in response.chunks(block) {
        padded.fill(0.0);
        padded[..chunk.len()].copy_from_slice(chunk);
        let at = out.len();
        out.resize(at + 2 * block, 0.0);
        transform.forward(&padded, &mut out[at..]);
    }
    out
}

/// One channel of a convolution stream.
#[derive(Clone)]
struct Lane {
    /// The previous input block and the current one.
    window: Vec<f32>,
    /// The spectra of the last `partitions` windows, newest at `newest`.
    history: Vec<f32>,
    newest: usize,
}

/// Convolves blocks of `block` frames with a partitioned impulse response,
/// one block in, one block out, per channel.
pub struct Convolver {
    block: usize,
    partitions: usize,
    /// The response's partitions, per response channel.
    responses: Vec<Vec<f32>>,
    lanes: Vec<Lane>,
    transform: Transform,
    accumulator: Vec<f32>,
    time: Vec<f32>,
}

impl Convolver {
    /// `response` holds `response_channels` responses one after another,
    /// each of `floats` floats in the packed partitioned layout. Lane `c`
    /// uses response `c` if there is one, else the first.
    pub fn new(
        block: usize,
        channels: usize,
        response: &[f32],
        floats: usize,
        response_channels: usize,
    ) -> Option<Convolver> {
        let spectrum = 2 * block;
        if !block.is_power_of_two()
            || channels == 0
            || response_channels == 0
            || floats == 0
            || !floats.is_multiple_of(spectrum)
            || response.len() < floats * response_channels
        {
            return None;
        }
        let partitions = floats / spectrum;
        Some(Convolver {
            block,
            partitions,
            responses: response
                .chunks(floats)
                .take(response_channels)
                .map(<[f32]>::to_vec)
                .collect(),
            lanes: vec![
                Lane {
                    window: vec![0.0; spectrum],
                    history: vec![0.0; partitions * spectrum],
                    newest: 0,
                };
                channels
            ],
            transform: Transform::new(spectrum),
            accumulator: vec![0.0; spectrum],
            time: vec![0.0; spectrum],
        })
    }

    pub fn block(&self) -> usize {
        self.block
    }

    pub fn channels(&self) -> usize {
        self.lanes.len()
    }

    /// Takes the next `block` input frames of `channel` and writes the
    /// same number of output frames.
    pub fn process(&mut self, channel: usize, input: &[f32], output: &mut [f32]) {
        let (b, spectrum) = (self.block, 2 * self.block);
        let lane = &mut self.lanes[channel];
        lane.window.copy_within(b.., 0);
        lane.window[b..].copy_from_slice(&input[..b]);
        lane.newest = (lane.newest + 1) % self.partitions;
        let at = lane.newest * spectrum;
        self.transform
            .forward(&lane.window, &mut lane.history[at..at + spectrum]);

        let response = &self.responses[channel.min(self.responses.len() - 1)];
        self.accumulator.fill(0.0);
        for k in 0..self.partitions {
            let slot = (lane.newest + self.partitions - k) % self.partitions * spectrum;
            multiply_add(
                &mut self.accumulator,
                &lane.history[slot..slot + spectrum],
                &response[k * spectrum..(k + 1) * spectrum],
            );
        }
        self.transform.inverse(&self.accumulator, &mut self.time);
        // Overlap-save: the second half is the linear convolution.
        output[..b].copy_from_slice(&self.time[b..]);
    }
}

/// Packed cartesian bins to magnitude and phase, in place. The DC and
/// Nyquist floats are real and stay as they are.
// shortcut: how the hardware writes those two in polar form is not known.
pub fn to_polar(spectrum: &mut [f32]) {
    for bin in spectrum.as_chunks_mut::<2>().0.iter_mut().skip(1) {
        let [re, im] = *bin;
        *bin = [re.hypot(im), im.atan2(re)];
    }
}

/// Magnitude and phase back to packed cartesian bins, in place.
pub fn from_polar(spectrum: &mut [f32]) {
    for bin in spectrum.as_chunks_mut::<2>().0.iter_mut().skip(1) {
        let [magnitude, phase] = *bin;
        let (sin, cos) = phase.sin_cos();
        *bin = [magnitude * cos, magnitude * sin];
    }
}

/// Scales a block by a gain ramping linearly from `begin` (first frame)
/// towards `end`, as convolution commands ask.
pub fn ramp(block: &mut [f32], frames: usize, channels: usize, begin: f32, end: f32) {
    if begin == 1.0 && end == 1.0 {
        return;
    }
    let step = (end - begin) / frames as f32;
    for (i, frame) in block.chunks_mut(channels).take(frames).enumerate() {
        let gain = begin + i as f32 * step;
        frame.iter_mut().for_each(|s| *s *= gain);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic noise source.
    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as f32 / u32::MAX as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn direct(signal: &[f32], response: &[f32]) -> Vec<f32> {
        (0..signal.len())
            .map(|i| {
                (0..response.len().min(i + 1))
                    .map(|k| signal[i - k] * response[k])
                    .sum()
            })
            .collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tolerance, "at {i}: {x} vs {y}");
        }
    }

    #[test]
    fn forward_packs_dc_nyquist_and_bins() {
        let n = 16;
        // DC 1, a cosine at bin 2 of amplitude 1, Nyquist 0.5.
        let input: Vec<f32> = (0..n)
            .map(|i| {
                1.0 + (TAU * 2.0 * i as f32 / n as f32).cos()
                    + 0.5 * if i % 2 == 0 { 1.0 } else { -1.0 }
            })
            .collect();
        let mut spectrum = vec![0.0; n];
        forward(&input, &mut spectrum);
        let mut expected = vec![0.0; n];
        expected[0] = n as f32;
        expected[1] = 0.5 * n as f32;
        expected[4] = n as f32 / 2.0;
        assert_close(&spectrum, &expected, 1e-4);
    }

    #[test]
    fn inverse_undoes_forward() {
        for n in [2, 8, 512, 1024, 2048] {
            let input = noise(n, n as u32);
            let mut spectrum = vec![0.0; n];
            let mut back = vec![0.0; n];
            let mut transform = Transform::new(n);
            transform.forward(&input, &mut spectrum);
            transform.inverse(&spectrum, &mut back);
            assert_close(&back, &input, 1e-5);
        }
    }

    #[test]
    fn convolves_like_the_direct_sum() {
        for (block, taps) in [(512, 1500), (1024, 1024), (512, 1)] {
            let response = noise(taps, 7);
            let signal = noise(block * 5, 3);
            let partitions = partition(&response, block);
            let mut convolver = Convolver::new(block, 1, &partitions, partitions.len(), 1).unwrap();
            let mut out = vec![0.0; signal.len()];
            for (i, o) in signal.chunks(block).zip(out.chunks_mut(block)) {
                convolver.process(0, i, o);
            }
            assert_close(&out, &direct(&signal, &response), 1e-3);
        }
    }

    #[test]
    fn stereo_lanes_take_their_own_response() {
        let block = 512;
        let (left, right) = (noise(700, 1), noise(700, 2));
        let mut both = partition(&left, block);
        let floats = both.len();
        both.extend(partition(&right, block));
        let signal = noise(block * 3, 9);
        let mut convolver = Convolver::new(block, 2, &both, floats, 2).unwrap();
        let (mut l, mut r) = (vec![0.0; signal.len()], vec![0.0; signal.len()]);
        for ((i, lo), ro) in signal
            .chunks(block)
            .zip(l.chunks_mut(block))
            .zip(r.chunks_mut(block))
        {
            convolver.process(0, i, lo);
            convolver.process(1, i, ro);
        }
        assert_close(&l, &direct(&signal, &left), 1e-3);
        assert_close(&r, &direct(&signal, &right), 1e-3);

        // A mono response serves both lanes.
        let mono = partition(&left, block);
        let mut convolver = Convolver::new(block, 2, &mono, mono.len(), 1).unwrap();
        let mut out = vec![0.0; block];
        convolver.process(1, &signal[..block], &mut out);
        assert_close(&out, &direct(&signal[..block], &left), 1e-3);
    }

    #[test]
    fn refuses_responses_that_are_not_whole_partitions() {
        assert!(Convolver::new(512, 1, &[0.0; 1000], 1000, 1).is_none());
        assert!(Convolver::new(512, 1, &[0.0; 1024], 1024, 2).is_none());
        assert!(Convolver::new(512, 0, &[0.0; 1024], 1024, 1).is_none());
    }

    #[test]
    fn polar_round_trips() {
        let mut spectrum = vec![3.0, -2.0, 0.0, 1.0, -1.0, 0.0, 3.0, 4.0];
        let cartesian = spectrum.clone();
        to_polar(&mut spectrum);
        assert_eq!(spectrum[..2], [3.0, -2.0]);
        assert_close(&spectrum[2..4], &[1.0, std::f32::consts::FRAC_PI_2], 1e-6);
        assert_close(&spectrum[6..], &[5.0, 4f32.atan2(3.0)], 1e-6);
        from_polar(&mut spectrum);
        assert_close(&spectrum, &cartesian, 1e-6);
    }

    #[test]
    fn ramp_scales_from_begin_towards_end() {
        let mut block = vec![1.0; 8];
        ramp(&mut block, 4, 2, 1.0, 0.0);
        assert_eq!(block, [1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25]);
        let mut unity = vec![2.0; 4];
        ramp(&mut unity, 4, 1, 1.0, 1.0);
        assert_eq!(unity, [2.0; 4]);
    }
}

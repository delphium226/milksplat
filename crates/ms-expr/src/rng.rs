//! xoshiro256++ random number generator (no dependencies).

#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn seeded(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1) from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_per_seed() {
        let mut a = Rng::seeded(42);
        let mut b = Rng::seeded(42);
        let mut c = Rng::seeded(43);
        let xs: Vec<f64> = (0..8).map(|_| a.next_f64()).collect();
        let ys: Vec<f64> = (0..8).map(|_| b.next_f64()).collect();
        let zs: Vec<f64> = (0..8).map(|_| c.next_f64()).collect();
        assert_eq!(xs, ys);
        assert_ne!(xs, zs);
    }

    /// Reference values: xoshiro256++ (Blackman/Vigna reference C algorithm)
    /// with state = four successive splitmix64 outputs from seed 0, i.e.
    /// s = [0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4, 0x06c45d188009454f,
    /// 0xf88bb8a8724c81ec]. Outputs computed independently with a Python
    /// transcription of the reference algorithm.
    #[test]
    fn known_answer_seed_zero() {
        let mut r = Rng::seeded(0);
        assert_eq!(r.next_u64(), 0x53175d61490b23df);
        assert_eq!(r.next_u64(), 0x61da6f3dc380d507);
        assert_eq!(r.next_u64(), 0x5c0fdf91ec9a7bfc);
    }

    #[test]
    fn in_unit_interval_and_spread() {
        let mut r = Rng::seeded(0);
        let mut sum = 0.0;
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
            sum += x;
        }
        let mean = sum / 10_000.0;
        assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
    }
}

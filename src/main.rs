use fixedbitset::FixedBitSet;
use rayon::prelude::*;
// use std::error;
// use std::fmt;
// use std::io::prelude::*;
// use std::io::BufReader;
// use std::sync::Arc;
use std::iter::Iterator;

// #[derive(PartialEq, fmt::Debug)]
// enum PrimeError {
//     NotEnough,
// }

// impl fmt::Display for PrimeError {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.write_str(&(self as &dyn error::Error).to_string())
//     }
// }

// impl error::Error for PrimeError {
//     fn description(&self) -> &str {
//         match *self {
//             PrimeError::NotEnough => "Not Enough",
//         }
//     }
// }

// pub fn get_primes(limit: i64) -> Result<Vec<u64>, Box<dyn error::Error>> {
//     let flimit: f32 = limit as f32;
//     let estimated_capacity: usize = (flimit / flimit.ln()) as usize;
//     let mut primes: Vec<i64> = Vec::with_capacity(estimated_capacity);
//     let primes_file = File::open(PRIMES_FILE)?;
//     let primes_file = BufReader::new(primes_file);
//     // let mut buffer = [0; 200];
//     let mut done = false;
//     for line in primes_file.lines() {
//         let prime = line.unwrap().parse::<i64>().unwrap();
//         if prime > limit {
//             done = true;
//             break;
//         }
//         primes.push(prime);
//     }
//     if !done {
//         return Err(PrimeError::NotEnough)?;
//     }
//     Ok(primes)
// }

fn format_bitset(b: &FixedBitSet) -> String {
    let mut s = String::from("");
    for i in 0..(b.len()) {
        s = format!("{}{}", s, b[i] as i8).to_owned();
    }
    s.into()
}

fn generate_first_prime_bitset(capacity: usize) -> FixedBitSet {
    // due to fixedbitset being initialized to 0s, one shall mark when a number is not prime.
    let mut candidates: FixedBitSet = FixedBitSet::with_capacity(capacity);
    // println!("{}", capacity);
    // println!("{:?}", candidates);
    static INITIAL_PRIMES: [i32; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    for p in INITIAL_PRIMES.iter() {
        if *p as usize > capacity {
            // println!("{:?}", format_bitset(&candidates));
            // println!("\n\n\n");
            return candidates;
        }
        for i in *p..(1 + (capacity as i32) / p) {
            let m = i * p;
            if m >= (capacity as i32) {
                break;
            }
            candidates.insert(m as usize);
        }
    }

    // println!("{:?}", candidates);
    for i in INITIAL_PRIMES[9]..(capacity as i32) {
        // if not composite == prime
        if !candidates[i as usize] {
            for k in i..(1 + (capacity as i32) / i) {
                let m = k * i;
                if m >= (capacity as i32) {
                    break;
                }
                candidates.insert(m as usize);
            }
        }
    }
    // println!("\n\n\n");
    candidates
}

fn generate_bitset_from(
    capacity: usize,
    lower: i64,
    upper: i64,
    primes: &FixedBitSet,
) -> FixedBitSet {
    // println!(
    //     "generate_bitset_from called with lower {}, upper {}",
    //     lower, upper
    // );
    let mut candidates: FixedBitSet = FixedBitSet::with_capacity(capacity);
    for i in 2..(capacity as i64) {
        // if not composite == prime
        if !primes[i as usize] {
            // i is prime
            // println!("{}", i);
            for k in (lower / i - 1)..(2 + upper / i) {
                let m = k * i;
                if m < lower {
                    continue;
                }
                if m >= upper {
                    break;
                }
                candidates.insert((m - lower) as usize);
            }
        }
    }
    // println!("{}", format_bitset(&candidates));
    // println!("\n\n\n exiting generate_bitset_from");
    candidates
}

pub struct PrimeChunkIterator {
    prime_chunks: Vec<FixedBitSet>,
    chunk_size: usize, // the maximal value of index
    limit: usize,
    prime: usize,
}

impl PrimeChunkIterator {
    pub fn new(prime_chunks: Vec<FixedBitSet>, chunk_size: usize, limit: usize) -> Self {
        PrimeChunkIterator {
            prime_chunks,
            chunk_size,
            limit,
            prime: 1,
        }
    }
}

impl Iterator for PrimeChunkIterator {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        // assert bit is set at the index and chunk index that coorespond to `prime`
        let mut chunk_index = self.prime / self.chunk_size;
        let mut index = self.prime % self.chunk_size;
        // assert!(
        //     !self.prime_chunks[chunk_index][index],
        //     "at {} * {} + {} = {} = {} : {}",
        //     chunk_index,
        //     self.chunk_size,
        //     index,
        //     chunk_index * self.chunk_size + index,
        //     self.prime,
        //     format_bitset(&self.prime_chunks[chunk_index])
        // );
        loop {
            index += 1;
            if index >= self.chunk_size {
                chunk_index += 1;
                index = 0;
            }
            if chunk_index >= self.prime_chunks.len() {
                return None;
            }
            if chunk_index * self.chunk_size + index > self.limit {
                return None;
            }
            if !self.prime_chunks[chunk_index][index] {
                self.prime = chunk_index * self.chunk_size + index;
                println!("found prime {}", self.prime);
                return Some(self.prime as u64);
            }
        }
    }
}

fn chunk_to_vec(chunk: &FixedBitSet, chunk_size: usize, offset: usize) -> Vec<u64> {
    // println!("{} {}: {}", offset, format_bitset(chunk), chunk_size);
    let mut result: Vec<u64> = Vec::new();
    let start = if offset == 0 { 2 } else { 0 };
    for i in start..chunk_size {
        if !chunk[i] {
            result.push((offset + i) as u64);
        }
    }
    result
}

// this method does not guarantee that all primes in the resulting vec are below `limit`.
// please manually check or filter yourself after converting.
impl Into<Vec<u64>> for PrimeChunkIterator {
    fn into(self) -> Vec<u64> {
        let res = (0..self.prime_chunks.len())
            .into_par_iter()
            .map(|i| chunk_to_vec(&self.prime_chunks[i], self.chunk_size, i * self.chunk_size))
            .fold(
                || Vec::<u64>::new(),
                |mut a, b| {
                    a.extend(b);
                    a
                },
            )
            .reduce(
                || Vec::<u64>::new(),
                |mut a, b| {
                    a.extend(b);
                    a
                },
            );
        res
    }
}

fn bit_prime_sieve(limit: i64, chunk_size_bias: f32) -> PrimeChunkIterator {
    let sqrt = (limit as f32).powf(0.5);
    let chunk_size = (sqrt * chunk_size_bias) as usize;
    let num_chunks = (limit as usize) / chunk_size;
    let proto_chunk = generate_first_prime_bitset(chunk_size);
    let mut v = Vec::new();
    v.push(proto_chunk.clone());
    v.par_extend(
        (1..(1 + num_chunks))
            .into_par_iter()
            .map(|i: usize| {
                generate_bitset_from(
                    chunk_size,
                    (i as i64) * (chunk_size as i64),
                    (1 + i as i64) * (chunk_size as i64),
                    &proto_chunk,
                )
            })
            .collect::<Vec<FixedBitSet>>(),
    );

    PrimeChunkIterator::new(v, chunk_size, limit as usize)
}

fn main() -> std::io::Result<()> {
    use std::fs::File;
    use std::io::prelude::*;
    use std::io::{BufWriter, Seek, SeekFrom};
    let threads = 20;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    use std::time::Instant;
    let now = Instant::now();
    let limit = 100_000_000;
    let vec: Vec<u64> = bit_prime_sieve(limit, 32.0).into();
    println!(
        "calculated {} primes in {}ms",
        vec.len(),
        now.elapsed().as_millis()
    );
    let now = Instant::now();
    // let limit = 1000;
    let mut primes_file = BufWriter::new(File::create(format!("primes_to_{}.txt", limit))?);
    for p in vec {
        if p > (limit as u64) {
            break;
        }
        primes_file.write_fmt(format_args!("{}\n", p))?;
    }
    primes_file.flush()?;
    let bytes_written = primes_file.seek(SeekFrom::Current(0))?;
    println!(
        "{} bytes written in {}ms",
        bytes_written,
        now.elapsed().as_millis()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_high_performance() {
        let threads = 20;
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
        use std::time::Instant;
        let now = Instant::now();
        let limit = 1_000_000_000;
        let vec: Vec<u64> = bit_prime_sieve(limit, 1.0).into();
        // let limit = 1000;
        let mut s = 0;
        for p in vec {
            if p > (limit as u64) {
                break;
            }
            s += p;
            if limit <= 1000 {
                println!("{}", p);
            }
        }
        println!("{}", s);
        println!("{}", now.elapsed().as_millis());
    }
    #[test]
    fn test_below_1million() {
        let threads = 20;
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
        use std::time::Instant;
        let now = Instant::now();
        let limit = 1_000_000;
        let vec: Vec<u64> = bit_prime_sieve(limit, 1.0).into();
        // let limit = 1000;
        let mut s = 0;
        for p in vec {
            if p > limit as u64 {
                break;
            }
            s += p;
            if limit <= 1000 {
                println!("{}", p);
            }
        }
        println!("{}", s);
        println!("{}", now.elapsed().as_millis());
    }
    #[test]
    fn test_into_100() {
        let limit = 100;
        let vec: Vec<u64> = bit_prime_sieve(limit, 3.0).into();
        // let limit = 1000;
        let mut s = 0;
        for p in vec {
            if p > limit as u64 {
                break;
            }
            s += p;
            if limit <= 1000 {
                println!("{}", p);
            }
        }
        assert!(s == 1060);
    }
    #[test]
    fn test_into_1000() {
        let limit = 1000;
        let vec: Vec<u64> = bit_prime_sieve(limit, 1.0).into();
        // let limit = 1000;
        let mut s = 0;
        for p in vec {
            if p >= limit as u64 {
                break;
            }
            s += p;

            if limit <= 1000 {
                println!("{}", p);
            }
        }
        assert!(s == 76096, "{}", s);
    }
    #[test]
    fn test_into_10000() {
        let limit = 10000;
        let vec: Vec<u64> = bit_prime_sieve(limit, 1.0).into();
        // let limit = 1000;
        let mut s = 0;
        for p in vec {
            if p > limit as u64 {
                break;
            }
            s += p;

            if limit <= 1000 {
                println!("{}", p);
            }
        }
        println!("{}", s);
        assert!(s == 5736396);
    }

    #[test]
    fn test_100() {
        let limit = 100;
        let mut s = 0;
        for p in bit_prime_sieve(limit, 2.0) {
            if p > limit as u64 {
                break;
            }
            s += p;
            if limit <= 1000 {
                println!("{}", p);
            }
        }
        assert!(s == 1060);
    }
    #[test]
    fn test_1000() {
        let limit = 1000;
        let mut s = 0;
        for p in bit_prime_sieve(limit, 1.0) {
            if p > limit as u64 {
                break;
            }
            s += p;

            if limit <= 1000 {
                println!("{}", p);
            }
        }
        assert!(s == 76096);
    }
    #[test]
    fn test_10000() {
        let limit = 10000;
        let mut s = 0;
        for p in bit_prime_sieve(limit, 1.0) {
            if p > limit as u64 {
                break;
            }
            s += p;

            if limit <= 1000 {
                println!("{}", p);
            }
        }
        assert!(s == 5736396);
    }
}

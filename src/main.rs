extern crate spreadsheet;
extern crate rand;

use rand::Rng;
use rand::distributions::{Range, IndependentSample};
use spreadsheet::{Cell, Spreadsheet};

use std::sync::{Arc, Mutex};
use std::thread::*;
use std::io;
use std::io::Write;

fn distance(x: &[f32], b: &[f32]) -> f32 {
    let mut sums: f32 = 0.0;
    for (i, j) in x.iter().zip(b.iter()) {
        sums += (i - j).powi(2);
    }
    sums
}

#[derive(Debug)]
struct Cluster {
    members: Vec<usize>,
    center: Vec<f32>,
}

/// Thread safe k-means clustering algorithm
fn kmeans(mutex: Arc<Mutex<i32>>, matrix: Arc<Vec<Vec<f32>>>, kn: usize) -> Vec<Cluster> {

    assert!(kn < matrix.len());

    let mut clusters: Vec<Cluster> = Vec::new();

    let mut membership: Vec<usize> = Vec::new();
    let mut cluster_scores: Vec<Vec<f32>> = Vec::new();
    let mut centroids: Vec<&Vec<f32>> = Vec::new();

    let range = Range::new(0, matrix.len());
    let mut rng = rand::thread_rng();

    for k in 0..kn {
        let x = matrix.get(range.ind_sample(&mut rng)).unwrap();
        centroids.push(x);
        cluster_scores.push(Vec::new());
        clusters.push(Cluster { members: Vec::new(), center: x.clone()});
    }

    let mut iter = matrix.iter().enumerate();

    // iterate through each row in the matrix, and pick the best cluster membership
    while let Some((i, current)) = iter.next() {
        let mut scores: Vec<f32> = Vec::new();
        let mut cn =  centroids.iter().enumerate();
        while let Some((k, c)) = cn.next() {
            let s = distance(&current[..], c);
            //println!("{}, {}", k, &s);
            scores.push(s);
        }            

        let min = scores.iter().cloned().fold(1./0., f32::min);
        let winning_k = scores.iter().position(|&x| x == min).unwrap();
        //println!("{} best cluster is {} with a score of {}", i, winning_k, min);

        membership.push(winning_k);
        if let Some(c) = clusters.get_mut(winning_k) {
            c.members.push(i);
        }
        if let Some(wcss) = cluster_scores.get_mut(i) {
            wcss.push(min);
        }
    }   

    // mutex prevents threads from writing to stdout out-of-order
    let i = mutex.lock().unwrap();
    println!("Scores w/ k={}", kn);
    for (k, c) in centroids.iter().zip(cluster_scores.iter()) {
        println!("{:?}: score {}", k, c.iter().cloned().sum::<f32>() / c.len() as f32);
    }

    clusters
   
}

fn main() {
    //let a = args().collect::<Vec<String>>();
    let filename = "iris.txt"; //&a[1];
    let s = Spreadsheet::read(filename, '\t').unwrap();

    let mut enumerate = s.data.iter().enumerate();

    let mut matrix: Vec<Vec<f32>> = Vec::new();
    let mut classes: Vec<String> = Vec::new();

    while let Some((i, row)) = enumerate.next() {
        // iterate across each row, collecting all numeric data types
        let floats = row.iter().filter(|&cell| {
            match *cell {
                 Cell::Integer(_) | Cell::Float(_) => true,
                 _ => false,
            }
        }).map(|cell| {
            match *cell {
                Cell::Integer(i) => i as f32,
                Cell::Float(i) => i,
                _ => 0.0,
            }
        }).collect::<Vec<f32>>();

        // iterate across each row and collect the class
        let mut class = row.iter().filter(|&cell| {
            match *cell {
                 Cell::String(_) => true,
                 _ => false,
            }
        }).map(|cell| {
            match *cell {
                Cell::String(ref s) => s.clone(),
                _ => "".into()
            }
        }).collect::<Vec<String>>();

        if class.len() != 1 {
            panic!("Too many string classes specified");
        }

        // draining iterator allows us to take ownership
        classes.push(class.drain(0..).next().unwrap());
        matrix.push(floats);
        
    }

    matrix.shrink_to_fit();
    classes.shrink_to_fit();

    let mut handles: Vec<JoinHandle<_>> = Vec::new();

    // thread safe data structure for sharing the matrix
    let m = Arc::new(matrix);
    let c = Arc::new(classes);
    // stdout lock
    let l = Arc::new(Mutex::new(0));

    // spawn a thread for each K-value
    for K in 1..5 {
        let mat = m.clone();
        let lock = l.clone();
        handles.push(std::thread::spawn(move || {
            let c = kmeans(lock, mat, K);
            println!("{:?}", c);
        }));
    }

    for thread in handles {
        let _ = thread.join();
    }
}

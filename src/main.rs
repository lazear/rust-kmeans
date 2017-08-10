extern crate spreadsheet;
extern crate rand;

use rand::distributions::{Range, IndependentSample};
use spreadsheet::{Cell, Spreadsheet};

use std::sync::{Arc, Mutex};
use std::thread::*;

fn distance(x: &[f32], b: &[f32]) -> f32 {
    let mut sums: f32 = 0.0;
    for (i, j) in x.iter().zip(b.iter()) {
        sums += (i - j).powi(2);
    }
    sums
}

#[derive(Debug)]
struct Cluster {
    // store index of all members
    members: Vec<usize>,
    // store index of center
    center: usize,
}

/// Thread safe k-means clustering algorithm
fn kmeans(mutex: Arc<Mutex<i32>>, matrix: Arc<Vec<Vec<f32>>>, k: usize) -> Vec<Cluster> {

    // Make sure that our K value does not exceed number of data points
    assert!(k < matrix.len());

    let mut clusters: Vec<Cluster> = Vec::with_capacity(k);

    let mut rng = rand::thread_rng();
    let range = Range::new(0, matrix.len());  

    // Start by picking initial unique seeds
    for _ in 0..k {
        // Pick a random data point as a centroid
        let x = range.ind_sample(&mut rng);
        clusters.push(Cluster { members: Vec::new(), center: x});
    }

    let mut iter = matrix.iter().enumerate();

    // iterate through each row in the matrix, and pick the best cluster membership
    while let Some((i, current)) = iter.next() {
        let mut scores: Vec<f32> = Vec::new();

        // iterate through the cluster list, and find the euclidean distance
        // between the current data point and each cluster's center
        for c in clusters.iter() {
            scores.push(distance(&current[..], &matrix[c.center]));
        }

        // pick the cluster-center that we are closest to
        let min = scores.iter().cloned().fold(1./0., f32::min);
        let best_cluster = scores.iter().position(|&x| x == min).unwrap();

        // update the cluster list
        if let Some(c) = clusters.get_mut(best_cluster) {
            c.members.push(i);
        }
    }   

    for _ in 0..10000 {
        // Pick the best center for each cluster
        //println!("round {}", r);
        for c in clusters.iter_mut() {
            let mut scores: Vec<f32> = Vec::with_capacity(c.members.len());
            // Simulate each member as the cluster center
            for new_center in c.members.iter() {
                let mut sum = 0f32;
                for member in c.members.iter() {
                    sum += distance(&matrix[*member], &matrix[*new_center]);
                }
                scores.push(sum);
            }

            let min = scores.iter().cloned().fold(1./0., f32::min);

            // Update the cluster's center
            if let Some(best) = scores.iter().position(|&x| x == min) {
                c.center = best;
            }
            c.members.clear();
        }   


        let mut iter = matrix.iter().enumerate();

        // iterate through each row in the matrix, and pick the best cluster membership
        while let Some((i, current)) = iter.next() {
            let mut scores: Vec<f32> = Vec::new();

            // iterate through the cluster list, and find the euclidean distance
            // between the current data point and each cluster's center
            for c in clusters.iter() {
                scores.push(distance(&current[..], &matrix[c.center]));
            }

            // pick the cluster-center that we are closest to
            let min = scores.iter().cloned().fold(1./0., f32::min);
            let best_cluster = scores.iter().position(|&x| x == min).unwrap();

            // update the cluster list
            if let Some(c) = clusters.get_mut(best_cluster) {
                c.members.push(i);
            }
        } 

        // Now we clear the membership list, and then pick new members for each cluster
    }

    // We now have the initial cluster centers and members picked
    // Next, we need to try to optimize centers to pick the best values



    // mutex prevents threads from writing to stdout out-of-order
    let i = mutex.lock().unwrap();
    println!("Scores w/ k={}", k);
    for (q, c) in clusters.iter().enumerate() {
        let sum: f32 = c.members.iter()
                               .fold(0f32, |acc, &x| acc + distance(&matrix[x], &matrix[c.center]));
        println!("{:?}: score {}", q, sum / c.members.iter().sum::<usize>() as f32);
    }

   // println!("{:?}", membership);

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

        // if class.len() != 1 {
        //     panic!("Too many string classes specified");
        // }

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

   // kmeans(l, m, 3);

    //spawn a thread for each K-value
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

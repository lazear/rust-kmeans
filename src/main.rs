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

fn matrix_to_cells(m: &Vec<Vec<f32>>, s: &Vec<String>) -> Vec<Vec<Cell>> {
    m.iter().zip(s.iter())
            .map(|(i, j)| {
                let mut l = i.iter().map(|&x| Cell::Float(x)).collect::<Vec<Cell>>();
                l.push(Cell::String(j.clone())); 
                l 
            }).collect()
}

#[derive(Debug)]
struct Cluster {
    // store index of all members
    members: Vec<usize>,
    // store index of center
    center: usize,
}

/// Thread safe k-means clustering algorithm
fn kmeans(mutex: Arc<Mutex<i32>>, matrix: Arc<Vec<Vec<f32>>>, k: usize) -> (f32, Vec<Vec<f32>>) {

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

    // We now have the initial cluster centers and members picked
    // Next, we need to try to optimize centers to pick the best values

    for _ in 0..matrix.len().pow(2) {
        // Pick the best center for each cluster
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

            // Now we clear the membership list, and then pick new members for each cluster
            c.members.clear();
        }   

        // iterate through each row in the matrix, and pick the best cluster membership
        let mut iter = matrix.iter().enumerate();
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
    }

    // mutex prevents threads from writing to stdout out-of-order
    // if we do not declare a variable, the lock will be dropped too early
    let i = mutex.lock().unwrap();
    let mut membership: Vec<usize> = (0..matrix.len()).collect();

    println!("Scores w/ k={}", k);
    let mut score = 0f32;
    for (q, c) in clusters.iter().enumerate() {
        let sum: f32 = c.members.iter()
                 .inspect(|&m| membership[*m] = q)
                 .fold(0f32, |acc, &x| acc + distance(&matrix[x], &matrix[c.center]));
        score += sum;
        println!("{:?}: score {}", q, sum / c.members.iter().sum::<usize>() as f32);
    }
    println!("Final score: {}", score);

    // generate a new matrix containing the cluster # appended to the end
    let mut results: Vec<Vec<f32>> = Vec::with_capacity(matrix.len());
    for (row, cluster_num) in matrix.iter().zip(membership.iter()) {
        let mut data: Vec<f32> = row.clone();
        data.push(*cluster_num as f32);
        results.push(data);
    }
    (score, results)
}

fn main() {
    let filename = "iris.txt";
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

        // draining iterator allows us to take ownership
        classes.push(class.drain(0..).next().unwrap());
        matrix.push(floats);
        
    }

    matrix.shrink_to_fit();
    classes.shrink_to_fit();

    // Thread handles
    let mut handles: Vec<JoinHandle<_>> = Vec::new();
    // stdout lock
    let stdout = Arc::new(Mutex::new(0));

    // thread safe data structure for sharing the matrix
    let matrix = Arc::new(matrix);
    // Mutable data to be stored results of kmean calculation
    let results = Arc::new(Mutex::new(Vec::<Vec<Vec<f32>>>::new()));
    let scores = Arc::new(Mutex::new(Vec::<f32>::new()));
    

    for K in 1..6 {
        let mat = matrix.clone();
        let lock = stdout.clone();
        let r = results.clone();
        let s = scores.clone();
        handles.push(std::thread::spawn(move || {
            let (score, data) = kmeans(lock, mat, K);
            r.try_lock().unwrap().push(data);
            s.try_lock().unwrap().push(score);
        }));

    }

    for thread in handles {
        let _ = thread.join();
    }

    if let Ok(scores) = Arc::try_unwrap(scores) {
        if let Ok(scores) = scores.try_lock() {
            let min: f32 = scores.iter().cloned().fold(1./0., f32::min);
            if let Some(winner) = scores.iter().position(|&x| x == min) {
                if let Ok(matrix) = Arc::try_unwrap(results) {
                    if let Ok(matrix) = matrix.try_lock() {
                        if let Some(best_matrix) = matrix.get(winner){
                            let output = Spreadsheet {
                                headers: s.headers,
                                data: matrix_to_cells(best_matrix, &classes),
                                rows: s.rows.clone(),
                                cols: s.cols.clone() + 1,
                            };

                            output.write("output.tsv".into()).unwrap();
                            println!("Winning score={}", min);
                        }
                    }
                }
            } else {
                println!("Error encountered while generating scores");
            }
        }
    }
}

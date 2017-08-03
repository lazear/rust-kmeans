extern crate spreadsheet;
use spreadsheet::{Cell, Spreadsheet};
use std::collections::HashMap;
use std::env::*;

fn distance(x: &[f32], b: &[f32]) -> f32 {
    let mut sums: f32 = 0.0;
    for (i, j) in x.iter().zip(b.iter()) {
        sums += (i - j).powi(2);
    }
    sums
}

fn kmeans(matrix: &Vec<Vec<f32>>, k_values: &[usize]) {

    assert!(k_values.len() < matrix.len());

    
    let mut centroids: Vec<&Vec<f32>> = Vec::new();

    for k in k_values {
        centroids.push(matrix.get(*k).unwrap());
    }

    let mut iter = matrix.iter().enumerate();

    // iterate through each row in the matrix, and pick the best cluster membership
    while let Some((i, current)) = iter.next() {
        let mut scores: Vec<f32> = Vec::new();
        let mut cn =  centroids.iter().enumerate();
        while let Some((k, c)) = cn.next() {
            let s = distance(&current[..], c);
            println!("{}, {}", k, &s);
            scores.push(s);
        }            

        let min = scores.iter().cloned().fold(1./0., f32::min);
        let winning_k = scores.iter().position(|&x| x == min).unwrap();
        println!("{} best cluster is {} with a score of {}", i, winning_k, min);
    }




}

fn main() {
    let a = args().collect::<Vec<String>>();
    let filename = &a[1];
    let s = Spreadsheet::read(filename, '\t').unwrap();

    let mut enumerate = s.data.iter().enumerate();

    let mut matrix: Vec<Vec<f32>> = Vec::new();
    let mut classes: Vec<&str> = Vec::new();

    while let Some((i, row)) = enumerate.next() {
        // iterate across the cross and collect the underlying type
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

        // iterate across the cross and collect the class
        let class = row.iter().filter(|&cell| {
            match *cell {
                 Cell::String(_) => true,
                 _ => false,
            }
        }).map(|cell| {
            match *cell {
                Cell::String(ref s) => &s[..],
                _ => "",
            }
        }).collect::<Vec<&str>>();

        if class.len() != 1 {
            panic!("Too many string classes specified");
        }

        classes.push(class[0]);
        matrix.push(floats);
        
    }
    
    kmeans(&matrix, &[15, 55, 120]);

    //println!("{:?}", matrix.iter().zip(classes.iter()).collect::<Vec<(&Vec<f32>, &&str)>>());

}

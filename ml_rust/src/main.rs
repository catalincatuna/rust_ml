use std::{fs::File, i32, io::Write};

use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::prelude::*;
use ndarray::Array2;

fn main() {
    let original_data: Array2<f32> = array!(
        [1., 1., 1000., 1., 10.],
        [1., 0., 600., 1., 5.],
        [1., 0., 800., 1., 7.],
        [1., 0., 400., 1., 4.],
        [1., 0., 0., 1., 2.],
        [0., 0., 0., 1., 2.],
        [0., 0., 0., 1., 2.],
        [0., 1., 0., 1., 2.],
        [0., 1., 0., 1., 2.],
        [1., 1., 0., 1., 2.],
        [0., 1., 500., 0., 3.],
        [1., 0., 300., 1., 6.],
        [0., 0., 200., 0., 1.],
        [1., 1., 700., 1., 8.],
        [0., 1., 100., 0., 2.],
        [1., 0., 900., 1., 9.],
        [0., 0., 400., 0., 4.],
        [1., 1., 600., 1., 7.],
        [0., 1., 300., 0., 5.],
        [1., 0., 800., 1., 10.]
    );

    let feature_names = vec!["Watched TV", "Pet cat", "Rust lines", "Ate pizza"];

    let num_features = original_data.len_of(Axis(1)) - 1;
    let features = original_data.slice(s![.., 0..num_features]).to_owned();
    let labels = original_data.column(num_features).to_owned();

    let linfa_dataset = Dataset::new(features, labels)
        .map_targets(|x| match *x as i32 {
            i32::MIN..=4 => "Sad",
            5..=8 => "Ok",
            9..=i32::MAX => "Happy",
            _ => "Unknown",
        })
        .with_feature_names(feature_names);

    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .fit(&linfa_dataset)
        .unwrap();

    File::create("dt.tex")
        .unwrap()
        .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
        .unwrap();
}

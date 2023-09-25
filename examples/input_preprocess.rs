use std::{fs::File, io::Write};

use itertools::Itertools;
use json::{object, JsonValue};

fn main() {
    let inputs_raw =
        std::fs::read_to_string("/mnt/c/Users/aweso/Downloads/all_samples_for_halo2.json").unwrap();
    let inputs = json::parse(&inputs_raw).unwrap();

    let inputs = inputs.members().enumerate().map(|(index, input)| {
        let output_image = &input[0];
        let noise = &input[1];
        let category_vec = &input[2];

        object! {output_image: output_image.clone(), noise: noise.clone(), category_vec: category_vec.clone(), index: index}
    }).collect_vec();

    let inputs = vec![
        inputs[0..167].to_vec(),
        inputs[167..334].to_vec(),
        inputs[334..501].to_vec(),
        inputs[501..667].to_vec(),
        inputs[667..833].to_vec(),
        inputs[833..].to_vec(),
    ];

    let inputs = inputs.into_iter().map(JsonValue::Array).collect_vec();

    for (index, inputs) in inputs.into_iter().enumerate() {
        let output_string = inputs.to_string();
        let mut f = File::create(format!("./inputs/images_{index}.json")).unwrap();

        f.write_all(output_string.as_bytes()).unwrap();
    }
}

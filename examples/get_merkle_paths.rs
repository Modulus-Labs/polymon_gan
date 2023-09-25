use std::{path::Path, fs::File, io::Write};

use itertools::Itertools;
use json::{object, JsonValue};
use regex::Regex;

use halo2_base::{
    halo2_proofs::{
        circuit::Value,
        dev::MockProver,
        halo2curves::{
            bn256::{Bn256, Fr},
            FieldExt,
        },
        plonk::{keygen_pk, keygen_vk, Circuit, ProvingKey, VerifyingKey, verify_proof},
        poly::{
            commitment::Params,
            kzg::{commitment::ParamsKZG, multiopen::ProverSHPLONK},
        },
        SerdeFormat,
    },
    utils::fs::gen_srs,
};

fn main() {

    // let image: i32 = 1;
    // let layer = 6;

    let re: Regex = Regex::new("agg_hash_([0-9|-]+)").unwrap();

    let inputs = (0..6).into_iter().flat_map(|index| {
        let inputs_raw =
            std::fs::read_to_string(format!("/home/aweso/modulus_labs/polymon_gan/inputs/images_{index}.json")).unwrap();
        let json = json::parse(&inputs_raw).unwrap();
        json.members().cloned().collect_vec()
    }).collect_vec();

    for (image, input) in (0..1000).into_iter().zip(inputs.into_iter()) {
        let hashes = (1..10).into_iter().map(|layer| {
            let mut curr = 0;
            for i in 1..=10 - layer - 1 {
                curr *= 2;
                if image >= (curr + 1) * (1 << 10 - i) {
                    // curr += 1 << i - 1;
                    curr += 1;
                }
            }
        
            curr *= 2;
        
            if image >= (curr + 1) * (1 << layer) {
                curr += 1;
            }
            let index = curr;

            // dbg!((layer, index));

            // dbg!(format!("/home/aweso/modulus_labs/polymon_gan/agg_instances_{}/", layer + 1));

            let path = if layer == 1 {
                "/home/aweso/modulus_labs/polymon_gan/agg_instances/".to_owned()
            } else {
                format!("/home/aweso/modulus_labs/polymon_gan/agg_instances_{}/", layer - 1)
            };

            let files = std::fs::read_dir(Path::new(&path))
            .unwrap()
            .into_iter()
            .map(|snark_file| {
                let snark_file = snark_file.unwrap();
                let file_name = snark_file.file_name().into_string().unwrap();
        
                let id = re.captures(&file_name).unwrap()[1].to_owned();
        
                let id_split: Vec<String> = id.split('|').flat_map(|x| x.split('-').map(|x| x.to_owned()).collect::<Vec<String>>()).collect_vec();
        
                // let id_split: Vec<String> = vec![id.clone()];
        
                // let compare = |left: &&String, right: &&String| {
                //     Ord::cmp(
                //         &i32::from_str_radix(&left, 10).unwrap_or(1000),
                //         &i32::from_str_radix(&right, 10).unwrap_or(1000)
                //     )
                // };
        
                // let id_min = id_split.iter().min_by(compare).unwrap();
                // let id_max = id_split.iter().max_by(compare).unwrap();
                (snark_file, id_split)
            })
            .sorted_by(|first, second| {
                Ord::cmp(
                    &i32::from_str_radix(&first.1[0], 10).unwrap(),
                    &i32::from_str_radix(&second.1[0], 10).unwrap(),
                )
            }).map(|x| x.0)
            .collect_vec();

            let thing = files.get((index + 1) as usize).unwrap_or_else(|| &files[index as usize]).to_owned();
            // dbg!(thing.path());
            thing.path()
        }).map(|file| {
            let inputs_raw =
                std::fs::read_to_string(file).unwrap();
            let json = json::parse(&inputs_raw).unwrap();

            json["final_hash"].clone()
        }).collect_vec();

        let inputs_raw =
            std::fs::read_to_string(format!("/home/aweso/modulus_labs/polymon_gan/instances/image_{image}.json")).unwrap();
        let json = json::parse(&inputs_raw).unwrap();

        let image_values: JsonValue = json["output_image"].clone();

        let input_hash: JsonValue = json["input_hash"].clone();
        let output_hash: JsonValue = json["output_hash"].clone();

        let input_noise: JsonValue = input["noise"].clone();
        let category_vec: JsonValue = input["category_vec"].clone();

        let json = object! {merkle_path: hashes, input_hash: input_hash, output_hash: output_hash, image: image_values, input_noise: input_noise, category_vec: category_vec, index: image};

        let output_string = json.to_string();
        let mut f = File::create(format!("/home/aweso/modulus_labs/polymon_gan/final_images/image_{image}.json")).unwrap();

        f.write_all(output_string.as_bytes()).unwrap();
}

    // dbg!(files);
}
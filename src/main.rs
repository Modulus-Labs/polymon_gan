use std::{cell::RefCell, env::set_var, fs::File, io::{Write, BufWriter}, path::Path, time::Instant};

use circuit::GANCircuit;
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
            commitment::{Params, ParamsProver},
            kzg::{commitment::ParamsKZG, multiopen::{ProverSHPLONK, VerifierSHPLONK}, strategy::AccumulatorStrategy}, VerificationStrategy,
        },
        SerdeFormat,
    },
    utils::fs::gen_srs,
};
use halo2_machinelearning::felt_from_i64;
// use input_parsing::read_input;

use itertools::Itertools;
use json::{object, JsonValue};
use ndarray::Array1;
use poseidon_circuit::{Hashable, PrimeField};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use regex::Regex;
use snark_verifier::{pcs::kzg::{Bdfg21, Kzg}, loader::native::NativeLoader};
use snark_verifier_sdk::{
    gen_pk,
    
    gen_snark_shplonk,
    read_snark, CircuitExt, types::PoseidonTranscript, gen_evm_verifier_shplonk, gen_evm_proof_shplonk, encode_calldata, evm_verify, PublicAggregationCircuit,
    // read_pk, evm::{evm_verify, gen_evm_proof_shplonk, gen_evm_verifier_shplonk, encode_calldata}, CircuitExt,
};

use crate::{
    agg_circuit::MerkleAggregationCircuit, input_parsing::read_input,
    // circuit::{HASH_OUTPUT, OUTPUT_IMAGE},
};

pub mod agg_circuit;
pub mod circuit;
pub mod clamp;
pub mod generator_block;
pub mod input_packing;
pub mod input_parsing;

fn main() {
    // let mut instance = input.map(|x| value_to_option(*x).unwrap()).to_vec();
    // instance.append(&mut circuit.category_vec.map(|&x| value_to_option(x).unwrap()).to_vec());

    let now = Instant::now();

    // let mut params_max: ParamsKZG<Bn256> = gen_srs(24);
    // params_max.downsize(22);

    // let mut f = File::create("./params_22.params").unwrap();

    // params_max.write(&mut f).unwrap();
    
    let mut f = File::open("./params_22.params").unwrap();

    let params_max = ParamsKZG::read(&mut f).unwrap();


    println!("params generated in {}", now.elapsed().as_secs_f32());

    // set_var("VERIFY_CONFIG", "./configs/verify_circuit_big.config");

    let mut rng = ChaCha20Rng::from_entropy();

    // let snark = read_snark(Path::new("./test_snark")).unwrap();

    let now = Instant::now();

    // let input: Vec<i64> = vec![70147, -73987, 87368, -44360, -29247, -124004, -10372, -85356, 1541, -25968, -20995, 80488, -27080, -16523, -105025, 116763, 45732, -28779, -8166, 61043, 75154, 49209, 34688, 23311, -2633, -103012, 52498, 54229, 20347, 117133, -74929, -9738, 51743, 98152, 117692, 3708, -67673, -77858, 101531, -71525, -65367, 41154, 47856, 148643, -21650, 76813, 85847, 86281, 87806, 94260, 114532, 104066, -67379, -4276, 80622, 79709, -12844, 150931, 13986, 93886, 40562, 21843, -69419, -103558];

    // let input = input.iter().map(|x: &i64| Value::known(felt_from_i64(*x))).collect::<Array1<_>>();
    
    // let network_params = read_input("/home/ubuntu/gan_prover/", "gan_weights.json");

    // let params = {
    //     let mut params = params_max.clone();
    //     params.downsize(19);
    //     params
    // };

    // let dummy_circuit = GANCircuit::<Fr> {
    //     input: input.clone(),
    //     category_vec: Array1::from_vec(vec![Value::known(Fr::one()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero())]),
    //     params: network_params.clone(),
    //     input_hash: RefCell::new(None),
    //     output_hash: RefCell::new(None),
    //     output_image: RefCell::new(None),
    // };

    // let pk = gen_pk(&params, &dummy_circuit, None);
    


    // let snark =  {
    //     let input: Vec<i64> = vec![70147, -73987, 87368, -44360, -29247, -124004, -10372, -85356, 1541, -25968, -20995, 80488, -27080, -16523, -105025, 116763, 45732, -28779, -8166, 61043, 75154, 49209, 34688, 23311, -2633, -103012, 52498, 54229, 20347, 117133, -74929, -9738, 51743, 98152, 117692, 3708, -67673, -77858, 101531, -71525, -65367, 41154, 47856, 148643, -21650, 76813, 85847, 86281, 87806, 94260, 114532, 104066, -67379, -4276, 80622, 79709, -12844, 150931, 13986, 93886, 40562, 21843, -69419, -103558];

    //     let input = input.iter().map(|x: &i64| Value::known(felt_from_i64(*x))).collect::<Array1<_>>();
        
    //     let network_params = read_input("/home/ubuntu/gan_prover/", "gan_weights.json");
    
    //     let params = {
    //         let mut params = params_max.clone();
    //         params.downsize(19);
    //         params
    //     };
    
    //     let dummy_circuit = GANCircuit::<Fr> {
    //         input: input.clone(),
    //         category_vec: Array1::from_vec(vec![Value::known(Fr::one()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero())]),
    //         params: network_params.clone(),
    //         input_hash: RefCell::new(None),
    //         output_hash: RefCell::new(None),
    //         output_image: RefCell::new(None),
    //     };
    
    //     let pk = gen_pk(&params, &dummy_circuit, None);
    
    //     MockProver::run(19, &dummy_circuit, vec![vec![Fr::zero(), Fr::zero()]]).unwrap();
    
    //     gen_snark_shplonk(&params, &pk, dummy_circuit, &mut rng, Some(Path::new("./test_snark_scroll")))
    // };

    // let snark = read_snark(Path::new("./test_snark_agg_merkle")).unwrap();

    // println!("prover takes: {}", now.elapsed().as_secs_f32());

    // let agg_circuit =
    //     PublicAggregationCircuit::new(&params_max, vec![snark], false, &mut rng);

    // let instances = agg_circuit.instances();

    // let pk = gen_pk(&params_max, &agg_circuit, None);

    // MockProver::run(22, &agg_circuit, vec![vec![Fr::zero()]]).unwrap();

    // MockProver::run(22, &agg_circuit, agg_circuit.instances()).unwrap().assert_satisfied();

    // let now = Instant::now();

    // // let out = gen_snark_shplonk(&params_max, &pk, agg_circuit, &mut rng, Some(Path::new(&format!("./test_snark_agg_merkle"))));
    // let proof = gen_evm_proof_shplonk(&params_max, &pk, agg_circuit.clone(), agg_circuit.instances(), &mut rng);

    // println!("outer proof generated in {}", now.elapsed().as_secs_f32());

    // let verifier_contract = gen_evm_verifier_shplonk::<PublicAggregationCircuit>(&params_max, pk.get_vk(), agg_circuit.num_instance(), None);

    // println!("contract len: {:?}", verifier_contract.len());

    // println!("instances are {:?}, instance_len is {:?}, proof len is {:?}", agg_circuit.instances(), agg_circuit.num_instance(), proof.len());
    
    // let calldata = encode_calldata(&agg_circuit.instances(), &proof);
    
    // evm_verify(verifier_contract, agg_circuit.instances(), proof);


    // let mut transcript_read = PoseidonTranscript::<NativeLoader, &[u8]>::new(out.proof.as_slice());
    // let result = VerificationStrategy::<_, VerifierSHPLONK<_>>::finalize(
    //     verify_proof::<_, VerifierSHPLONK<_>, _, _, _>(
    //         params_max.verifier_params(),
    //         pk.get_vk(),
    //         AccumulatorStrategy::new(params_max.verifier_params()),
    //         &[&[instances[0].as_slice()]],
    //         &mut transcript_read,
    //     )
    //     .unwrap(),
    // );
    // assert!(result == true);

    // println!("prover takes: {}", now.elapsed().as_secs_f32());

    // let now = Instant::now();

    // let pk = read_pk::<GANCircuit::<Fr>>(Path::new("./gan_pk")).unwrap_or_else(|_| {
    //     let vk = keygen_vk(&params, &circuit).unwrap();

    //     println!("vk generated in {}", now.elapsed().as_secs_f32());

    //     let now = Instant::now();

    //     let pk = keygen_pk(&params, vk, &circuit).unwrap();

    //     let mut f = File::create(Path::new("./gan_pk")).unwrap();

    //     pk.write(&mut f, SerdeFormat::Processed).unwrap();

    //     pk
    // });

    // let dummy_circuit = GANCircuit::<Fr> {
    //     input: input.clone(),
    //     category_vec: Array1::from_vec(vec![Value::known(Fr::one()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero()), Value::known(Fr::zero())]),
    //     params: network_params.clone(),
    //     input_hash: RefCell::new(None),
    //     output_hash: RefCell::new(None),
    //     output_image: RefCell::new(None),
    // };

    // let pk = gen_pk(&params_max, &dummy_circuit, None);

    // MockProver::run(19, &dummy_circuit, vec![vec![Fr::zero(), Fr::zero()]]).unwrap();

    // let now = Instant::now();

    // let out = gen_snark_shplonk(&params_max, &pk, dummy_circuit, &mut rng, Some(Path::new(&format!("./test_snark"))));

    // println!("prover takes: {}", now.elapsed().as_secs_f32());

    // let instance = vec![dummy_circuit.input_hash.borrow().unwrap().clone(), 
    // dummy_circuit.output_hash.borrow().unwrap().clone()
    // // Fr::zero()
    // ];

    // MockProver::run(19, &dummy_circuit, vec![instance]).unwrap().assert_satisfied();



    // println!("pk generated in {}", now.elapsed().as_secs_f32());

    // let inputs: Vec<(Vec<i16>, Vec<Value<Fr>>, Vec<Value<Fr>>, JsonValue)> = {
    //     let inputs_raw = std::fs::read_to_string("./samples.json").unwrap();
    //     let inputs = json::parse(&inputs_raw).unwrap();

    //     inputs.members().map(|input| {
    //         let convert = |value: &JsonValue| -> Value<Fr> {
    //             Value::known(felt_from_i64(value.as_i64().unwrap()))
    //         };
    //         let output_image = input["output_image"].members().map(|value| value.as_i16().unwrap()).collect_vec();
    //         let noise = input["noise"].members().map(convert).collect_vec();
    //         let category_vec = input["category_vec"].members().map(convert).collect_vec();
    //         let index = input["index"].clone();

    //         (output_image, noise, category_vec, index)
    //     }).collect_vec()
    // };

    // for (_source_image, input, category_vec, json_index) in inputs.into_iter() {
    //     let index: usize = json_index.as_usize().unwrap();
    //     println!("Proving image {index}");
    //     let circuit = GANCircuit::<Fr> {
    //         input: Array1::from_vec(input),
    //         category_vec: Array1::from_vec(category_vec),
    //         params: network_params.clone(),
    //         input_hash: RefCell::new(None),
    //         output_hash: RefCell::new(None),
    //         output_image: RefCell::new(None),
    //     };

    //     MockProver::run(19, &circuit, vec![vec![Fr::zero(), Fr::zero()]]).unwrap();

    //     //Save Hashes and stuff
    //     {
    //         let output_image = circuit.output_image.borrow();
    //         let output_hash = circuit.output_hash.borrow();
    //         let input_hash = circuit.input_hash.borrow();

    //         let output_image = JsonValue::Array(output_image.as_ref().unwrap().into_iter().map(|x| JsonValue::Number((x.get_lower_128() as u64).into())).collect_vec());
    //         let output_hash = JsonValue::Array(output_hash.as_ref().unwrap().to_bytes().into_iter().map(|x| JsonValue::Number(x.into())).collect_vec());
    //         let input_hash = JsonValue::Array(input_hash.as_ref().unwrap().to_bytes().into_iter().map(|x| JsonValue::Number(x.into())).collect_vec());

    //         let output = object! {output_image: output_image, output_hash: output_hash, input_hash: input_hash, index: json_index};

    //         let output_string = output.to_string();
    //         let mut f = File::create(format!("./instances/image_{index}.json")).unwrap();

    //         f.write_all(output_string.as_bytes()).unwrap();

    //     }

    //     let now = Instant::now();

    //     let out = gen_snark_shplonk(&params, &pk, circuit, &mut rng, Some(Path::new(&format!("./snarks/gan_snark_{index}"))));
    //     println!("inner snark generated in {} for image {index}", now.elapsed().as_secs_f32());
    // }

//     let re = Regex::new("agg_snark_([0-9|-]+)").unwrap();

//     let dir = std::fs::read_dir(Path::new(&format!("./agg_snarks_2/")))
//     .unwrap()
//     .into_iter()
//     .map(|snark_file| {
//         let snark_file = snark_file.unwrap();
//         let file_name = snark_file.file_name().into_string().unwrap();

//         let id = re.captures(&file_name).unwrap()[1].to_owned();

//         let id_split: Vec<String> = id.split('|').flat_map(|x| x.split('-').map(|x| x.to_owned()).collect::<Vec<String>>()).collect_vec();

//         // let id_split: Vec<String> = vec![id.clone()];

//         let compare = |left: &&String, right: &&String| {
//             Ord::cmp(
//                 &i32::from_str_radix(&left, 10).unwrap_or(1000),
//                 &i32::from_str_radix(&right, 10).unwrap_or(1000)
//             )
//         };

//         let id_min = id_split.iter().min_by(compare).unwrap();
//         let id_max = id_split.iter().max_by(compare).unwrap();
//         (snark_file, format!("{id_min}-{id_max}"), id_split)
//     })
//     .sorted_by(|first, second| {
//         Ord::cmp(
//             &i32::from_str_radix(&first.2[0], 10).unwrap(),
//             &i32::from_str_radix(&second.2[0], 10).unwrap(),
//         )
//     })
//     .collect_vec();

// let mut pk: Option<ProvingKey<_>> = None;

// for snarks in dir.chunks(2) {
//     let snark1 = &snarks[0];
//     let snark2 = &snarks.get(1);

//     let id = format!(
//         "{}|{}",
//         snark1.1,
//         snark2.map(|x| x.1.clone()).unwrap_or_default()
//     );
//     //let id = snark1.1.clone();
//     let snark1 = read_snark(snark1.0.path()).unwrap();

//     // let second = &snark1.instances[0].remove(13);
//     // let first = &snark1.instances[0].remove(12);

//     // let new_instance = Fr::hasher().hash([first.clone(), second.clone()]);

//     // snark1.instances[0].push(new_instance);

//     let snark2 = snark2
//         .map(|snark2| read_snark(snark2.0.path()).unwrap())
//         .unwrap_or_else(|| {
//             snark1.clone()
//         });

//     // let second: &Fr = &snark2.instances[0].remove(13);
//     // let first = &snark2.instances[0].remove(12);

//     // let new_instance = Fr::hasher().hash([first.clone(), second.clone()]);

//     // snark2.instances[0].push(new_instance);

//     let agg_circuit =
//         MerkleAggregationCircuit::new(&params_max, vec![snark1, snark2], true, &mut rng);

//     if pk.is_none() {
//         let pk_agg = gen_pk(&params_max, &agg_circuit, None);

//         pk = Some(pk_agg);
//     };

//     let now = Instant::now();

//     // MockProver::run(22, &agg_circuit, vec![vec![Fr::zero(), Fr::zero()]]).unwrap();

//     let final_hash = agg_circuit.final_hash.clone();
//     let final_hash = JsonValue::Array(
//         final_hash
//             .to_bytes()
//             .into_iter()
//             .map(|x| JsonValue::Number(x.into()))
//             .collect_vec(),
//     );
//     let output = object! {id: id.clone(), final_hash: final_hash};

//     let output_string = output.to_string();
//     let mut f = File::create(format!("./agg_instances_3/agg_hash_{id}.json")).unwrap();

//     f.write_all(output_string.as_bytes()).unwrap();

//     let _proof = gen_snark_shplonk(
//         &params_max,
//         pk.as_ref().unwrap(),
//         agg_circuit,
//         &mut rng,
//         Some(Path::new(&format!("./agg_snarks_3/agg_snark_{id}"))),
//     );

//     println!("agg_snark generated in {}", now.elapsed().as_secs_f32());
// }


    // });

//     let re = Regex::new("agg_snark_([0-9|-]+)").unwrap();

//     for i in 4..=10 {

//         // let dir = std::fs::read_dir(Path::new(&format!("./agg_snarks_{}/", i - 1)))
//         let dir = std::fs::read_dir(Path::new(&format!("./agg_snarks_{}/", i - 1)))
//             .unwrap()
//             .into_iter()
//             .map(|snark_file| {
//                 let snark_file = snark_file.unwrap();
//                 let file_name = snark_file.file_name().into_string().unwrap();

//                 let id = re.captures(&file_name).unwrap()[1].to_owned();

//                 let id_split: Vec<String> = id.split('|').flat_map(|x| x.split('-').map(|x| x.to_owned()).collect::<Vec<String>>()).collect_vec();

//                 // let id_split: Vec<String> = vec![id.clone()];

//                 let compare = |left: &&String, right: &&String| {
//                     Ord::cmp(
//                         &i32::from_str_radix(&left, 10).unwrap_or(1000),
//                         &i32::from_str_radix(&right, 10).unwrap_or(1000)
//                     )
//                 };

//                 let id_min = id_split.iter().min_by(compare).unwrap();
//                 let id_max = id_split.iter().max_by(compare).unwrap();
//                 (snark_file, format!("{id_min}-{id_max}"), id_split)
//             })
//             .sorted_by(|first, second| {
//                 Ord::cmp(
//                     &i32::from_str_radix(&first.2[0], 10).unwrap(),
//                     &i32::from_str_radix(&second.2[0], 10).unwrap(),
//                 )
//             })
//             .collect_vec();

//         let mut pk: Option<ProvingKey<_>> = None;

//         for snarks in dir.chunks(2) {
//             let snark1 = &snarks[0];
//             let snark2 = &snarks.get(1);

//             let id = format!(
//                 "{}|{}",
//                 snark1.1,
//                 snark2.map(|x| x.1.clone()).unwrap_or_default()
//             );
//             //let id = snark1.1.clone();
//             let snark1 = read_snark(snark1.0.path()).unwrap();

//             // let second = &snark1.instances[0].remove(13);
//             // let first = &snark1.instances[0].remove(12);

//             // let new_instance = Fr::hasher().hash([first.clone(), second.clone()]);

//             // snark1.instances[0].push(new_instance);

//             let snark2 = snark2
//                 .map(|snark2| read_snark(snark2.0.path()).unwrap())
//                 .unwrap_or_else(|| {
//                     snark1.clone()
//                 });

//             // let second: &Fr = &snark2.instances[0].remove(13);
//             // let first = &snark2.instances[0].remove(12);

//             // let new_instance = Fr::hasher().hash([first.clone(), second.clone()]);

//             // snark2.instances[0].push(new_instance);

//             let agg_circuit =
//                 MerkleAggregationCircuit::new(&params_max, vec![snark1, snark2], true, &mut rng);

//             if pk.is_none() {
//                 let pk_agg = gen_pk(&params_max, &agg_circuit, None);

//                 pk = Some(pk_agg);
//             };

//             let now = Instant::now();

//             // MockProver::run(22, &agg_circuit, vec![vec![Fr::zero(), Fr::zero()]]).unwrap();

//             let final_hash = agg_circuit.final_hash.clone();
//             let final_hash = JsonValue::Array(
//                 final_hash
//                     .to_bytes()
//                     .into_iter()
//                     .map(|x| JsonValue::Number(x.into()))
//                     .collect_vec(),
//             );
//             let output = object! {id: id.clone(), final_hash: final_hash};

//             let output_string = output.to_string();
//             let mut f = File::create(format!("./agg_instances_{i}/agg_hash_{id}.json")).unwrap();

//             f.write_all(output_string.as_bytes()).unwrap();

//             let _proof = gen_snark_shplonk(
//                 &params_max,
//                 pk.as_ref().unwrap(),
//                 agg_circuit,
//                 &mut rng,
//                 Some(Path::new(&format!("./agg_snarks_{i}/agg_snark_{id}"))),
//             );

//             println!("agg_snark generated in {}", now.elapsed().as_secs_f32());
// }
//     }

    // let snark = read_snark(Path::new("stuff")).unwrap();

    // let agg_circuit = MerkleAggregationCircuit::new(&params_max, vec![snark], false, &mut rng);

    // let now = Instant::now();

    // let vk_agg = keygen_vk(&params_max, &agg_circuit).unwrap();

    // println!("vk_agg generated in {}", now.elapsed().as_secs_f32());

    // let now = Instant::now();

    // let pk_agg = keygen_pk(&params_max, vk_agg, &agg_circuit).unwrap();

    // println!("pk_agg generated in {}", now.elapsed().as_secs_f32());

    // let _now = Instant::now();

    // let _proof = gen_snark_shplonk(&params_max, &pk_agg, agg_circuit, &mut rng, None::<Box<Path>>);

    // println!("agg_snark generated in {}", now.elapsed().as_secs_f32());

    // MockProver::run(17, &circuit, vec![instance]).unwrap();

    // let output = OUTPUT.get().unwrap();

    // let output = Array::from_shape_vec((3, 64, 64), output.to_vec()).unwrap();
    // let output = output.map(|&x| felt_to_i64(value_to_option(x).unwrap()));

    // let output_real = {
    //     let outputs_raw = std::fs::read_to_string("/home/ubuntu/".to_owned() + "circuit_data.json").unwrap();
    //     let outputs = json::parse(&outputs_raw).unwrap();

    //     let outputs = outputs[0][2].members().map(|x| x.as_i64().unwrap()).collect_vec();

    //     Array::from_shape_vec((3, 64, 64), outputs).unwrap()
    // };

    // println!("random comparison #1: {:?}, {:?}", output.get((0, 5, 5)), output_real.get((0, 5, 5)));
    // println!("random comparison #2: {:?}, {:?}", output.get((0, 10, 10)), output_real.get((0, 10, 10)));
    // println!("random comparison #3: {:?}, {:?}", output.get((0, 15, 15)), output_real.get((0, 15, 15)));
    // println!("random comparison #4: {:?}, {:?}", output.get((0, 20, 20)), output_real.get((0, 20, 20)));

    // use plotters::prelude::*;
    // let root = BitMapBackend::new("gan_circuit.png", (1024*8, 3096*8)).into_drawing_area();
    // root.fill(&WHITE).unwrap();
    // let root = root.titled("gan_circuit", ("sans-serif", 500)).unwrap();
    // halo2_base::halo2_proofs::dev::CircuitLayout::default().render(20, &dummy_circuit, &root).unwrap();

    let snark = read_snark(Path::new("./final_snark")).unwrap();

    let agg_circuit = PublicAggregationCircuit::new(&params_max, vec![snark], true, &mut rng);

    let now = Instant::now();

    let vk_agg = keygen_vk(&params_max, &agg_circuit).unwrap();

    println!("vk_agg generated in {}", now.elapsed().as_secs_f32());

    let now = Instant::now();

    let pk_agg = keygen_pk(&params_max, vk_agg, &agg_circuit).unwrap();

    println!("pk_agg generated in {}", now.elapsed().as_secs_f32());

    let now = Instant::now();

    let proof = gen_evm_proof_shplonk(&params_max, &pk_agg, agg_circuit.clone(), agg_circuit.instances(), &mut rng);

    let mut f = File::create("./proof_dir/proof").unwrap();

    f.write_all(proof.as_slice()).unwrap();

    println!("outer proof generated in {}", now.elapsed().as_secs_f32());

    let verifier_contract = gen_evm_verifier_shplonk::<PublicAggregationCircuit>(&params_max, pk_agg.get_vk(), agg_circuit.num_instance(), None);

    let mut f = File::create("./proof_dir/verifier_contract_bytecode").unwrap();

    f.write_all(verifier_contract.as_slice()).unwrap();

    println!("contract len: {:?}", verifier_contract.len());

    println!("instances are {:?}, instance_len is {:?}, proof len is {:?}", agg_circuit.instances(), agg_circuit.num_instance(), proof.len());
    
    let calldata = encode_calldata(&agg_circuit.instances(), &proof);

    let mut f = File::create("./proof_dir/official_calldata").unwrap();

    f.write_all(calldata.as_slice()).unwrap();
    
    evm_verify(verifier_contract, agg_circuit.instances(), proof);

    let instances = &agg_circuit.instances()[0][0..12];

    let instances_output: Vec<_> = instances.iter().flat_map(|value| value.to_repr().as_ref().iter().rev().cloned().collect::<Vec<_>>()).collect();

    let mut f = File::create("./proof_dir/limbs_instance").unwrap();

    f.write_all(instances_output.as_slice()).unwrap();

    println!("Done!");
}

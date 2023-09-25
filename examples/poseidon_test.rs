use std::marker::PhantomData;

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
use itertools::Itertools;
use poseidon_circuit::{Hashable, P128Pow5T3Constants};
use snark_verifier::{util::hash::Poseidon, loader::Loader, PoseidonSpec, State};
use snark_verifier_sdk::NativeLoader;
use poseidon::{Spec, MDSMatrix, Matrix, MDSMatrices};

fn main() {
    // let input_hash: [u8; 32] = [77,155,203,102,45,34,233,50,248,157,103,58,98,246,39,193,221,176,205,198,152,55,10,143,196,143,47,141,15,25,18,8];
    // let output_hash: [u8; 32] = [195,114,5,78,159,86,164,106,226,189,22,102,216,229,60,166,119,99,39,13,145,105,203,154,45,129,86,82,32,208,82,44];

    let hash_1: [u8; 32] = [115,89,48,109,50,201,132,181,148,201,160,205,74,51,77,34,235,33,176,71,159,99,71,24,121,29,219,217,83,109,43,44];

    let hash_2: [u8; 32] = [105,191,153,43,190,2,78,208,2,239,254,153,7,32,162,80,139,64,65,116,56,189,190,247,63,11,192,83,163,147,92,32];

    let input_hash = Fr::from_bytes(&hash_1).unwrap();
    let output_hash = Fr::from_bytes(&hash_2).unwrap();

    let output = Fr::hasher().hash([input_hash, output_hash]).to_bytes();

    // let loader = NativeLoader;

    let mds = MDSMatrix(Matrix(<Fr as P128Pow5T3Constants>::mds()));
    let constants = <Fr as P128Pow5T3Constants>::round_constants();

    // let constants = std::iter::once([Fr::zero(), Fr::zero(), Fr::zero()]).chain(constants.into_iter()).collect_vec();

    let constants = Spec::<Fr, 3, 2>::calculate_optimized_constants(8, 57, constants, &mds);

    let (sparse_matrices, pre_sparse_mds) = Spec::<Fr, 3, 2>::calculate_sparse_matrices(57, &mds);

    let mds_matrices = MDSMatrices {
        mds,
        pre_sparse_mds,
        sparse_matrices,
    };

    let spec = Spec::<Fr, 3, 2> {
        r_f: 8,
        mds_matrices,
        constants,
    };

    let state = State::<> {
        inner: [Fr::zero(), Fr::zero(), Fr::zero()],
        _marker: PhantomData,
    };

    let mut poseidon_chip = Poseidon::<
        Fr,
        Fr,
        3,
        2,
    > {
        spec,
        default_state: state.clone(),
        state,
        buf: vec![],
    };

    poseidon_chip.update(&[input_hash, output_hash]);

    let output_2 = poseidon_chip.squeeze().to_bytes();

    dbg!((output, output_2));
}
use std::path::Path;

use halo2_base::{
    halo2_proofs::{
        halo2curves::bn256::{Bn256, Fr},
        plonk::verify_proof,
        poly::{
            commitment::ParamsProver,
            kzg::{
                commitment::ParamsKZG, multiopen::VerifierSHPLONK, strategy::AccumulatorStrategy,
            },
        },
    },
    utils::fs::gen_srs,
};
use itertools::Itertools;
use snark_verifier_sdk::{
    gen_pk,
    halo2::{read_snark, PoseidonTranscript},
    read_pk, NativeLoader,
};

fn main() {}

use std::ops::Neg;

use halo2_base::halo2_proofs::{circuit::Value, halo2curves::FieldExt};
use halo2_machinelearning::{
    felt_from_i64,
    nn_ops::{
        matrix_ops::linear::{
            c_batchnorm::CBatchnormChipParams, conv::Conv3DLayerParams,
            dist_add_fixed::DistributedAddFixedChipParams,
            dist_addmultadd_fixed::DistributedAddMulAddChipParams,
        },
        vector_ops::linear::fc::FcChipParams,
    },
};
use json::JsonValue;
use ndarray::{Array, Array1, Array2};

use crate::{circuit::GANParams, generator_block::GeneratorBlockChipParams};

const CATEGORY_SIZE: usize = 10;

pub fn read_input<F: FieldExt>(prefix: &str, file_name: &str) -> GANParams<F> {
    let inputs_raw = std::fs::read_to_string(prefix.to_owned() + file_name).unwrap();
    let inputs = json::parse(&inputs_raw).unwrap();

    let lin_1 = parse_fc(&inputs["l1"]);

    let gen_blocks = [
        parse_gen_block(&inputs["block2"]),
        parse_gen_block(&inputs["block3"]),
        parse_gen_block(&inputs["block4"]),
        parse_gen_block(&inputs["block5"]),
    ];

    let bn_6 = parse_bn(&inputs["b6"]);

    let conv_final = parse_conv(&inputs["l6"]);

    GANParams {
        lin_1,
        gen_blocks,
        bn_6,
        conv_final: conv_final.0,
        conv_final_bias: conv_final.1,
    }
}

fn parse_gen_block<F: FieldExt>(block: &JsonValue) -> GeneratorBlockChipParams<F> {
    let conv_1 = parse_conv(&block["c1"]);
    let conv_2 = parse_conv(&block["c2"]);

    let channel_count_1 = block["c1"]["weight_shape"][0].as_usize().unwrap();
    let channel_count_2 = block["c2"]["weight_shape"][0].as_usize().unwrap();

    let bn_1 = parse_cbatchnorm(&block["b1"], channel_count_1);
    let bn_2 = parse_cbatchnorm(&block["b2"], channel_count_2);

    let conv_sc = parse_conv(&block["c_sc"]);

    GeneratorBlockChipParams {
        conv_1_params: conv_1.0,
        conv_1_bias: conv_1.1,
        conv_2_params: conv_2.0,
        conv_2_bias: conv_2.1,
        residual_conv_params: conv_sc.0,
        residual_conv_bias: conv_sc.1,
        cbn_1_params: bn_1,
        cbn_2_params: bn_2,
    }
}

fn parse_conv<F: FieldExt>(
    block: &JsonValue,
) -> (Conv3DLayerParams<F>, DistributedAddFixedChipParams<F>) {
    let kernal_dim: Vec<_> = block["weight_shape"]
        .members()
        .map(|dim| dim.as_usize().unwrap())
        .collect();
    let kernal_dim: [usize; 4] = kernal_dim.try_into().unwrap();
    let kernal_vec: Vec<_> = block["weight"]
        .members()
        .map(|weight| Value::known(felt_from_i64(weight.as_i64().unwrap())))
        .collect();
    let conv_kernal = Array::from_shape_vec(kernal_dim, kernal_vec).unwrap();
    let conv_params = Conv3DLayerParams {
        kernals: conv_kernal,
    };

    let bias_vec: Vec<_> = block["bias"]
        .members()
        .map(|weight| Value::known(felt_from_i64(weight.as_i64().unwrap())))
        .collect();
    let bias = Array1::from_vec(bias_vec);
    let bias_params = DistributedAddFixedChipParams { scalars: bias };

    (conv_params, bias_params)
}

fn parse_cbatchnorm<F: FieldExt>(
    block: &JsonValue,
    channel_count: usize,
) -> CBatchnormChipParams<F> {
    let coeffs: Vec<_> = block["coeff"][0]
        .members()
        .map(|coeff| Value::known(felt_from_i64(coeff.as_i64().unwrap())))
        .collect();
    let mut shift: Vec<_> = block["e_x"][0]
        .members()
        .map(|shift| Value::known(felt_from_i64(shift.as_i64().unwrap().neg())))
        .collect();
    let bias: Vec<_> = block["beta"]
        .members()
        .map(|bias| Value::known(felt_from_i64(bias.as_i64().unwrap())))
        .collect();

    let scalar_mult = Array2::from_shape_vec((CATEGORY_SIZE, channel_count), coeffs).unwrap();
    shift.truncate(channel_count);
    let scalar_add = Array1::from_vec(shift);

    let scalar_bias = Array2::from_shape_vec((CATEGORY_SIZE, channel_count), bias).unwrap();

    CBatchnormChipParams {
        scalar_mult,
        scalar_add,
        scalar_bias,
    }
}

fn parse_bn<F: FieldExt>(bn_json: &JsonValue) -> DistributedAddMulAddChipParams<F> {
    let coeffs: Vec<_> = bn_json["coeff"][0]
        .members()
        .map(|coeff| Value::known(felt_from_i64(coeff.as_i64().unwrap())))
        .collect();
    let shift: Vec<_> = bn_json["e_x"][0]
        .members()
        .map(|shift| Value::known(felt_from_i64(shift.as_i64().unwrap())))
        .collect();
    let bias: Vec<_> = bn_json["beta"]
        .members()
        .map(|bias| Value::known(felt_from_i64(bias.as_i64().unwrap())))
        .collect();

    let scalars = coeffs
        .iter()
        .zip(shift.iter())
        .zip(bias.iter())
        .map(|((&coeff, &shift), &bias)| (coeff, shift, bias))
        .collect();

    DistributedAddMulAddChipParams { scalars }
}

fn parse_fc<F: FieldExt>(fc_json: &JsonValue) -> FcChipParams<F> {
    let weights: Vec<_> = fc_json["weight"]
        .members()
        .map(|weight| Value::known(felt_from_i64(weight.as_i64().unwrap())))
        .collect();

    let weight_dim: Vec<_> = fc_json["weight_shape"]
        .members()
        .map(|dim| dim.as_usize().unwrap())
        .collect();
    let weight_dim: [usize; 2] = weight_dim.try_into().unwrap();

    let weights = Array::from_shape_vec(weight_dim, weights).unwrap();
    let weights = weights.permuted_axes([1, 0]);

    let biases: Array1<_> = fc_json["bias"]
        .members()
        .map(|bias| Value::known(felt_from_i64(bias.as_i64().unwrap())))
        .collect();

    FcChipParams { weights, biases }
}

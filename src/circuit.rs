use std::cell::RefCell;

use halo2_base::{
    halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        halo2curves::{bn256::Fr, FieldExt},
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Fixed, Instance},
    },
    utils::value_to_option,
};
use halo2_machinelearning::nn_ops::{
    lookup_ops::DecompTable,
    matrix_ops::{
        linear::{
            c_batchnorm::{CBatchnormChip, CBatchnormChipConfig, CBatchnormChipParams},
            conv::{Conv3DLayerChip, Conv3DLayerConfig, Conv3DLayerParams},
            dist_add_fixed::{
                DistributedAddFixedChip, DistributedAddFixedChipParams, DistributedAddFixedConfig,
            },
            dist_addmultadd_fixed::DistributedAddMulAddConfig,
            dist_addmultadd_fixed::{DistributedAddMulAddChip, DistributedAddMulAddChipParams},
        },
        non_linear::{
            norm_2d::{Normalize2DChipConfig, Normalize2dChip, Normalize2dConfig},
            relu_norm_2d::{ReluNorm2DChip, ReluNorm2DChipConfig, ReluNorm2DConfig},
        },
    },
    vector_ops::{
        linear::fc::{FcChip, FcChipConfig, FcChipParams, FcConfig},
        non_linear::tanh::{TanHChip, TanHChipConfig, TanHConfig},
    },
    ColumnAllocator, Decomp16, InputSizeConfig, NNLayer,
};
use ndarray::{Array, Array1, Array2, Array3, Array4};
use once_cell::sync::OnceCell;
use poseidon_circuit::{
    poseidon::{primitives::ConstantLengthIden3, PaddedWord, Pow5Chip, Pow5Config, Sponge},
    Hashable,
};
use snark_verifier_sdk::CircuitExt;

use crate::{
    clamp::ClampChipConfig,
    clamp::{ClampChip, ClampConfig},
    generator_block::{
        GeneratorBlockChip, GeneratorBlockChipConfig, GeneratorBlockChipParams,
        GeneratorBlockConfig,
    },
    input_packing::{self, InputPackingChip, InputPackingConfig},
};

pub static HASH_OUTPUT: OnceCell<Value<Fr>> = OnceCell::new();
pub static HASH_INPUT: OnceCell<Value<Fr>> = OnceCell::new();
pub static OUTPUT_IMAGE: OnceCell<Array3<Value<Fr>>> = OnceCell::new();

#[derive(Clone)]
pub struct GANParams<F: FieldExt> {
    pub lin_1: FcChipParams<F>,
    pub gen_blocks: [GeneratorBlockChipParams<F>; 4],
    pub bn_6: DistributedAddMulAddChipParams<F>,
    pub conv_final: Conv3DLayerParams<F>,
    pub conv_final_bias: DistributedAddFixedChipParams<F>,
}

#[derive(Clone, Debug)]
pub struct GANConfig<F: FieldExt> {
    pub instance: Column<Instance>,
    pub input_advice: Column<Advice>,
    pub gen_block_chips: [GeneratorBlockConfig<F>; 4],
    pub range_table: DecompTable<F, Decomp16>,
    pub clamp_chip: ClampConfig<F>,
    pub final_scale: DistributedAddMulAddConfig<F>,
    pub lin_chip: FcConfig<F>,
    pub conv_6_chip: Conv3DLayerConfig<F>,
    pub conv_6_bias_chip: DistributedAddFixedConfig<F>,
    pub relu_chip: ReluNorm2DConfig<F>,
    pub norm_chip: Normalize2dConfig<F>,
    pub norm_small_chip: Normalize2dConfig<F>,
    pub tanh_chip: TanHConfig<F>,
    pub hash_chip: Pow5Config<F, 3, 2>,
    pub input_packing_chip: InputPackingConfig,
}

const CATEGORY_SIZE: usize = 10;
pub struct GANCircuit<F: FieldExt> {
    pub input: Array1<Value<F>>,
    pub category_vec: Array1<Value<F>>,
    pub params: GANParams<F>,
    pub output_hash: RefCell<Option<F>>,
    pub input_hash: RefCell<Option<F>>,
    pub output_image: RefCell<Option<Array3<F>>>,
}

impl Circuit<Fr> for GANCircuit<Fr> {
    type Config = GANConfig<Fr>;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        let gen_block_builder =
            |input_depth: usize, output_depth: usize| GeneratorBlockChipParams::<Fr> {
                conv_1_params: Conv3DLayerParams {
                    kernals: Array4::from_shape_simple_fn(
                        (input_depth, 3, 3, output_depth),
                        Value::unknown,
                    ),
                },
                conv_1_bias: DistributedAddFixedChipParams {
                    scalars: Array1::from_shape_simple_fn(output_depth, Value::unknown),
                },
                conv_2_params: Conv3DLayerParams {
                    kernals: Array4::from_shape_simple_fn(
                        (output_depth, 3, 3, output_depth),
                        Value::unknown,
                    ),
                },
                conv_2_bias: DistributedAddFixedChipParams {
                    scalars: Array1::from_shape_simple_fn(output_depth, Value::unknown),
                },
                residual_conv_params: Conv3DLayerParams {
                    kernals: Array4::from_shape_simple_fn(
                        (input_depth, 3, 3, output_depth),
                        Value::unknown,
                    ),
                },
                residual_conv_bias: DistributedAddFixedChipParams {
                    scalars: Array1::from_shape_simple_fn(output_depth, Value::unknown),
                },
                cbn_1_params: CBatchnormChipParams {
                    scalar_mult: Array2::from_shape_simple_fn(
                        (CATEGORY_SIZE, input_depth),
                        Value::unknown,
                    ),
                    scalar_add: Array1::from_shape_simple_fn(input_depth, Value::unknown),
                    scalar_bias: Array2::from_shape_simple_fn(
                        (CATEGORY_SIZE, input_depth),
                        Value::unknown,
                    ),
                },
                cbn_2_params: CBatchnormChipParams {
                    scalar_mult: Array2::from_shape_simple_fn(
                        (CATEGORY_SIZE, output_depth),
                        Value::unknown,
                    ),
                    scalar_add: Array1::from_shape_simple_fn(output_depth, Value::unknown),
                    scalar_bias: Array2::from_shape_simple_fn(
                        (CATEGORY_SIZE, output_depth),
                        Value::unknown,
                    ),
                },
            };

        let gan_params = GANParams {
            lin_1: FcChipParams {
                weights: Array2::from_shape_simple_fn((64, 4096), Value::unknown),
                biases: Array1::from_shape_simple_fn(4096, Value::unknown),
            },
            gen_blocks: [
                gen_block_builder(256, 192),
                gen_block_builder(192, 128),
                gen_block_builder(128, 64),
                gen_block_builder(64, 32),
            ],
            bn_6: DistributedAddMulAddChipParams {
                scalars: Array1::from_shape_simple_fn(32, || {
                    (Value::unknown(), Value::unknown(), Value::unknown())
                }),
            },
            conv_final: Conv3DLayerParams {
                kernals: Array4::from_shape_simple_fn((32, 3, 3, 3), Value::unknown),
            },
            conv_final_bias: DistributedAddFixedChipParams {
                scalars: Array1::from_shape_simple_fn(3, Value::unknown),
            },
        };

        GANCircuit {
            input: Array1::from_shape_simple_fn(64, Value::unknown),
            category_vec: Array1::from_shape_simple_fn(CATEGORY_SIZE, Value::unknown),
            params: gan_params,
            output_hash: RefCell::new(None),
            input_hash: RefCell::new(None),
            output_image: RefCell::new(None),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let mut advice_allocator = ColumnAllocator::<Advice>::new(meta, 392);
        let mut fixed_allocator = ColumnAllocator::<Fixed>::new(meta, 96);

        let range_table = DecompTable::<_, Decomp16>::configure(meta);

        let norm_config = Normalize2DChipConfig {
            input_height: 4,
            input_width: 4,
            input_depth: 256,
            range_table: range_table.clone(),
            folding_factor: 1,
        };

        let norm_small_chip = Normalize2dChip::configure(
            meta,
            norm_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let bn_small_config = CBatchnormChipConfig {
            input_height: 4,
            input_width: 4,
            input_depth: 256,
            category_size: CATEGORY_SIZE,
        };

        let bn_small_chip = CBatchnormChip::configure(
            meta,
            bn_small_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let relu_norm_small_config = ReluNorm2DChipConfig {
            input_height: 4,
            input_width: 4,
            input_depth: 256,
            range_table: range_table.clone(),
            folding_factor: 1,
        };

        let relu_norm_small_chip = ReluNorm2DChip::configure(
            meta,
            relu_norm_small_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let gen_block_1_config = GeneratorBlockChipConfig {
            output_width: 8,
            output_height: 8,
            output_depth: 192,
            input_depth: 256,
            category_size: CATEGORY_SIZE,
            bn_small_chip,
            relu_norm_small_chip,
            range_table: range_table.clone(),
        };

        let gen_block_1 = GeneratorBlockChip::configure(
            meta,
            gen_block_1_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let lin_config = FcChipConfig {
            weights_height: 4096,
            weights_width: 64,
            folding_factor: 1,
        };

        let lin_chip = FcChip::configure(
            meta,
            lin_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let gen_block_2_config = GeneratorBlockChipConfig {
            output_width: 16,
            output_height: 16,
            output_depth: 128,
            input_depth: 192,
            category_size: CATEGORY_SIZE,
            bn_small_chip: gen_block_1.bn_upscaled_chip.clone(),
            relu_norm_small_chip: gen_block_1.relu_norm_upscaled_chip.clone(),
            range_table: range_table.clone(),
        };

        let gen_block_2 = GeneratorBlockChip::configure(
            meta,
            gen_block_2_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let gen_block_3_config = GeneratorBlockChipConfig {
            output_width: 32,
            output_height: 32,
            output_depth: 64,
            input_depth: 128,
            category_size: CATEGORY_SIZE,
            bn_small_chip: gen_block_2.bn_upscaled_chip.clone(),
            relu_norm_small_chip: gen_block_2.relu_norm_upscaled_chip.clone(),
            range_table: range_table.clone(),
        };

        let gen_block_3 = GeneratorBlockChip::configure(
            meta,
            gen_block_3_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let gen_block_4_config = GeneratorBlockChipConfig {
            output_width: 64,
            output_height: 64,
            output_depth: 32,
            input_depth: 64,
            category_size: CATEGORY_SIZE,
            bn_small_chip: gen_block_3.bn_upscaled_chip.clone(),
            relu_norm_small_chip: gen_block_3.relu_norm_upscaled_chip.clone(),
            range_table: range_table.clone(),
        };

        let gen_block_4 = GeneratorBlockChip::configure(
            meta,
            gen_block_4_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let relu_chip = gen_block_4.relu_norm_upscaled_chip.clone();

        let conv_6_chip = gen_block_4.conv_chip.clone();
        let conv_6_bias_chip = gen_block_4.conv_bias_chip.clone();

        let norm_final_config = Normalize2DChipConfig {
            input_height: 64,
            input_width: 64,
            input_depth: 32,
            range_table: range_table.clone(),
            folding_factor: 8,
        };

        let norm_final = Normalize2dChip::configure(
            meta,
            norm_final_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let norm_chip = norm_final;

        let gen_block_chips = [gen_block_1, gen_block_2, gen_block_3, gen_block_4];

        let clamp_config = ClampChipConfig {
            range_table: range_table.clone(),
        };

        let clamp_chip = ClampChip::configure(
            meta,
            clamp_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let final_scale_config = InputSizeConfig {
            input_height: 64,
            input_width: 64,
            input_depth: 3,
        };

        let final_scale = DistributedAddMulAddChip::configure(
            meta,
            final_scale_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let tanh_config = TanHChipConfig {
            range_table: range_table.clone(),
        };

        let tanh_chip = TanHChip::configure(
            meta,
            tanh_config,
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        let hash_advice = advice_allocator.take(meta, 4);
        let state = hash_advice[0..3].to_vec();
        let partial_sbox = hash_advice[3];

        // let rc_a = (0..3).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        // let rc_b = (0..3).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        let hash_fixed = fixed_allocator.take(meta, 6);
        let rc_a = hash_fixed[0..3].to_vec();
        let rc_b = hash_fixed[3..6].to_vec();

        meta.enable_constant(rc_b[0]);

        let hash_chip = Pow5Chip::configure::<<Fr as Hashable>::SpecType>(
            meta,
            state.try_into().unwrap(),
            partial_sbox,
            rc_a.try_into().unwrap(),
            rc_b.try_into().unwrap(),
        );

        let input_packing_chip = InputPackingChip::configure(
            meta,
            (64, 64),
            &mut advice_allocator,
            &mut fixed_allocator,
        );

        println!(
            "advice column count {}, fixed column count {}",
            meta.num_advice_columns(),
            meta.num_fixed_columns()
        );

        GANConfig {
            instance: {
                let col = meta.instance_column();
                meta.enable_equality(col);
                col
            },
            input_advice: {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            },
            gen_block_chips,
            range_table,
            clamp_chip,
            final_scale,
            lin_chip,
            conv_6_chip,
            conv_6_bias_chip,
            relu_chip,
            norm_chip,
            norm_small_chip,
            tanh_chip,
            hash_chip,
            input_packing_chip,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), PlonkError> {
        config
            .range_table
            .layout(layouter.namespace(|| "Range Table"))?;
        let layer_params = &self.params;
        let lin_chip = FcChip::construct(config.lin_chip.clone());
        let norm_small_chip = Normalize2dChip::construct(config.norm_small_chip.clone());
        let gen_block_chips = config
            .gen_block_chips
            .iter()
            .map(|chip| GeneratorBlockChip::construct(chip.clone()))
            .collect::<Vec<_>>();

        // let gen_block_chips = [GeneratorBlockChip::construct(config.gen_block_chips[0].clone()), GeneratorBlockChip::construct(config.gen_block_chips[1].clone())];

        let bn_chip = DistributedAddMulAddChip::construct(config.final_scale.clone());
        let relu_norm_chip = ReluNorm2DChip::construct(config.relu_chip.clone());

        let conv_final_chip = Conv3DLayerChip::construct(config.conv_6_chip.clone());
        let conv_bias_chip = DistributedAddFixedChip::construct(config.conv_6_bias_chip.clone());
        let norm_chip = Normalize2dChip::construct(config.norm_chip.clone());

        let tanh_chip = TanHChip::construct(config.tanh_chip.clone());

        let clamp_chip = ClampChip::construct(config.clamp_chip.clone());

        let input_packing_chip =
            InputPackingChip::<Fr>::construct(config.input_packing_chip.clone());

        let inputs = layouter.assign_region(
            || "input assignment",
            |mut region| {
                let inputs: Result<Vec<_>, _> = self
                    .input
                    .iter()
                    .enumerate()
                    .map(|(row, &input)| {
                        region.assign_advice(
                            || "copy input to advice",
                            config.input_advice,
                            row,
                            || input,
                        )
                    })
                    .collect();
                Ok(Array::from_shape_vec(64, inputs?).unwrap())
            },
        )?;

        let category_vec = layouter.assign_region(
            || "input assignment",
            |mut region| {
                let inputs: Result<Vec<_>, _> = self
                    .category_vec
                    .iter()
                    .enumerate()
                    .map(|(row, &input)| {
                        region.assign_advice(
                            || "copy input to advice",
                            config.input_advice,
                            row + inputs.len(),
                            || input,
                        )
                    })
                    .collect();
                Ok(Array::from_shape_vec(CATEGORY_SIZE, inputs?).unwrap())
            },
        )?;

        //hash input
        {
            let mut outputs = inputs.iter().chain(category_vec.iter());
            let initial_hash = {
                let chip = Pow5Chip::construct(config.hash_chip.clone());
                let mut sponge: Sponge<
                    Fr,
                    _,
                    <Fr as Hashable>::SpecType,
                    _,
                    ConstantLengthIden3<2>,
                    3,
                    2,
                > = Sponge::new(chip, layouter.namespace(|| "Poseidon Sponge"))?;
                sponge.absorb(
                    layouter.namespace(|| "sponge 0 message 1"),
                    PaddedWord::Message(outputs.next().unwrap().clone()),
                )?;
                sponge.absorb(
                    layouter.namespace(|| "sponge 0 message 2"),
                    PaddedWord::Message(outputs.next().unwrap().clone()),
                )?;
                sponge
                    .finish_absorbing(layouter.namespace(|| "finish absorbing sponge 0"))?
                    .squeeze(layouter.namespace(|| "sponge 0 output"))
            };

            let final_hash = outputs
                .enumerate()
                .fold(initial_hash, |accum, (index, input)| {
                    let chip = Pow5Chip::construct(config.hash_chip.clone());
                    let mut sponge: Sponge<
                        Fr,
                        _,
                        <Fr as Hashable>::SpecType,
                        _,
                        ConstantLengthIden3<2>,
                        3,
                        2,
                    > = Sponge::new(
                        chip,
                        layouter.namespace(|| format!("Poseidon Sponge {index}")),
                    )?;
                    sponge.absorb(
                        layouter.namespace(|| format!("sponge {index} message 1")),
                        PaddedWord::Message(accum?),
                    )?;
                    sponge.absorb(
                        layouter.namespace(|| format!("sponge {index} message 2")),
                        PaddedWord::Message(input.clone()),
                    )?;
                    sponge
                        .finish_absorbing(
                            layouter.namespace(|| format!("finish absorbing sponge {index}")),
                        )?
                        .squeeze(layouter.namespace(|| format!("sponge {index} output")))
                })?;

            layouter.constrain_instance(final_hash.cell(), config.instance, 0)?;

            let input_hash = value_to_option(final_hash.value());

            if input_hash.is_some() {
                *self.input_hash.borrow_mut() = input_hash.cloned();
            }
        }

        let input = lin_chip.add_layer(
            &mut layouter.namespace(|| "lin_1"),
            inputs,
            layer_params.lin_1.clone(),
        )?;
        let input = input.into_shape((256, 4, 4)).unwrap();

        let input = norm_small_chip.add_layer(&mut layouter, input, ())?;

        // let input_i32: ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 3]>> = input.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        // println!("pre_gen_block {:?}", input_i32);

        let mut input = input.permuted_axes([0, 2, 1]);

        println!("preproccessing done!");

        // input = gen_block_chips[0].add_layer(
        //         &mut layouter,
        //         (input, category_vec.clone()),
        //         layer_params.gen_blocks[0].clone(),
        //     )?;
        //     // let input_i32 = input.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        //     // println!("gen_block_results {:?}", input_i32);
        //     println!("gen block done!");


        for (gen_chip, gen_params) in gen_block_chips
            .into_iter()
            .zip(layer_params.gen_blocks.iter())
        {
            input = gen_chip.add_layer(
                &mut layouter,
                (input, category_vec.clone()),
                gen_params.clone(),
            )?;
            // let input_i32 = input.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

            // println!("gen_block_results {:?}", input_i32);
            println!("gen block done!");
        }

        let input = bn_chip.add_layer(&mut layouter, input, self.params.bn_6.clone())?;
        let input = relu_norm_chip.add_layer(&mut layouter, input, ())?;

        // let input_i32 = input.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        // println!("final_bn {:?}", input_i32);

        let input =
            conv_final_chip.add_layer(&mut layouter, input, self.params.conv_final.clone())?;
        let input =
            conv_bias_chip.add_layer(&mut layouter, input, self.params.conv_final_bias.clone())?;
        let input = norm_chip.add_layer(&mut layouter, input, ())?;

        println!("final conv done!");

        let dim = input.dim();

        let input = input.to_shape(dim.0 * dim.1 * dim.2).unwrap().to_owned();

        // let input_i32 = input.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        // println!("pre_tanh {:?}", input_i32);

        let input = tanh_chip.add_layer(&mut layouter, input, ())?;
        let input = input.to_shape(dim).unwrap().to_owned();

        println!("tanh done!");

        let final_scale_params = DistributedAddMulAddChipParams {
            scalars: Array1::from_shape_simple_fn(3, || {
                (
                    Value::known(Fr::from(128)),
                    Value::known(Fr::zero()),
                    Value::known(Fr::from(8_388_608)),
                )
            }),
        };

        let input = bn_chip.add_layer(&mut layouter, input, final_scale_params)?;

        // let input_i32 = input.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        // println!("pre_clamp {:?}", input_i32);

        println!("final_scale done!");

        let dim = input.dim();

        let input = input.to_shape(dim.0 * dim.1 * dim.2).unwrap().to_owned();

        let input_final = clamp_chip.add_layer(&mut layouter, input, ())?;

        let input_final = input_final.to_shape(dim).unwrap().to_owned();

        let input_final = norm_chip.add_layer(&mut layouter, input_final, ())?;

        if value_to_option(input_final[(0, 0, 0)].value()).is_some() {
            *self.output_image.borrow_mut() =
                Some(input_final.map(|x| value_to_option(x.value()).unwrap().clone()));
        }

        let output = input_packing_chip.add_layer(
            &mut layouter.namespace(|| "pack outputs"),
            input_final,
            (),
        )?;

        //hash output
        {
            let mut outputs = output.iter();
            let initial_hash = {
                let chip = Pow5Chip::construct(config.hash_chip.clone());
                let mut sponge: Sponge<
                    Fr,
                    _,
                    <Fr as Hashable>::SpecType,
                    _,
                    ConstantLengthIden3<2>,
                    3,
                    2,
                > = Sponge::new(chip, layouter.namespace(|| "Poseidon Sponge"))?;
                sponge.absorb(
                    layouter.namespace(|| "sponge 0 message 1"),
                    PaddedWord::Message(outputs.next().unwrap().clone()),
                )?;
                sponge.absorb(
                    layouter.namespace(|| "sponge 0 message 2"),
                    PaddedWord::Message(outputs.next().unwrap().clone()),
                )?;
                sponge
                    .finish_absorbing(layouter.namespace(|| "finish absorbing sponge 0"))?
                    .squeeze(layouter.namespace(|| "sponge 0 output"))
            };

            let final_hash = outputs
                .enumerate()
                .fold(initial_hash, |accum, (index, input)| {
                    let chip = Pow5Chip::construct(config.hash_chip.clone());
                    let mut sponge: Sponge<
                        Fr,
                        _,
                        <Fr as Hashable>::SpecType,
                        _,
                        ConstantLengthIden3<2>,
                        3,
                        2,
                    > = Sponge::new(
                        chip,
                        layouter.namespace(|| format!("Poseidon Sponge {index}")),
                    )?;
                    sponge.absorb(
                        layouter.namespace(|| format!("sponge {index} message 1")),
                        PaddedWord::Message(accum?),
                    )?;
                    sponge.absorb(
                        layouter.namespace(|| format!("sponge {index} message 2")),
                        PaddedWord::Message(input.clone()),
                    )?;
                    sponge
                        .finish_absorbing(
                            layouter.namespace(|| format!("finish absorbing sponge {index}")),
                        )?
                        .squeeze(layouter.namespace(|| format!("sponge {index} output")))
                })?;

            layouter.constrain_instance(final_hash.cell(), config.instance, 1)?;

            let output_hash = value_to_option(final_hash.value());

            if output_hash.is_some() {
                *self.output_hash.borrow_mut() = output_hash.cloned();
            }
        }

        println!("done!");

        // let input_final_i32 = input_final.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        // println!("result {:?}", input_final_i32);

        // if let Some(_) = OUTPUT.get() {} else {
        //     OUTPUT.set(input_final.iter().map(|x| x.value().cloned()).collect()).unwrap();
        // }

        Ok(())
    }
}

impl CircuitExt<Fr> for GANCircuit<Fr> {
    fn num_instance(&self) -> Vec<usize> {
        vec![2]
    }

    fn instances(&self) -> Vec<Vec<Fr>> {
        vec![vec![
            self.input_hash.borrow().unwrap(),
            self.output_hash.borrow().unwrap(),
        ]]
    }
}

use halo2_base::halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter},
    plonk::{Advice, ConstraintSystem, Error as PlonkError, Fixed},
};
use halo2_machinelearning::nn_ops::{
    lookup_ops::DecompTable,
    matrix_ops::{
        linear::{
            c_batchnorm::{
                CBatchnormChip, CBatchnormChipConfig, CBatchnormChipParams, CBatchnormConfig,
            },
            conv::{
                Conv3DLayerChip, Conv3DLayerConfig, Conv3DLayerConfigParams, Conv3DLayerParams,
            },
            dist_add_fixed::{
                DistributedAddFixedChip, DistributedAddFixedChipParams, DistributedAddFixedConfig,
            },
            residual_add::{ResidualAdd2DChip, ResidualAdd2DConfig},
            upsample::upsample,
        },
        non_linear::{
            norm_2d::{Normalize2DChipConfig, Normalize2dChip, Normalize2dConfig},
            relu_norm_2d::{ReluNorm2DChip, ReluNorm2DChipConfig, ReluNorm2DConfig},
        },
    },
    ColumnAllocator, Decomp16, InputSizeConfig, NNLayer,
};

use ndarray::{Array1, Array3};

#[derive(Clone, Debug)]
pub struct GeneratorBlockConfig<F: FieldExt> {
    bn_small_chip: CBatchnormConfig<F>,
    pub bn_upscaled_chip: CBatchnormConfig<F>,
    relu_norm_small_chip: ReluNorm2DConfig<F>,
    pub relu_norm_upscaled_chip: ReluNorm2DConfig<F>,
    conv_small_chip: Conv3DLayerConfig<F>,
    pub conv_chip: Conv3DLayerConfig<F>,
    pub conv_bias_chip: DistributedAddFixedConfig<F>,
    pub norm_chip: Normalize2dConfig<F>,
    residual_add_chip: ResidualAdd2DConfig<F>,
    pointwise_conv_chip: Conv3DLayerConfig<F>,
}

pub struct GeneratorBlockChip<F: FieldExt> {
    config: GeneratorBlockConfig<F>,
}

impl<F: FieldExt> Chip<F> for GeneratorBlockChip<F> {
    type Config = GeneratorBlockConfig<F>;

    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

#[derive(Clone, Debug)]
pub struct GeneratorBlockChipParams<F: FieldExt> {
    pub conv_1_params: Conv3DLayerParams<F>,
    pub conv_1_bias: DistributedAddFixedChipParams<F>,
    pub conv_2_params: Conv3DLayerParams<F>,
    pub conv_2_bias: DistributedAddFixedChipParams<F>,
    pub residual_conv_params: Conv3DLayerParams<F>,
    pub residual_conv_bias: DistributedAddFixedChipParams<F>,
    pub cbn_1_params: CBatchnormChipParams<F>,
    pub cbn_2_params: CBatchnormChipParams<F>,
}

pub struct GeneratorBlockChipConfig<F: FieldExt> {
    pub output_width: usize,
    pub output_height: usize,
    pub output_depth: usize,
    pub input_depth: usize,
    pub category_size: usize,
    pub bn_small_chip: CBatchnormConfig<F>,
    pub relu_norm_small_chip: ReluNorm2DConfig<F>,
    pub range_table: DecompTable<F, Decomp16>,
}

impl<F: FieldExt> NNLayer<F> for GeneratorBlockChip<F> {
    type ConfigParams = GeneratorBlockChipConfig<F>;

    type LayerParams = GeneratorBlockChipParams<F>;

    type LayerInput = (Array3<AssignedCell<F, F>>, Array1<AssignedCell<F, F>>);

    type LayerOutput = Array3<AssignedCell<F, F>>;

    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config_params: Self::ConfigParams,
        advice_allocator: &mut ColumnAllocator<Advice>,
        fixed_allocator: &mut ColumnAllocator<Fixed>,
    ) -> <Self as Chip<F>>::Config {
        let conv_folding_factor = if config_params.output_depth <= 128 {
            16
        } else {
            8
        };
        let bn_upscaled_config = CBatchnormChipConfig {
            input_height: config_params.output_height,
            input_width: config_params.output_width,
            input_depth: config_params.output_depth,
            category_size: config_params.category_size,
        };

        let bn_upscaled_chip =
            CBatchnormChip::configure(meta, bn_upscaled_config, advice_allocator, fixed_allocator);

        let relu_norm_upscaled_config = ReluNorm2DChipConfig {
            input_height: config_params.output_height,
            input_width: config_params.output_width,
            input_depth: config_params.output_depth,
            range_table: config_params.range_table.clone(),
            folding_factor: 8,
        };

        let relu_norm_upscaled_chip = ReluNorm2DChip::configure(
            meta,
            relu_norm_upscaled_config,
            advice_allocator,
            fixed_allocator,
        );

        let conv_small_config = Conv3DLayerConfigParams {
            ker_width: 3,
            ker_height: 3,
            padding_width: 1,
            padding_height: 1,
            folding_factor: conv_folding_factor,
            input_height: config_params.output_height,
            input_width: config_params.output_width,
            input_depth: config_params.input_depth,
        };

        let conv_small_chip =
            Conv3DLayerChip::configure(meta, conv_small_config, advice_allocator, fixed_allocator);

        let conv_config = Conv3DLayerConfigParams {
            ker_width: 3,
            ker_height: 3,
            padding_width: 1,
            padding_height: 1,
            folding_factor: conv_folding_factor,
            input_height: config_params.output_height,
            input_width: config_params.output_width,
            input_depth: config_params.output_depth,
        };

        let conv_chip =
            Conv3DLayerChip::configure(meta, conv_config, advice_allocator, fixed_allocator);

        let conv_bias_config = InputSizeConfig {
            input_height: config_params.output_height,
            input_width: config_params.output_width,
            input_depth: config_params.output_depth,
        };

        let conv_bias_chip = DistributedAddFixedChip::configure(
            meta,
            conv_bias_config.clone(),
            advice_allocator,
            fixed_allocator,
        );

        let norm_config = Normalize2DChipConfig {
            input_height: config_params.output_height,
            input_width: config_params.output_width,
            input_depth: config_params.output_depth,
            range_table: config_params.range_table,
            folding_factor: 4,
        };

        let norm_chip =
            Normalize2dChip::configure(meta, norm_config, advice_allocator, fixed_allocator);

        let residual_add_chip =
            ResidualAdd2DChip::configure(meta, conv_bias_config, advice_allocator, fixed_allocator);

        let pointwise_conv_config = Conv3DLayerConfigParams {
            ker_width: 1,
            ker_height: 1,
            padding_width: 0,
            padding_height: 0,
            folding_factor: conv_folding_factor,
            input_height: config_params.output_height,
            input_width: config_params.output_width,
            input_depth: config_params.input_depth,
        };

        let pointwise_conv_chip = Conv3DLayerChip::configure(
            meta,
            pointwise_conv_config,
            advice_allocator,
            fixed_allocator,
        );

        GeneratorBlockConfig {
            bn_small_chip: config_params.bn_small_chip,
            bn_upscaled_chip,
            relu_norm_small_chip: config_params.relu_norm_small_chip,
            relu_norm_upscaled_chip,
            conv_small_chip,
            conv_chip,
            conv_bias_chip,
            norm_chip,
            residual_add_chip,
            pointwise_conv_chip,
        }
    }

    fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        (input, category): Self::LayerInput,
        layer_params: Self::LayerParams,
    ) -> Result<Self::LayerOutput, PlonkError> {
        let config = &self.config;
        let bn_small_chip = CBatchnormChip::construct(config.bn_small_chip.clone());
        let bn_upscaled_chip = CBatchnormChip::construct(config.bn_upscaled_chip.clone());
        let relu_norm_small_chip = ReluNorm2DChip::construct(config.relu_norm_small_chip.clone());
        let relu_norm_upscaled_chip =
            ReluNorm2DChip::construct(config.relu_norm_upscaled_chip.clone());
        let conv_small_chip = Conv3DLayerChip::construct(config.conv_small_chip.clone());
        let conv_chip = Conv3DLayerChip::construct(config.conv_chip.clone());
        let conv_bias_chip = DistributedAddFixedChip::construct(config.conv_bias_chip.clone());
        let norm_chip = Normalize2dChip::construct(config.norm_chip.clone());
        let residual_add_chip = ResidualAdd2DChip::construct(config.residual_add_chip.clone());
        let pointwise_conv_chip = Conv3DLayerChip::construct(config.pointwise_conv_chip.clone());

        let residual = input.clone();

        let input = bn_small_chip.add_layer(
            layouter,
            (input, category.clone()),
            layer_params.cbn_1_params,
        )?;

        let input = relu_norm_small_chip.add_layer(layouter, input, ())?;

        // let input_i32 = input.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        // println!("first bn {:?}", input_i32);

        let input = upsample(input, 2);

        let input = conv_small_chip.add_layer(layouter, input, layer_params.conv_1_params)?;

        let input = conv_bias_chip.add_layer(layouter, input, layer_params.conv_1_bias)?;

        let input = norm_chip.add_layer(layouter, input, ())?;

        // Ok(input)

        // // let input_i32 = input.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        // // println!("first conv {:?}", input_i32);

        let input =
            bn_upscaled_chip.add_layer(layouter, (input, category), layer_params.cbn_2_params)?;

        let input = relu_norm_upscaled_chip.add_layer(layouter, input, ())?;

        // let input_i32 = input.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        // println!("second bn {:?}", input_i32);

        let input = conv_chip.add_layer(layouter, input, layer_params.conv_2_params)?;

        let input = conv_bias_chip.add_layer(layouter, input, layer_params.conv_2_bias)?;

        let input = norm_chip.add_layer(layouter, input, ())?;

        // let input_i32 = input.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        // println!("second conv {:?}", input_i32);

        let residual = upsample(residual, 2);

        let residual =
            pointwise_conv_chip.add_layer(layouter, residual, layer_params.residual_conv_params)?;

        let residual =
            conv_bias_chip.add_layer(layouter, residual, layer_params.residual_conv_bias)?;

        let residual = norm_chip.add_layer(layouter, residual, ())?;

        // let input_i32 = residual.map(|x| felt_to_i64(*value_to_option(x.value()).unwrap()));

        // println!("residual {:?}", input_i32);

        residual_add_chip.add_layer(layouter, [input, residual], ())
    }
}

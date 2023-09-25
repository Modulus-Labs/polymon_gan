use std::marker::PhantomData;

use halo2_base::halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{
        Advice, Column, ConstraintSystem, Error as PlonkError, Expression, Fixed, Selector,
        TableColumn,
    },
    poly::Rotation,
};
use halo2_machinelearning::nn_ops::{ColumnAllocator, NNLayer};

use itertools::Itertools;
use ndarray::{Array1, Array3};

#[derive(Clone, Debug)]
pub struct InputPackingConfig {
    input: Column<Advice>,
    output: Array1<Column<Advice>>,
    output_width: usize,
    output_height: usize,
    selector: Selector,
    lookup: TableColumn,
}

pub struct InputPackingChip<F: FieldExt> {
    config: InputPackingConfig,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> Chip<F> for InputPackingChip<F> {
    type Config = InputPackingConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt> NNLayer<F> for InputPackingChip<F> {
    type ConfigParams = (usize, usize);

    type LayerInput = Array3<AssignedCell<F, F>>;

    type LayerOutput = Array1<AssignedCell<F, F>>;

    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config_params: Self::ConfigParams,
        advice_allocator: &mut ColumnAllocator<Advice>,
        _: &mut ColumnAllocator<Fixed>,
    ) -> <Self as Chip<F>>::Config {
        let lookup = meta.lookup_table_column();
        let output_len = 30;
        let advice = advice_allocator.take(meta, output_len + 1);
        let input = advice[0];
        let output = Array1::from_vec(advice[1..output_len + 1].to_vec());

        let comp_selector = meta.complex_selector();

        for &item in output.iter() {
            meta.lookup("lookup", |meta| {
                let s_elt = meta.query_selector(comp_selector);
                let word = meta.query_advice(item, Rotation::cur());
                vec![(s_elt * word, lookup)]
            });
        }

        let selector = meta.selector();

        meta.create_gate("Input Unpacking", |meta| {
            let sel = meta.query_selector(selector);
            let input = meta.query_advice(input, Rotation::cur());
            let output = output.map(|&column| meta.query_advice(column, Rotation::cur()));

            let base = F::from(256);
            let output_sum = output;
            let (_, sum) = output_sum
                .into_iter()
                .enumerate()
                .reduce(|(_, accum), (index, item)| {
                    let true_base = Expression::Constant(
                        (0..index).fold(F::from(1), |expr, _input| expr * base),
                    );
                    (0, accum + (item * true_base))
                })
                .unwrap();

            vec![sel * (sum - input)]
        });

        InputPackingConfig {
            input,
            output,
            selector,
            output_width: config_params.0,
            output_height: config_params.1,
            lookup,
        }
    }

    fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Self::LayerInput,
        _layer_params: Self::LayerParams,
    ) -> Result<Self::LayerOutput, PlonkError> {
        layouter.assign_table(
            || "packing rangecheck table",
            |mut table| {
                for offset in 0..256 {
                    let value: u64 = offset.try_into().unwrap();
                    table.assign_cell(
                        || format!("decomp_table row {offset}"),
                        self.config.lookup,
                        offset,
                        || Value::known(F::from(value)),
                    )?;
                }
                Ok(())
            },
        )?;

        layouter.assign_region(
            || "unpacking input",
            |mut region| {
                input
                    .iter()
                    .chunks(30)
                    .into_iter()
                    .enumerate()
                    .map(|(row, input)| {
                        let input = input.collect_vec();
                        self.config.selector.enable(&mut region, row)?;

                        let output = input.iter().enumerate().fold(
                            Value::known(F::zero()),
                            |accum, (index, item)| {
                                let base = F::from(256).pow(&[index as u64, 0, 0, 0]);
                                accum + (item.value().map(|&x| x * base))
                            },
                        );

                        self.config
                            .output
                            .iter()
                            .zip_longest(input.iter())
                            .enumerate()
                            .map(|(_index, items)| {
                                let &column = items.clone().left().unwrap();
                                if let Some(item) = items.right() {
                                    item.copy_advice(
                                        || "copy input for packing",
                                        &mut region,
                                        column,
                                        row,
                                    )
                                } else {
                                    region.assign_advice(|| "zero input for padding for packing", column, row, || Value::known(F::zero()))
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        region.assign_advice(
                            || "input_packing output",
                            self.config.input,
                            row,
                            || output,
                        )
                    })
                    .collect::<Result<Array1<AssignedCell<F, F>>, _>>()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use halo2_machinelearning::{
        felt_from_i64, felt_to_i64,
        nn_ops::{ColumnAllocator, DefaultDecomp, NNLayer},
    };

    use halo2_base::{
        halo2_proofs::{
            arithmetic::FieldExt,
            circuit::{Layouter, SimpleFloorPlanner, Value},
            dev::MockProver,
            halo2curves::bn256::Fr,
            plonk::{
                Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Fixed, Instance,
            },
        },
        utils::value_to_option,
    };
    use itertools::Itertools;
    use ndarray::{stack, Array, Array1, Array2, Array3, Array4, Axis, Zip};
    use poseidon_circuit::PrimeField;

    use super::{InputPackingChip, InputPackingConfig};

    #[derive(Clone, Debug)]
    struct InputUnpackingTestConfig {
        input: Column<Instance>,
        input_advice: Column<Advice>,
        output: Column<Instance>,
        conv_chip: InputPackingConfig,
    }

    struct InputUnpackingTestCircuit<F: FieldExt> {
        pub input: Array3<Value<F>>,
    }

    const WIDTH: usize = 8;
    const HEIGHT: usize = 8;
    const DEPTH: usize = 4;

    impl Circuit<Fr> for InputUnpackingTestCircuit<Fr> {
        type Config = InputUnpackingTestConfig;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                input: Array::from_shape_simple_fn((DEPTH, WIDTH, HEIGHT), || Value::unknown()),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let mut advice_allocator = ColumnAllocator::<Advice>::new(meta, 0);

            let mut fixed_allocator = ColumnAllocator::<Fixed>::new(meta, 0);

            let conv_chip = InputPackingChip::configure(
                meta,
                (WIDTH, HEIGHT),
                &mut advice_allocator,
                &mut fixed_allocator,
            );

            InputUnpackingTestConfig {
                input: {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                },
                output: {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                },
                input_advice: {
                    let col = meta.advice_column();
                    meta.enable_equality(col);
                    col
                },
                conv_chip,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), PlonkError> {
            let conv_chip = InputPackingChip::construct(config.conv_chip);

            let inputs = layouter.assign_region(
                || "input assignment",
                |mut region| {
                    let inputs: Result<Vec<_>, _> = self
                        .input
                        .iter()
                        .enumerate()
                        .map(|(row, _)| {
                            region.assign_advice_from_instance(
                                || "copy input to advice",
                                config.input,
                                row,
                                config.input_advice,
                                row,
                            )
                        })
                        .collect();
                    Ok(Array::from_shape_vec((DEPTH, WIDTH, HEIGHT), inputs?).unwrap())
                },
            )?;

            let output = conv_chip.add_layer(&mut layouter, inputs, ())?;

            for (row, output) in output.iter().enumerate() {
                layouter.constrain_instance(output.cell(), config.output, row)?;
            }

            Ok(())
        }
    }

    // const TEST_INPUT: [u64; DEPTH*WIDTH*HEIGHT] = [10, 0, 10, 0];

    #[test]
    ///test that a simple 8x8x4 w/ 3x3x4 conv works; input and kernal are all 1
    fn test_simple_conv() -> Result<(), PlonkError> {
        let circuit = InputUnpackingTestCircuit::<Fr> {
            input: Array::from_shape_simple_fn((DEPTH, WIDTH, HEIGHT), || Value::known(Fr::one())),
        };

        let mut output = vec![Fr::from_repr([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]).unwrap(); 8];

        output.push(Fr::from_repr([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap());

        MockProver::run(
            9,
            &circuit,
            vec![circuit
                .input
                .iter()
                .map(|x| value_to_option(*x).unwrap())
                .collect_vec(), output],
        )
        .unwrap()
        .assert_satisfied();

        Ok(())
    }
}

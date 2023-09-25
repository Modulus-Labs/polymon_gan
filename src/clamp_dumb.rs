use std::{marker::PhantomData, ops::Neg};

use halo2_base::halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error as PlonkError, Expression, Selector},
    poly::Rotation,
};

use ndarray::{stack, Array1, Array2, Axis, Array3, Array};

use halo2_machinelearning::{
    felt_from_i64,
    nn_ops::{lookup_ops::DecompTable, DefaultDecomp, Decomp16},
    nn_ops::{
        matrix_ops::non_linear::norm_2d::{Normalize2dChip, Normalize2dConfig},
        NNLayer,
    },
};

use itertools::Itertools;

#[derive(Clone, Debug)]
pub struct ClampConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub inputs: Array1<Column<Advice>>,
    pub outputs: Array1<Column<Advice>>,
    pub eltwise_inter: Array2<Column<Advice>>,
    pub ranges: Column<Advice>,
    pub comp_signs: Array1<Column<Advice>>,
    pub comp_selector: Selector,
    pub output_selector: Selector,
    _marker: PhantomData<F>,
}

/// Chip for 2d Sigmoid
///
/// Order for ndarrays is Channel-in, Width, Height
pub struct ClampChip<F: FieldExt, const BASE: usize> {
    config: ClampConfig<F>,
}

impl<F: FieldExt, const BASE: usize> Chip<F> for ClampChip<F, BASE> {
    type Config = ClampConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt, const BASE: usize> ClampChip<F, BASE> {
    const DEPTH_AXIS: Axis = Axis(0);
    const COLUMN_AXIS: Axis = Axis(1);
    const ROW_AXIS: Axis = Axis(2);
    const ADVICE_LEN: usize = 3;
    const CEIL: u64 = 16_711_680; //2^16
    const MAX_VALUE: u64 = 16_711_680;
    const FLOOR: i64 = 0;
    const MIN_VALUE: i64 = 0;

    pub fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: Array1<Column<Advice>>,
        ranges: Column<Advice>,
        outputs: Array1<Column<Advice>>,
        eltwise_inter: Array2<Column<Advice>>,
        range_table: DecompTable<F, Decomp16>,
    ) -> <Self as Chip<F>>::Config {
        let comp_selector = meta.complex_selector();
        let output_selector = meta.selector();

        let max_value = F::from(Self::MAX_VALUE);

        let min_value = felt_from_i64(Self::MIN_VALUE);

        for &item in eltwise_inter.iter() {
            meta.lookup("lookup", |meta| {
                let s_elt = meta.query_selector(comp_selector);
                let word = meta.query_advice(item, Rotation::cur());
                vec![(s_elt * word, range_table.range_check_table)]
            });
        }

        let mut comp_signs = vec![];

        let constant_1 = Expression::Constant(F::from(1));

        meta.create_gate("Sigmoid 2D Comparison", |meta| {
            let sel = meta.query_selector(comp_selector);

            //iterate over all elements to the input
            let (expressions, comp_signs_col) = eltwise_inter.axis_iter(Axis(0)).zip(inputs.iter()).fold((vec![], vec![]), |(mut expressions, mut comp_signs), (eltwise_inter, &input)| {
                let mut eltwise_inter = eltwise_inter.to_vec();
                let comp_sign_col = eltwise_inter.remove(0);
                let base: u64 = BASE.try_into().unwrap();
                assert_eq!(
                    Self::ADVICE_LEN, eltwise_inter.len(),
                    "Must pass in sufficient advice columns for eltwise intermediate operations: passed in {}, need {}", 
                    eltwise_inter.len(), Self::ADVICE_LEN
                );
                let input = meta.query_advice(input, Rotation::cur());
                let comp_sign = meta.query_advice(comp_sign_col, Rotation::cur());
                let iter = eltwise_inter.iter();
                let base = F::from(base);
                let word_sum = iter
                    .clone()
                    .enumerate()
                    .map(|(index, column)| {
                        let b = meta.query_advice(*column, Rotation::cur());
                        let true_base = (0..index).fold(F::from(1), |expr, _input| expr * base);
                        b * true_base
                    })
                    .reduce(|accum, item| accum + item)
                    .unwrap();

                let comp = meta.query_advice(ranges, Rotation::cur());
                let constant_1 = Expression::Constant(F::from(1));
                expressions.push(
                    sel.clone() * (word_sum - ((comp_sign.clone() * (input.clone() - comp.clone())) + ((constant_1-comp_sign) * (comp - input))))
                );

                comp_signs.push(comp_sign_col);

                (expressions, comp_signs)
            });

            comp_signs = comp_signs_col;
            expressions
        });

        meta.create_gate("Sigmoid 2D Output", |meta| -> Vec<Expression<F>> {
            inputs
                .iter()
                .zip(outputs.iter())
                .zip(comp_signs.iter())
                .fold(
                    vec![],
                    |mut expressions, ((&input, &output), &comp_sign)| {
                        let sel = meta.query_selector(output_selector);
                        let input = meta.query_advice(input, Rotation::cur());
                        let output = meta.query_advice(output, Rotation::cur());

                        let comp_sign_1 = meta.query_advice(comp_sign, Rotation::cur());
                        let comp_sign_2 = meta.query_advice(comp_sign, Rotation::next());

                        expressions.push(
                            sel * (output
                                - (comp_sign_1.clone() * Expression::Constant(max_value)
                                    + ((constant_1.clone() - comp_sign_1)
                                        * (comp_sign_2.clone() * (input)
                                            + ((constant_1.clone() - comp_sign_2)
                                                * Expression::Constant(min_value)))))),
                        );

                        expressions
                    },
                )
        });

        ClampConfig {
            inputs,
            outputs,
            eltwise_inter,
            ranges,
            comp_signs: Array1::from_vec(comp_signs),
            comp_selector,
            output_selector,
            _marker: PhantomData,
        }
    }

    pub fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: &Array3<AssignedCell<F, F>>,
    ) -> Result<Array3<AssignedCell<F, F>>, PlonkError> {
        let base: u128 = BASE.try_into().unwrap();
        let config = &self.config;

        let ciel = F::from(Self::CEIL);
        let max_value = F::from(Self::MAX_VALUE);

        let floor = felt_from_i64(Self::FLOOR);
        let min_value = felt_from_i64(Self::MIN_VALUE);

        let dim = inputs.dim();

        layouter.assign_region(
            || "apply 2d sigmoid",
            |mut region| {
                let outputs = inputs.axis_iter(Self::DEPTH_AXIS).enumerate().map(|(channel, inputs)| {
                    inputs.axis_iter(Axis(1))
                    .enumerate()
                    .map(|(row, inputs)| {
                        let offset = row * 2;
                        let offset_2 = offset + 1;
                        self.config.comp_selector.enable(&mut region, offset)?;
                        self.config.comp_selector.enable(&mut region, offset + 1)?;
                        self.config.output_selector.enable(&mut region, offset)?;
                        let outputs = inputs
                            .iter()
                            .zip(config.inputs.iter())
                            .zip(config.outputs.iter())
                            .zip(config.eltwise_inter.axis_iter(Self::COLUMN_AXIS))
                            .zip(config.comp_signs.iter())
                            .map(
                                |(
                                    (((input, &input_col), &output_col), eltwise_inter),
                                    &bit_sign_col,
                                )| {
                                    input.copy_advice(
                                        || "eltwise input",
                                        &mut region,
                                        input_col,
                                        offset,
                                    )?;
                                    input.copy_advice(
                                        || "eltwise input",
                                        &mut region,
                                        input_col,
                                        offset_2,
                                    )?;

                                    let comp_sign_1 =
                                        input.value().map(|x| x > &ciel && x < &F::TWO_INV);

                                    let comp_sign_2 =
                                        input.value().map(|x| x > &floor || x < &F::TWO_INV);

                                    // let word_repr: Value<Vec<u32>> = output_i32.map(|x| {
                                    //     let str = format!("{:o}", x.abs());
                                    //     str.chars()
                                    //         .map(|char| char.to_digit(8).unwrap())
                                    //         .rev()
                                    //         .collect()
                                    // });

                                    let difference_1 = input.value().map(|x| {
                                        if x > &ciel && x < &F::TWO_INV {
                                            *x - &ciel
                                        } else {
                                            ciel - x
                                        }
                                    });

                                    let difference_2 = input.value().map(|x| {
                                        if x > &floor || x < &F::TWO_INV {
                                            *x - &floor
                                        } else {
                                            floor - x
                                        }
                                    });

                                    let word_repr_1: Value<Vec<u16>> = difference_1.and_then(|x| {
                                        let mut result = vec![];
                                        let mut x = x.get_lower_128();

                                        loop {
                                            let m = x % base;
                                            x /= base;

                                            result.push(m as u16);
                                            if x == 0 {
                                                break;
                                            }
                                        }

                                        Value::known(result)
                                    });
                                    let word_repr_2: Value<Vec<u16>> = difference_2.and_then(|x| {
                                        let mut result = vec![];
                                        let mut x = x.get_lower_128();

                                        loop {
                                            let m = x % base;
                                            x /= base;

                                            result.push(m as u16);
                                            if x == 0 {
                                                break;
                                            }
                                        }

                                        Value::known(result)
                                    });

                                    region.assign_advice(
                                        || "sigmoid comp_sign_1",
                                        bit_sign_col,
                                        offset,
                                        || comp_sign_1.map(|x| F::from(x)),
                                    )?;
                                    region.assign_advice(
                                        || "sigmoid comp_sign_2",
                                        bit_sign_col,
                                        offset_2,
                                        || comp_sign_2.map(|x| F::from(x)),
                                    )?;
                                    region.assign_advice(
                                        || "sigmoid range ciel",
                                        config.ranges,
                                        offset,
                                        || Value::known(ciel),
                                    )?;
                                    region.assign_advice(
                                        || "sigmoid range floor",
                                        config.ranges,
                                        offset_2,
                                        || Value::known(floor),
                                    )?;
                                    let _: Vec<_> = (0..eltwise_inter.len() - 1)
                                        .map(|index_col| {
                                            region
                                                .assign_advice(
                                                    || "sigmoid word_repr_1",
                                                    eltwise_inter[index_col + 1],
                                                    offset,
                                                    || {
                                                        word_repr_1.clone().map(|x| match index_col
                                                            >= x.len()
                                                        {
                                                            false => F::from(x[index_col] as u64),
                                                            true => F::from(0),
                                                        })
                                                    },
                                                )
                                                .unwrap();
                                            region
                                                .assign_advice(
                                                    || "sigmoid word_repr_2",
                                                    eltwise_inter[index_col + 1],
                                                    offset_2,
                                                    || {
                                                        word_repr_2.clone().map(|x| match index_col
                                                            >= x.len()
                                                        {
                                                            false => F::from(x[index_col] as u64),
                                                            true => F::from(0),
                                                        })
                                                    },
                                                )
                                                .unwrap();
                                        })
                                        .collect();
                                    region.assign_advice(
                                        || "sigmoid_output",
                                        output_col,
                                        offset,
                                        || {
                                            input.value().map(|&x| {
                                                match (
                                                    x > ciel && x < F::TWO_INV,
                                                    x > floor || x < F::TWO_INV,
                                                ) {
                                                    (true, _) => max_value,
                                                    (false, true) => x,
                                                    (_, false) => min_value,
                                                }
                                            })
                                        },
                                    )
                                },
                            )
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok::<_, PlonkError>(Array1::from_vec(outputs))
                    }).collect::<Result<Vec<_>, _>>()})
                    .flatten_ok().flatten_ok().collect::<Result<Vec<_>, _>>()?;
                Ok(Array::from_shape_vec(dim, outputs).unwrap())
            },
        )
    }
}

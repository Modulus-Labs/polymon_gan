#![allow(clippy::clone_on_copy)]
#[cfg(feature = "display")]
use ark_std::{end_timer, start_timer};
use halo2_base::{
    gates::{GateInstructions},
    halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, AssignedCell, Value},
        halo2curves::bn256::{Bn256, Fq, Fr, G1Affine},
        plonk::{self, Circuit, Selector, Column, Advice, Fixed, Instance, ConstraintSystem},
        poly::{commitment::ParamsProver, kzg::commitment::ParamsKZG},
    },
    utils::value_to_option,
};
use itertools::Itertools;
use poseidon_circuit::{poseidon::{Pow5Config, Pow5Chip, Sponge, primitives::ConstantLengthIden3, PaddedWord}, Hashable};
use rand::Rng;
use snark_verifier::{
    loader::{
        self,
        halo2::{halo2_ecc::{self, ecc::EccChip, fields::fp::{FpConfig, FpStrategy}, halo2_base::{gates::{range::RangeConfig, flex_gate::FlexGateConfig}, utils::modulus, AssignedValue, ContextParams, Context}}},
        ScalarLoader, native::NativeLoader,
    },
    pcs::{kzg::{Bdfg21, Kzg, KzgSuccinctVerifyingKey, KzgAccumulator, KzgAs}, AccumulationSchemeProver},
    util::{hash::Poseidon, arithmetic::fe_to_limbs}, Error, verifier::PlonkVerifier,
};
use snark_verifier_sdk::{
    LIMBS, SnarkWitness, types::{POSEIDON_SPEC, PoseidonTranscript, Plonk}, BITS, aggregate, flatten_accumulator,
};
use std::{cell::RefCell, rc::Rc, fs::File, env};

use snark_verifier_sdk::{CircuitExt, Snark};

pub type Svk = KzgSuccinctVerifyingKey<G1Affine>;
pub type BaseFieldEccChip = halo2_ecc::ecc::BaseFieldEccChip<G1Affine>;
pub type Halo2Loader<'a> = loader::halo2::Halo2Loader<'a, G1Affine, BaseFieldEccChip>;
pub type Shplonk = Plonk<Kzg<Bn256, Bdfg21>>;

#[derive(Clone)]
pub struct MerkleAggregationCircuit {
    pub svk: KzgSuccinctVerifyingKey<G1Affine>,
    // the input snarks for the aggregation circuit
    // it is padded already so it will have a fixed length of MAX_AGG_SNARKS
    pub snarks: Vec<SnarkWitness>,
    // the public instance for this circuit consists of
    // - an accumulator (12 elements)
    // - the batch's public_input_hash (32 elements)
    // - the number of snarks that is aggregated (1 element)
    pub flattened_instances: Vec<Fr>,
    // accumulation scheme proof, private input
    pub as_proof: Value<Vec<u8>>,
    pub has_prev_accumulator: bool,
    pub final_hash: Fr,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct ConfigParams {
    pub strategy: FpStrategy,
    pub degree: u32,
    pub num_advice: Vec<usize>,
    pub num_lookup_advice: Vec<usize>,
    pub num_fixed: usize,
    pub lookup_bits: usize,
    pub limb_bits: usize,
    pub num_limbs: usize,
}

impl ConfigParams {
    pub(crate) fn aggregation_param() -> Self {
        Self {
            strategy: FpStrategy::Simple,
            degree: 21,
            num_advice: vec![10],
            num_lookup_advice: vec![2],
            num_fixed: 1,
            lookup_bits: 20,
            limb_bits: BITS,
            num_limbs: LIMBS,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AggregationConfig {
    pub base_field_config: FpConfig<Fr, Fq>,
    pub instance: Column<Instance>,
}

impl AggregationConfig {
    /// Build a configuration from parameters.
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        params: &ConfigParams,
    ) -> Self {
        assert!(
            params.limb_bits == BITS && params.num_limbs == LIMBS,
            "For now we fix limb_bits = {BITS}, otherwise change code",
        );

        // base field configuration for aggregation circuit
        let base_field_config = FpConfig::configure(
            meta,
            params.strategy.clone(),
            &params.num_advice,
            &params.num_lookup_advice,
            params.num_fixed,
            params.lookup_bits,
            BITS,
            LIMBS,
            modulus::<Fq>(),
            0,
            params.degree as usize,
        );

        let instance = meta.instance_column();
        meta.enable_equality(instance);

        Self {
            base_field_config,
            instance,
        }
    }

    /// Expose the instance column
    pub fn instance_column(&self) -> Column<Instance> {
        self.instance
    }

    /// Range gate configuration
    pub fn range(&self) -> &RangeConfig<Fr> {
        &self.base_field_config.range
    }

    /// Flex gate configuration
    pub fn flex_gate(&self) -> &FlexGateConfig<Fr> {
        &self.base_field_config.range.gate
    }

    /// Ecc gate configuration
    pub fn ecc_chip(&self) -> BaseFieldEccChip {
        EccChip::construct(self.base_field_config.clone())
    }
}

#[derive(Clone, Debug)]
pub struct MerkleAggregationConfig {
    pub aggregation: AggregationConfig,
    pub poseidon_hash: Pow5Config<Fr, 3, 2>,
}

pub(crate) fn extract_accumulators_and_proof(
    params: &ParamsKZG<Bn256>,
    snarks: &[Snark],
    rng: impl Rng + Send,
) -> Result<(KzgAccumulator<G1Affine, NativeLoader>, Vec<u8>), Error> {
    let svk = params.get_g()[0].into();

    let mut transcript_read =
        PoseidonTranscript::<NativeLoader, &[u8]>::from_spec(&[], POSEIDON_SPEC.clone());
    let accumulators = snarks
        .iter()
        .flat_map(|snark| {
            transcript_read.new_stream(snark.proof.as_slice());
            let proof = Shplonk::read_proof(
                &svk,
                &snark.protocol,
                &snark.instances,
                &mut transcript_read,
            );
            // each accumulator has (lhs, rhs) based on Shplonk
            // lhs and rhs are EC points
            Shplonk::succinct_verify(&svk, &snark.protocol, &snark.instances, &proof)
        })
        .collect::<Vec<_>>();

    let mut transcript_write =
        PoseidonTranscript::<NativeLoader, Vec<u8>>::from_spec(vec![], POSEIDON_SPEC.clone());
    // We always use SHPLONK for accumulation scheme when aggregating proofs
    let accumulator =
        // core step
        // KzgAs does KZG accumulation scheme based on given accumulators and random number (for adding blinding)
        // accumulated ec_pt = ec_pt_1 * 1 + ec_pt_2 * r + ... + ec_pt_n * r^{n-1}
        // ec_pt can be lhs and rhs
        // r is the challenge squeezed from proof
        KzgAs::<Kzg<Bn256, Bdfg21>>::create_proof::<PoseidonTranscript<NativeLoader, Vec<u8>>, _>(
            &Default::default(),
            &accumulators,
            &mut transcript_write,
            rng,
        )?;
    Ok((accumulator, transcript_write.finalize()))
}

impl MerkleAggregationCircuit {
    pub fn new(
        params: &ParamsKZG<Bn256>,
        snarks: Vec<Snark>,
        has_prev_accumulator: bool,
        rng: &mut (impl Rng + Send),
    ) -> Self {
        let svk = params.get_g()[0].into();
        // this aggregates MULTIPLE snarks
        //  (instead of ONE as in proof compression)
        let (accumulator, as_proof) =
            extract_accumulators_and_proof(params, &snarks, rng).unwrap();
        let KzgAccumulator::<G1Affine, NativeLoader> { lhs, rhs } = accumulator;
        let acc_instances = [lhs.x, lhs.y, rhs.x, rhs.y]
            .map(fe_to_limbs::<Fq, Fr, LIMBS, 88>)
            .concat();

        let hash_1 = snarks[0].instances[0][12];
        let hash_2 = snarks[1].instances[0][12];

        let final_hash = Fr::hasher().hash([hash_1, hash_2]);

        let flattened_instances: Vec<Fr> = [
            acc_instances.as_slice(),
            &[final_hash],
        ]
        .concat();

        Self {
            svk,
            flattened_instances,
            snarks: snarks.iter().cloned().map_into().collect(),
            as_proof: Value::known(as_proof),
            has_prev_accumulator,
            final_hash,
        }
    }
}

impl CircuitExt<Fr> for MerkleAggregationCircuit {
    fn num_instance(&self) -> Vec<usize> {
        vec![(4 * LIMBS) + 1]
    }

    fn instances(&self) -> Vec<Vec<Fr>> {
        vec![self.flattened_instances.clone()]
    }

    fn accumulator_indices() -> Option<Vec<(usize, usize)>> {
        Some((0..4 * LIMBS).map(|idx| (0, idx)).collect())
    }

    fn selectors(config: &Self::Config) -> Vec<Selector> {
        config.aggregation.base_field_config.range.gate.basic_gates[0]
            .iter()
            .map(|gate| gate.q_enable)
            .into_iter()
            .chain(config.poseidon_hash.selectors().into_iter()
            )
            .collect()
    }
}

impl Circuit<Fr> for MerkleAggregationCircuit {
    type Config = MerkleAggregationConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        unimplemented!()
    }

    fn configure(meta: &mut plonk::ConstraintSystem<Fr>) -> Self::Config {
        let poseidon_hash = {
            // let hash_advice: Vec<Column<Advice>> = {
            //     let advice = &aggregation.base_field_config.range.gate.basic_gates[0];
            //     advice.into_iter().map(|basic_gate| basic_gate.value).collect_vec()
            // };
            let hash_advice = (0..4).map(|_| {let col = meta.advice_column(); meta.enable_equality(col); col}).collect_vec();
            let state = hash_advice[0..3].to_vec();
            let partial_sbox = hash_advice[3];

            // let constants = &aggregation.base_field_config.range.gate.constants;

            // let rc_a = constants[3..6].to_vec();
            // let rc_b = constants[6..9].to_vec();
    
            let rc_a = (0..3).map(|_| meta.fixed_column()).collect::<Vec<_>>();
            let rc_b = (0..3).map(|_| meta.fixed_column()).collect::<Vec<_>>();
    
            meta.enable_constant(rc_b[0]);
    
            Pow5Chip::configure::<<Fr as Hashable>::SpecType>(
                meta,
                state.try_into().unwrap(),
                partial_sbox,
                rc_a.try_into().unwrap(),
                rc_b.try_into().unwrap(),
            )
    
        };

        let params = env::var("VERIFY_CONFIG").map_or_else(
            |_| ConfigParams::aggregation_param(),
            |path| {
                serde_json::from_reader(
                    File::open(path.as_str()).unwrap_or_else(|_| panic!("{path:?} does not exist")),
                )
                .unwrap()
            },
        );

        let aggregation = AggregationConfig::configure(meta, &params);
        MerkleAggregationConfig {
            aggregation,
            poseidon_hash,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), plonk::Error> {
        let MerkleAggregationConfig { aggregation: config, poseidon_hash } = config;
        #[cfg(feature = "display")]
        let witness_time = start_timer!(|| { "synthesize | EVM verifier" });
        let (accumulator_instances, snark_inputs, hashing) = {
            config
                .range()
                .load_lookup_table(&mut layouter)
                .expect("load range lookup table");
            let mut first_pass = false;
            let (accumulator_instances, snark_inputs, hashing) = layouter.assign_region(
                || "aggregation",
                |region| -> Result<(Vec<AssignedValue<Fr>>, Vec<AssignedValue<Fr>>, _), _> {
                    if first_pass {
                        first_pass = false;
                        return Ok((vec![], vec![], None));
                    }

                    // stores accumulators for all snarks, including the padded ones
                    let mut accumulator_instances: Vec<AssignedValue<Fr>> = vec![];
                    // stores public inputs for all snarks, including the padded ones
                    let mut snark_inputs: Vec<AssignedValue<Fr>> = vec![];
                    let ctx = Context::new(
                        region,
                        ContextParams {
                            max_rows: config.flex_gate().max_rows,
                            num_context_ids: 1,
                            fixed_columns: config.flex_gate().constants.clone(),
                        },
                    );

                    let ecc_chip = config.ecc_chip();
                    let loader = Halo2Loader::new(ecc_chip, ctx);

                    //
                    // extract the assigned values for
                    // - instances which are the public inputs of each chunk (prefixed with 12
                    //   instances from previous accumulators)
                    // - new accumulator to be verified on chain
                    //
                    let (assigned_aggregation_instances, acc) = aggregate::<Kzg<Bn256, Bdfg21>>(
                        &self.svk,
                        &loader,
                        &self.snarks,
                        self.as_proof.as_ref().map(Vec::as_slice)
                    );

                    // extract the following cells for later constraints
                    // - the accumulators
                    // - the public inputs from each snark
                    accumulator_instances.extend(flatten_accumulator(acc).iter().copied());
                    // the snark is not a fresh one, assigned_instances already contains an
                    // accumulator so we want to skip the first 12 elements from the public input
                    snark_inputs.extend(
                        assigned_aggregation_instances
                            .iter()
                            .flat_map(|instance_column| instance_column.iter().skip(0)),
                    );

                    config.range().finalize(&mut loader.ctx_mut());

                    loader.ctx_mut().print_stats(&["Range"]);

                    let hashing = {
                        let first =
                            assigned_aggregation_instances[0][12].clone();
                        let second =
                            assigned_aggregation_instances[1][12].clone();

                        (first, second)
                    };


                    Ok((accumulator_instances, snark_inputs, Some(hashing)))
                },
            ).unwrap();
            (accumulator_instances, snark_inputs, hashing)
        };


        // Expose instances
        if let Some((first, second)) = hashing {
            let poseidon_chip = Pow5Chip::construct(poseidon_hash.clone());
            let mut poseidon_chip: Sponge<
            Fr,
            _,
            <Fr as Hashable>::SpecType,
            _,
            ConstantLengthIden3<2>,
            3,
            2,
                > = Sponge::new(poseidon_chip, layouter.namespace(|| "Poseidon Sponge"))?;


            let first = AssignedCell::<Fr, Fr> {
                value: first.value().cloned(),
                cell: first.cell.clone(),
                _marker: std::marker::PhantomData,
            };

            let second = AssignedCell::<Fr, Fr> {
                value: second.value().cloned(),
                cell: second.cell.clone(),
                _marker: std::marker::PhantomData,
            };

            poseidon_chip.absorb(layouter.namespace(|| "absorbing first"), PaddedWord::Message(first.clone()))?;
            poseidon_chip.absorb(layouter.namespace(|| "absorbing second"), PaddedWord::Message(second.clone()))?;
            // poseidon_chip(&[first, second]);
            let mut output = poseidon_chip.finish_absorbing(layouter.namespace(|| "squeezing"))?;
            let output = output.squeeze(layouter.namespace(|| "get output hash"))?;

            layouter.constrain_instance(output.cell(), config.instance, 12)?;
        }

        for (i, cell) in accumulator_instances.into_iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, i)?;
        }

        #[cfg(feature = "display")]
        end_timer!(witness_time);
        Ok(())
    }
}

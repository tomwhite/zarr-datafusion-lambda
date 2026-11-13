//! A dispatcher whose combiner merges its workers' partial aggregate rows,
//! so that it returns one row a group rather than one a group from every
//! worker.

use std::mem::take;

use arrow::record_batch::RecordBatch;
use base64::prelude::{Engine, BASE64_STANDARD};
use map_lambda_combiner::{Combiner, Error};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use split::Reduce;
use tokio::runtime::Handle;

/// Merged once the rows waiting outnumber both these and the rows merged so
/// far, so that a small key set isn't merged after every output.
const MIN_MERGE_ROWS: usize = 65_536;

/// Outputs and states are both a fragment's `batches` with the
/// `timings_ms` and `s3_requests` of the workers in them, summed, so a
/// state combines like an output.
pub struct PartialReduce {
    reduce: Reduce,
    runtime: Handle,
    merged: Vec<RecordBatch>,
    waiting: Vec<RecordBatch>,
    timings_ms: Map<String, Value>,
    s3_requests: Map<String, Value>,
    combined: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    /// From `split::encode_reduce`.
    plan: String,
}

/// A worker's response, or a state.
#[derive(Deserialize, Serialize)]
pub struct Partial {
    batches: String,
    #[serde(default)]
    timings_ms: Map<String, Value>,
    #[serde(default)]
    s3_requests: Map<String, Value>,
}

impl Combiner for PartialReduce {
    const NAME: &'static str = split::COMBINER;
    type Partial = Partial;

    fn new(config: Value) -> Result<Self, Error> {
        let config: Config = serde_json::from_value(config)?;
        Ok(Self {
            reduce: Reduce::decode(&config.plan)?,
            // The dispatcher calls `combine` and `into_state` on a blocking
            // thread, which may block on its runtime.
            runtime: Handle::try_current()?,
            merged: vec![],
            waiting: vec![],
            timings_ms: Map::new(),
            s3_requests: Map::new(),
            combined: false,
        })
    }

    fn combine(&mut self, partial: Partial) -> Result<(), Error> {
        self.waiting
            .extend(split::decode_batches(&partial.batches)?);
        add(&mut self.timings_ms, partial.timings_ms)?;
        add(&mut self.s3_requests, partial.s3_requests)?;
        self.combined = true;
        if rows(&self.waiting) > rows(&self.merged).max(MIN_MERGE_ROWS) {
            self.merge()?;
        }
        Ok(())
    }

    fn into_state(mut self) -> Result<Option<Partial>, Error> {
        if !self.combined {
            return Ok(None);
        }
        self.merge()?;
        let batches = split::encode_batches(&self.reduce.schema(), &self.merged)?;
        Ok(Some(Partial {
            batches: BASE64_STANDARD.encode(batches),
            timings_ms: self.timings_ms,
            s3_requests: self.s3_requests,
        }))
    }
}

impl PartialReduce {
    fn merge(&mut self) -> Result<(), Error> {
        if self.waiting.is_empty() {
            return Ok(());
        }
        let states = [take(&mut self.merged), take(&mut self.waiting)].concat();
        self.merged = self.runtime.block_on(self.reduce.run(states))?;
        Ok(())
    }
}

fn rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// `from`'s numbers added to `into`'s, field by field.
fn add(into: &mut Map<String, Value>, from: Map<String, Value>) -> Result<(), Error> {
    for (field, n) in from {
        let sum = match (into.get(&field), n.as_u64(), n.as_f64()) {
            (_, _, None) => return Err(format!("`{field}` isn't a number: {n}").into()),
            (None, _, _) => n,
            (Some(a), Some(b), _) if a.is_u64() => Value::from(a.as_u64().unwrap_or(0) + b),
            (Some(a), _, Some(b)) => Value::from(a.as_f64().unwrap_or(0.0) + b),
        };
        into.insert(field, sum);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::datasource::MemTable;
    use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
    use datafusion::physical_plan::common::collect as collect_stream;
    use datafusion::physical_plan::{collect, ExecutionPlan, ExecutionPlanProperties};
    use datafusion::prelude::{SessionConfig, SessionContext};
    use serde_json::json;
    use tokio::runtime::Runtime;

    use super::*;

    const QUERY: &str = "SELECT k, sum(v) AS s, avg(v) AS a, count(*) AS n, min(v) AS lo \
                         FROM t GROUP BY k";

    /// A query over three partitions, each holding every one of its keys,
    /// run as workers would run its partial aggregate, one a partition.
    struct Workers {
        runtime: Runtime,
        partial: Arc<dyn ExecutionPlan>,
        expected: Vec<RecordBatch>,
        outputs: Vec<Value>,
    }

    impl Workers {
        fn new() -> Self {
            Self::with_keys(4)
        }

        fn with_keys(keys: i64) -> Self {
            let runtime = Runtime::new().unwrap();
            let (partial, expected, outputs) = runtime.block_on(async {
                let ctx = context(keys);
                let plan = ctx
                    .sql(QUERY)
                    .await
                    .unwrap()
                    .create_physical_plan()
                    .await
                    .unwrap();
                let expected = collect(plan.clone(), ctx.task_ctx()).await.unwrap();
                let partial = partial_aggregate(&plan).unwrap();
                let mut outputs = vec![];
                for p in 0..partial.output_partitioning().partition_count() {
                    let stream = partial.execute(p, ctx.task_ctx()).unwrap();
                    let batches = collect_stream(stream).await.unwrap();
                    let encoded = split::encode_batches(&partial.schema(), &batches).unwrap();
                    outputs.push(json!({
                        "batches": BASE64_STANDARD.encode(encoded),
                        "timings_ms": {"execute": 1.5},
                        "s3_requests": {"gets": 2, "lists": 1},
                        "chunks_in_flight": 2,
                    }));
                }
                (partial, expected, outputs)
            });
            Self {
                runtime,
                partial,
                expected,
                outputs,
            }
        }

        /// Output `i`, parsed as the dispatcher parses it.
        fn output(&self, i: usize) -> Partial {
            partial(self.outputs[i].clone())
        }

        fn combiner(&self, config: Value) -> Result<PartialReduce, Error> {
            let _entered = self.runtime.enter();
            PartialReduce::new(config)
        }

        fn config(&self) -> Value {
            json!({"plan": split::encode_reduce(&self.partial).unwrap().unwrap()})
        }

        /// A state's rows finished, as the client finishes them.
        fn finish(&self, state: &Partial) -> (usize, String) {
            let states = split::decode_batches(&state.batches).unwrap();
            let source = self.partial.downcast_ref::<AggregateExec>().unwrap();
            let finished = Arc::new(
                AggregateExec::try_new(
                    AggregateMode::Final,
                    source.group_expr().as_final(),
                    source.aggr_expr().to_vec(),
                    source.filter_expr().to_vec(),
                    MemorySourceConfig::try_new_exec(
                        std::slice::from_ref(&states),
                        self.partial.schema(),
                        None,
                    )
                    .unwrap(),
                    source.input_schema(),
                )
                .unwrap(),
            );
            let actual = self
                .runtime
                .block_on(collect(finished, SessionContext::new().task_ctx()))
                .unwrap();
            (rows(&states), sorted_rows(&actual))
        }
    }

    fn context(keys: i64) -> SessionContext {
        let n = keys * 3;
        let partitions = (0..3)
            .map(|p| {
                let k = Int64Array::from_iter_values((0..n).map(|i| i % keys));
                let v = Float64Array::from_iter_values((0..n).map(|i| (p * n + i) as f64));
                vec![
                    RecordBatch::try_from_iter([("k", Arc::new(k) as _), ("v", Arc::new(v) as _)])
                        .unwrap(),
                ]
            })
            .collect::<Vec<_>>();
        let table = MemTable::try_new(partitions[0][0].schema(), partitions).unwrap();
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(3));
        ctx.register_table("t", Arc::new(table)).unwrap();
        ctx
    }

    fn partial_aggregate(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        if plan
            .downcast_ref::<AggregateExec>()
            .is_some_and(|a| *a.mode() == AggregateMode::Partial)
        {
            return Some(plan.clone());
        }
        plan.children().into_iter().find_map(partial_aggregate)
    }

    /// The rows alone, sorted: the query's projection renames the columns.
    fn sorted_rows(batches: &[RecordBatch]) -> String {
        let mut lines = arrow::util::pretty::pretty_format_batches(batches)
            .unwrap()
            .to_string()
            .lines()
            .skip(3)
            .filter(|line| !line.starts_with('+'))
            .map(|line| line.split('|').map(str::trim).collect::<Vec<_>>().join("|"))
            .collect::<Vec<_>>();
        lines.sort();
        lines.join("\n")
    }

    fn partial(value: Value) -> Partial {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn outputs_combine_to_one_row_a_group_with_the_same_answer() {
        let workers = Workers::new();
        let mut combiner = workers.combiner(workers.config()).unwrap();

        for i in 0..workers.outputs.len() {
            combiner.combine(workers.output(i)).unwrap();
        }
        let state = combiner.into_state().unwrap().unwrap();

        let (rows, actual) = workers.finish(&state);
        assert_eq!(rows, 4);
        assert_eq!(actual, sorted_rows(&workers.expected));
    }

    #[test]
    fn a_state_combines_like_an_output() {
        let workers = Workers::new();
        let mut first = workers.combiner(workers.config()).unwrap();
        let mut second = workers.combiner(workers.config()).unwrap();

        first.combine(workers.output(0)).unwrap();
        first.combine(workers.output(1)).unwrap();
        second
            .combine(first.into_state().unwrap().unwrap())
            .unwrap();
        second.combine(workers.output(2)).unwrap();
        let state = second.into_state().unwrap().unwrap();

        let (rows, actual) = workers.finish(&state);
        assert_eq!(rows, 4);
        assert_eq!(actual, sorted_rows(&workers.expected));
        assert_eq!(json!(state.s3_requests), json!({"gets": 6, "lists": 3}));
        assert_eq!(json!(state.timings_ms), json!({"execute": 4.5}));
    }

    #[test]
    fn outputs_merge_once_more_rows_wait_than_have_merged() {
        // two outputs' rows pass the minimum, and a third's don't pass the two merged
        let keys = MIN_MERGE_ROWS * 5 / 8;
        let workers = Workers::with_keys(keys as i64);
        let mut combiner = workers.combiner(workers.config()).unwrap();
        let held = |c: &PartialReduce| (rows(&c.merged), rows(&c.waiting));

        combiner.combine(workers.output(0)).unwrap();
        assert_eq!(held(&combiner), (0, keys));
        combiner.combine(workers.output(1)).unwrap();
        assert_eq!(held(&combiner), (keys, 0));
        combiner.combine(workers.output(2)).unwrap();
        assert_eq!(held(&combiner), (keys, keys));
        let state = combiner.into_state().unwrap().unwrap();

        let (rows, actual) = workers.finish(&state);
        assert_eq!(rows, keys);
        assert_eq!(actual, sorted_rows(&workers.expected));
    }

    #[test]
    fn nothing_combined_gives_no_state() {
        let workers = Workers::new();
        let combiner = workers.combiner(workers.config()).unwrap();

        assert!(combiner.into_state().unwrap().is_none());
    }

    #[test]
    fn a_config_without_a_reduce_is_refused() {
        let workers = Workers::new();

        assert!(workers.combiner(json!({})).is_err());
        assert!(workers.combiner(json!({"plan": "AAAA"})).is_err());
        let mut unknown = workers.config();
        unknown["flush_bytes"] = json!(1);
        assert!(workers.combiner(unknown).is_err());
    }

    #[test]
    fn an_output_without_batches_is_refused() {
        let workers = Workers::new();
        let mut combiner = workers.combiner(workers.config()).unwrap();

        assert!(serde_json::from_value::<Partial>(json!({"timings_ms": {}})).is_err());
        let output = partial(json!({"batches": "", "s3_requests": {"gets": "two"}}));
        assert!(combiner.combine(output).is_err());
    }
}

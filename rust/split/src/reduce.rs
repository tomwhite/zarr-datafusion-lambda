//! Merging workers' partial rows away from the workers, as a combining
//! dispatcher does: the `PartialReduce` a worker runs over its own chunks,
//! run over many workers' states.

use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::SessionContext;

use crate::{decode_plan, encode_plan, partial_reduce, session};

/// The name of the combiner that runs a [`Reduce`], which a combining call
/// asks for.
pub const COMBINER: &str = "partial-reduce";

/// The merge of `fragment`'s states, encoded for [`Reduce::decode`], if the
/// fragment ends in a partial aggregate whose states merge to its own schema.
pub fn encode_reduce(fragment: &Arc<dyn ExecutionPlan>) -> Result<Option<String>> {
    let Some(partial) = fragment
        .downcast_ref::<AggregateExec>()
        .filter(|a| *a.mode() == AggregateMode::Partial)
    else {
        return Ok(None);
    };
    // stands in for the states, which the reduce is given as it runs
    let states = Arc::new(EmptyExec::new(partial.schema()));
    partial_reduce(partial, states)?
        .map(encode_plan)
        .transpose()
}

/// A fragment's states merged to one row a group, in the fragment's schema,
/// so that what it returns can be merged again.
pub struct Reduce {
    plan: Arc<dyn ExecutionPlan>,
    ctx: SessionContext,
}

impl Reduce {
    /// From [`encode_reduce`]'s plan.
    pub fn decode(plan: &str) -> Result<Self> {
        let ctx = session(1);
        let plan = decode_plan(&ctx, plan)?;
        let reduces = plan
            .downcast_ref::<AggregateExec>()
            .is_some_and(|a| *a.mode() == AggregateMode::PartialReduce);
        if !reduces {
            return Err(DataFusionError::Plan(
                "can't merge states with a plan that isn't a PartialReduce".into(),
            ));
        }
        Ok(Self { plan, ctx })
    }

    /// The states' schema, in and out.
    pub fn schema(&self) -> SchemaRef {
        self.plan.schema()
    }

    pub async fn run(&self, states: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
        // An aggregate without groups would answer nothing with a row.
        if states.iter().all(|batch| batch.num_rows() == 0) {
            return Ok(vec![]);
        }
        let states = MemorySourceConfig::try_new_exec(&[states], self.schema(), None)?;
        let plan = self.plan.clone().with_new_children(vec![states])?;
        collect(plan, self.ctx.task_ctx()).await
    }
}

#[cfg(test)]
mod tests {
    use datafusion::physical_plan::common::collect as collect_stream;
    use datafusion::physical_plan::ExecutionPlanProperties;

    use super::*;
    use crate::merge::tests::{context, partial_aggregate, sorted_rows};

    /// `query`'s answer, and the partial rows of each of its partial
    /// aggregate's partitions, as workers each running one would return
    /// them, with the partial aggregate.
    async fn workers(
        query: &str,
    ) -> (
        Vec<RecordBatch>,
        Vec<Vec<RecordBatch>>,
        Arc<dyn ExecutionPlan>,
    ) {
        let ctx = context().await;
        let plan = ctx
            .sql(query)
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
            outputs.push(collect_stream(stream).await.unwrap());
        }
        (expected, outputs, partial)
    }

    /// `states` finished, as the client does.
    async fn finish(
        partial: &Arc<dyn ExecutionPlan>,
        states: Vec<RecordBatch>,
    ) -> Vec<RecordBatch> {
        let source = partial.downcast_ref::<AggregateExec>().unwrap();
        let states = MemorySourceConfig::try_new_exec(&[states], partial.schema(), None).unwrap();
        let finished = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Final,
                source.group_expr().as_final(),
                source.aggr_expr().to_vec(),
                source.filter_expr().to_vec(),
                states,
                source.input_schema(),
            )
            .unwrap(),
        );
        collect(finished, SessionContext::new().task_ctx())
            .await
            .unwrap()
    }

    fn rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(RecordBatch::num_rows).sum()
    }

    const GROUPED: &str = "SELECT k, sum(v) AS s, avg(v) AS a, count(*) AS n, min(v) AS lo, \
                           count(*) FILTER (WHERE v > 20) AS big FROM t GROUP BY k";

    #[tokio::test]
    async fn every_workers_states_merge_to_one_row_a_group_with_the_same_answer() {
        let (expected, outputs, partial) = workers(GROUPED).await;
        let reduce = Reduce::decode(&encode_reduce(&partial).unwrap().unwrap()).unwrap();

        let merged = reduce.run(outputs.concat()).await.unwrap();

        assert_eq!(rows(&merged), 4);
        let actual = finish(&partial, merged).await;
        assert_eq!(sorted_rows(&actual), sorted_rows(&expected));
    }

    #[tokio::test]
    async fn merged_states_merge_again_with_the_same_answer() {
        let (expected, outputs, partial) = workers(GROUPED).await;
        let reduce = Reduce::decode(&encode_reduce(&partial).unwrap().unwrap()).unwrap();

        let first = reduce.run(outputs[..2].concat()).await.unwrap();
        let second = reduce.run(outputs[2].clone()).await.unwrap();
        let merged = reduce.run([first, second].concat()).await.unwrap();

        assert_eq!(rows(&merged), 4);
        let actual = finish(&partial, merged).await;
        assert_eq!(sorted_rows(&actual), sorted_rows(&expected));
    }

    #[tokio::test]
    async fn states_without_groups_merge_to_one_row() {
        let (expected, outputs, partial) =
            workers("SELECT sum(v) AS s, stddev(v) AS sd, max(v) AS hi FROM t").await;
        let reduce = Reduce::decode(&encode_reduce(&partial).unwrap().unwrap()).unwrap();

        let merged = reduce.run(outputs.concat()).await.unwrap();

        assert_eq!(rows(&merged), 1);
        let actual = finish(&partial, merged).await;
        assert_eq!(sorted_rows(&actual), sorted_rows(&expected));
    }

    #[tokio::test]
    async fn no_states_merge_to_nothing() {
        let (_, _, partial) = workers("SELECT sum(v) AS s FROM t").await;
        let reduce = Reduce::decode(&encode_reduce(&partial).unwrap().unwrap()).unwrap();

        assert!(reduce.run(vec![]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn only_a_partial_aggregate_has_a_reduce() {
        let (_, _, partial) = workers(GROUPED).await;
        let below = partial.children()[0].clone();

        assert!(encode_reduce(&below).unwrap().is_none());
        assert!(Reduce::decode(&encode_plan(below).unwrap()).is_err());
    }
}

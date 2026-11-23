//! Fanning a query out on map-lambda. [`FanOutRule`] cuts a physical plan at
//! each partial aggregate over a Zarr scan, and [`MapLambdaZarrExec`] runs what it
//! cut off on `fragment` lambdas, a run of chunks each, streaming the partial
//! states back to the rest of the plan as workers finish. When no group's rows
//! span two invocations, the final aggregate runs on the workers too. A plan
//! that would leave a Zarr scan or write in this process is refused, unless
//! `lambda.enabled` is off, which runs all of it here.

pub mod region;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use datafusion::arrow::array::{AsArray, UInt64Array};
use datafusion::arrow::datatypes::UInt64Type;
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{extensions_options, JoinType, ScalarValue};
use datafusion::config::{ConfigExtension, ConfigOptions};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::sink::DataSinkExec;
use datafusion::datasource::source::DataSourceExec;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, CastExpr, Column, Literal, TryCastExpr};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::{
    CrossJoinExec, HashJoinExec, NestedLoopJoinExec, PartitionMode,
};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    collect, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::prelude::SessionContext;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use map_lambda_client::{connect, Batch, Combine, CombineBatch, Invoker, Session};
use serde::Serialize;
use serde_json::value::RawValue;
use serde_json::{json, Map, Value};
use split::PartitionSpec;
use zarr_datafusion::optimizer::CardinalityRule;
use zarr_datafusion::physical_plan::partition::split_selection;
use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;
use zarr_datafusion::reader::filter::CoordSelection;
use zarr_datafusion::reader::schema_inference::{ZarrArrayMeta, ZarrStoreMeta};
use zarr_datafusion::writer::{create_skeleton, SkeletonSpec, ZarrDataSink};

/// Where fragments run, and how many at once.
#[derive(Debug, Clone)]
pub struct Settings {
    /// The project the lambdas were deployed as, which prefixes their names on AWS.
    pub project: Option<String>,
    pub worker_function: String,
    /// Worker invocations in flight.
    pub concurrency: usize,
    /// How many dispatchers share them, or the fewest that hold them. Each
    /// streams its workers' results back at 2 MB/s after its first 6 MB.
    pub dispatchers: Option<usize>,
    pub retries: usize,
    /// A local `cargo lambda watch`, for dispatchers and workers both.
    pub endpoint_url: Option<String>,
    pub chunks_per_invocation: usize,
    /// How many of an invocation's chunks a worker runs at once, else its default.
    pub chunks_in_flight: Option<usize>,
    /// Keep every invocation's record and what its worker measured in the [`Report`].
    pub return_stats: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            project: None,
            worker_function: "fragment".into(),
            concurrency: 1000,
            dispatchers: None,
            retries: 2,
            endpoint_url: None,
            chunks_per_invocation: 1,
            chunks_in_flight: None,
            return_stats: false,
        }
    }
}

/// Called with how many of a fan-out's invocations have succeeded, and how
/// many it has: with none before it invokes, then as they arrive.
pub type OnProgress = Arc<dyn Fn(usize, usize) + Send + Sync>;

extensions_options! {
    /// The `lambda.` settings, which SQL's `SET` changes for a session.
    pub struct LambdaOptions {
        /// Whether a plan that can be cut fans out: `SET lambda.enabled = false`
        /// runs every plan in this process, as plain zarr-datafusion would.
        pub enabled: bool, default = true
        /// The most a join's table read from files may take in memory, decoded,
        /// for every worker to read it whole and join its chunks with it: well
        /// under a worker's memory, which holds its chunks in flight too.
        // TODO: measure against a worker's memory.
        pub broadcast_bytes: usize, default = 400 * 1024 * 1024
    }
}

impl ConfigExtension for LambdaOptions {
    const PREFIX: &'static str = "lambda";
}

/// A map-lambda session and the settings its fan-outs run with.
pub struct FanOut {
    session: Session,
    settings: Settings,
    on_progress: Option<OnProgress>,
}

impl fmt::Debug for FanOut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FanOut")
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl FanOut {
    pub fn new(invoker: Arc<dyn Invoker>, settings: Settings) -> Self {
        let mut calls = map_lambda_client::Settings::default();
        calls.project = settings.project.clone();
        calls.dispatchers = settings.dispatchers;
        calls.concurrency = settings.concurrency;
        calls.retries = settings.retries;
        calls.return_stats = settings.return_stats;
        let session = Session::new(invoker, settings.endpoint_url.clone()).with_settings(calls);
        Self {
            session,
            settings,
            on_progress: None,
        }
    }

    pub fn with_progress(mut self, on_progress: OnProgress) -> Self {
        self.on_progress = Some(on_progress);
        self
    }

    /// Against real Lambda, or the emulator at `settings.endpoint_url`.
    pub async fn connect(settings: Settings) -> Self {
        let invoker = connect(settings.endpoint_url.as_deref()).await;
        Self::new(invoker, settings)
    }
}

/// A session over Zarr tables that fans out through `fan_out`, planning what
/// runs here for `target_partitions`, or by default for this machine's cores.
pub fn session(fan_out: Arc<FanOut>, target_partitions: Option<usize>) -> SessionContext {
    let ctx = split::session_from(
        split::builder()
            .with_physical_optimizer_rule(Arc::new(FanOutRule(fan_out)))
            // After the cut, so it only pushes down aggregates that stay here.
            .with_physical_optimizer_rule(Arc::new(CardinalityRule::new()))
            .with_physical_optimizer_rule(Arc::new(split::PerValueRule)),
        target_partitions,
    );
    let state = ctx.state_ref();
    let mut state = state.write();
    let options = state.config_mut().options_mut();
    options.extensions.insert(LambdaOptions::default());
    // Collect a hash join's build side whenever a worker could hold it, so that
    // the cut can broadcast it: its own cap on the build side's files decides.
    options.optimizer.hash_join_single_partition_threshold = 1024 * 1024 * 1024;
    options.optimizer.hash_join_single_partition_threshold_rows = usize::MAX;
    drop(state);
    ctx
}

/// A table's location, and its options: zarr-datafusion's, such as the `group`
/// to read in an icechunk repo, or a `format` other than Zarr, such as `PARQUET`.
#[derive(Clone, Debug, Default)]
pub struct Table {
    pub location: String,
    pub options: BTreeMap<String, String>,
}

impl Table {
    fn create(&self, name: &str) -> String {
        let quoted = |s: &str| format!("'{}'", s.replace('\'', "''"));
        let mut options = self.options.clone();
        let format = options
            .remove("format")
            .unwrap_or_else(|| "ZARR".to_string());
        let mut sql = format!(
            "CREATE EXTERNAL TABLE \"{}\" STORED AS {} LOCATION {}",
            name.replace('"', "\"\""),
            format,
            quoted(&self.location)
        );
        if !options.is_empty() {
            let options = options
                .iter()
                .map(|(key, value)| format!("{} {}", quoted(key), quoted(value)))
                .collect::<Vec<_>>();
            sql += &format!(" OPTIONS ({})", options.join(", "));
        }
        sql
    }
}

/// Register each table by name.
pub async fn register(ctx: &SessionContext, tables: &HashMap<String, Table>) -> Result<()> {
    for (name, table) in tables {
        ctx.sql(&table.create(name)).await?;
    }
    Ok(())
}

/// Run `query`: its result, and a report from each fan-out in its plan.
pub async fn run(
    ctx: &SessionContext,
    query: &str,
) -> Result<(SchemaRef, Vec<RecordBatch>, Vec<Report>)> {
    let plan = ctx.sql(query).await?.create_physical_plan().await?;
    let batches = collect(plan.clone(), ctx.task_ctx()).await?;
    let mut reports = vec![];
    for_each_fan_out(&plan, |exec| reports.push(exec.report()))?;
    Ok((plan.schema(), batches, reports))
}

/// Call `f` on each fan-out in `plan`.
pub fn for_each_fan_out(
    plan: &Arc<dyn ExecutionPlan>,
    mut f: impl FnMut(&MapLambdaZarrExec),
) -> Result<()> {
    plan.apply(|node| {
        if let Some(exec) = node.downcast_ref::<MapLambdaZarrExec>() {
            f(exec);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(())
}

/// Replaces each partial aggregate over a chunkwise Zarr scan with a
/// [`MapLambdaZarrExec`] that runs it on workers.
#[derive(Debug)]
pub struct FanOutRule(pub Arc<FanOut>);

impl PhysicalOptimizerRule for FanOutRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let options = config.extensions.get::<LambdaOptions>();
        if !options.is_none_or(|options| options.enabled) {
            return Ok(plan);
        }
        let broadcast_bytes = options.map_or(LambdaOptions::default().broadcast_bytes, |options| {
            options.broadcast_bytes
        });
        let plan = plan
            .transform_up(|node| {
                let exec = match self.cut(&node, broadcast_bytes)? {
                    Some(exec) => Some(exec),
                    None => match finish(&node)? {
                        Some(exec) => Some(exec),
                        None => write(&node, &self.0, broadcast_bytes)?,
                    },
                };
                Ok(match exec {
                    Some(exec) => Transformed::yes(exec),
                    None => Transformed::no(node),
                })
            })?
            .data;
        // Refused rather than run here, where it would work but slowly.
        if let Some(why) = left_here(&plan, &mut vec![]) {
            return Err(DataFusionError::Plan(format!(
                "this query can't run on the workers: {why}. \
                 SET lambda.enabled = false to run all of it here"
            )));
        }
        Ok(plan)
    }

    fn name(&self) -> &str {
        "fan_out"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

impl FanOutRule {
    fn cut(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        broadcast_bytes: usize,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(partial) = node
            .downcast_ref::<AggregateExec>()
            .filter(|a| *a.mode() == AggregateMode::Partial)
        else {
            return Ok(None);
        };
        let Some(probe) = chunkwise_scan(node, broadcast_bytes) else {
            return Ok(None);
        };
        let scan = probe.downcast_ref::<ZarrExec>().expect("a Zarr scan");
        let Some(chunks) = chunks(scan) else {
            return Ok(None);
        };
        let per_invocation = self.0.settings.chunks_per_invocation.max(1);
        let (runs, whole_groups) = if groups_within_tiles(partial, scan, &chunks) {
            by_tile(chunks, per_invocation)
        } else {
            let runs = chunks.chunks(per_invocation).map(<[_]>::to_vec).collect();
            (runs, groups_within_chunks(partial, scan, &chunks))
        };
        let mut exec = MapLambdaZarrExec::new(
            collect_left(without_round_robin(node.clone())?)?,
            runs,
            whole_groups,
            self.0.clone(),
        )?;
        exec.joined = joined_scans(node, probe, broadcast_bytes);
        exec.reduce = split::encode_reduce(&exec.plan)?;
        Ok(Some(Arc::new(exec)))
    }
}

/// A final aggregate over a fan-out whose invocations each hold all of their
/// groups' rows, moved onto the workers.
fn finish(node: &Arc<dyn ExecutionPlan>) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(last) = node.downcast_ref::<AggregateExec>().filter(|a| {
        matches!(
            a.mode(),
            AggregateMode::Final | AggregateMode::FinalPartitioned
        )
    }) else {
        return Ok(None);
    };
    let mut input = last.input();
    while input.is::<RepartitionExec>() || input.is::<CoalescePartitionsExec>() {
        input = input.children()[0];
    }
    let Some(fan_out) = input
        .downcast_ref::<MapLambdaZarrExec>()
        .filter(|f| f.whole_groups)
    else {
        return Ok(None);
    };
    let plan = Arc::new(AggregateExec::try_new(
        AggregateMode::Final,
        last.group_expr().clone(),
        last.aggr_expr().to_vec(),
        last.filter_expr().to_vec(),
        fan_out.plan.clone(),
        last.input_schema(),
    )?);
    let mut finished =
        MapLambdaZarrExec::new(plan, fan_out.runs.clone(), false, fan_out.fan_out.clone())?;
    finished.finished = true;
    finished.joined = fan_out.joined.clone();
    Ok(Some(Arc::new(finished)))
}

/// A Zarr sink over a fan-out of finished rows, or over a Zarr scan's rows
/// with no aggregate, moved onto the workers, each writing the chunks its
/// tiles fall in.
fn write(
    node: &Arc<dyn ExecutionPlan>,
    lambdas: &Arc<FanOut>,
    broadcast_bytes: usize,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(sink) = node.downcast_ref::<DataSinkExec>() else {
        return Ok(None);
    };
    let mut input = sink.input();
    loop {
        let rowwise = input.is::<ProjectionExec>()
            || input.is::<FilterExec>()
            || input.is::<SortExec>()
            || input.is::<SortPreservingMergeExec>()
            || input.is::<RepartitionExec>()
            || input.is::<CoalescePartitionsExec>()
            || input.is::<CooperativeExec>();
        if let Some(probe) = broadcast_probe(input, broadcast_bytes) {
            input = probe;
            continue;
        }
        if !rowwise {
            break;
        }
        input = input.children()[0];
    }
    let Some(zarr) = sink.sink().downcast_ref::<ZarrDataSink>() else {
        return Ok(None);
    };
    let rows;
    let fan_out = match (
        input.downcast_ref::<MapLambdaZarrExec>(),
        input.downcast_ref::<ZarrExec>().and_then(chunks),
    ) {
        (Some(fan_out), _) if fan_out.finished => fan_out,
        (None, Some(chunks)) => {
            let per_invocation = lambdas.settings.chunks_per_invocation.max(1);
            let runs = chunks.chunks(per_invocation).map(<[_]>::to_vec).collect();
            rows = MapLambdaZarrExec {
                joined: joined_scans(node, input, broadcast_bytes),
                ..MapLambdaZarrExec::new(sink.input().clone(), runs, true, lambdas.clone())?
            };
            &rows
        }
        _ => return Ok(None),
    };
    let runs = if fan_out.tiles_chunks(zarr.spec()) {
        fan_out.runs.clone()
    } else {
        match fan_out
            .runs_by_output_chunk(zarr.spec())
            .or_else(|| fan_out.runs_by_output_box(zarr.spec()))
        {
            Some(runs) => runs,
            None => return Ok(None),
        }
    };
    let fragment = Arc::new(DataSinkExec::new(
        collect_left(without_round_robin(inline(
            sink.input().clone(),
            &fan_out.plan,
        )?)?)?,
        Arc::new(zarr.clone().writing_chunks()),
        sink.sort_order().clone(),
    ));
    Ok(Some(Arc::new(MapLambdaZarrExec {
        joined: fan_out.joined.clone(),
        ..MapLambdaZarrExec::new(fragment, runs, false, fan_out.fan_out.clone())?
    })))
}

/// `plan` with the fan-out in it replaced by the fragment it runs.
fn inline(
    plan: Arc<dyn ExecutionPlan>,
    fragment: &Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if plan.is::<MapLambdaZarrExec>() {
        return Ok(fragment.clone());
    }
    let children = plan
        .children()
        .into_iter()
        .map(|child| inline(child.clone(), fragment))
        .collect::<Result<_>>()?;
    plan.with_new_children(children)
}

/// `fragment` without the round-robin repartitions planned for this process's
/// cores, so each worker runs one partition per chunk it is given.
// TODO: plan for the worker's own vCPUs, by running `EnforceDistribution` over
// the bound fragment there.
fn without_round_robin(fragment: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    fragment
        .transform_up(|node| {
            Ok(match node.downcast_ref::<RepartitionExec>() {
                Some(r) if matches!(r.partitioning(), Partitioning::RoundRobinBatch(_)) => {
                    Transformed::yes(r.input().clone())
                }
                _ => Transformed::no(node),
            })
        })
        .map(|t| t.data)
}

/// The Zarr scan under a partial aggregate, if every node between them works
/// row by row, so that a worker given some chunks computes exactly their share
/// of the partial states. A join counts when every worker can be sent its
/// build side whole.
fn chunkwise_scan(
    partial: &Arc<dyn ExecutionPlan>,
    broadcast_bytes: usize,
) -> Option<&Arc<dyn ExecutionPlan>> {
    let mut node = *partial.children().first()?;
    loop {
        if node.is::<ZarrExec>() {
            return Some(node);
        }
        if let Some(probe) = broadcast_probe(node, broadcast_bytes) {
            node = probe;
            continue;
        }
        let rowwise = node.is::<ProjectionExec>()
            || node.is::<FilterExec>()
            || node.is::<RepartitionExec>()
            || node.is::<CoalescePartitionsExec>()
            || node.is::<CooperativeExec>();
        let [child] = node.children()[..] else {
            return None;
        };
        if !rowwise {
            return None;
        }
        node = child;
    }
}

/// Why some of `plan` would run here rather than on the workers, if any would: a
/// Zarr scan the cut left in place, or a Zarr write over a fan-out. `above` is
/// the path down to `node`.
fn left_here<'a>(
    node: &'a Arc<dyn ExecutionPlan>,
    above: &mut Vec<&'a Arc<dyn ExecutionPlan>>,
) -> Option<String> {
    if let Some(scan) = node.downcast_ref::<ZarrExec>() {
        return Some(scan_left_here(scan, node, above));
    }
    if let Some(sink) = node.downcast_ref::<DataSinkExec>() {
        let fan_out = fan_out_under(sink.input());
        if sink.sink().downcast_ref::<ZarrDataSink>().is_some() && fan_out.is_some() {
            return Some(if fan_out.is_some_and(|f| f.finished) {
                "the Zarr output's chunks don't each lie in one invocation's share of \
                 the source, so two workers would write one"
                    .to_string()
            } else {
                "groups span invocations, so the final aggregate and the Zarr write \
                 would run here"
                    .to_string()
            });
        }
    }
    above.push(node);
    for child in node.children() {
        if let Some(why) = left_here(child, above) {
            return Some(why);
        }
    }
    above.pop();
    None
}

/// The fan-out a Zarr sink writes, through nodes that pass its rows on.
fn fan_out_under(plan: &Arc<dyn ExecutionPlan>) -> Option<&MapLambdaZarrExec> {
    if let Some(fan_out) = plan.downcast_ref::<MapLambdaZarrExec>() {
        return Some(fan_out);
    }
    let [child] = plan.children()[..] else {
        return None;
    };
    fan_out_under(child)
}

/// Why the cut left `scan` here, from the nodes above it.
fn scan_left_here(
    scan: &ZarrExec,
    node: &Arc<dyn ExecutionPlan>,
    above: &[&Arc<dyn ExecutionPlan>],
) -> String {
    if scan.limit().is_some() || scan.partitions().is_empty() {
        return "a Zarr scan can't be split into chunks: it has a limit, or reads its store \
                whole"
            .to_string();
    }
    let path: Vec<_> = above.iter().copied().chain([node]).collect();
    for pair in path.windows(2).rev() {
        let (parent, child) = (pair[0], pair[1]);
        let sides = match parent.downcast_ref::<NestedLoopJoinExec>() {
            Some(join) => Some((join.left(), join.right())),
            None => parent
                .downcast_ref::<HashJoinExec>()
                .map(|join| (join.left(), join.right())),
        };
        if let Some((build, probe)) = sides {
            let other = if Arc::ptr_eq(build, child) {
                probe
            } else {
                build
            };
            return if !split::zarr_scans(other).unwrap_or_default().is_empty() {
                "two Zarr scans are joined, which needs the build side read row by row \
                 and joined on every coordinate the two share"
                    .to_string()
            } else if Arc::ptr_eq(build, child) {
                "a Zarr scan is a join's build side: write the smaller side first".to_string()
            } else if contains_fan_out(other) {
                "a Zarr scan is joined with another fan-out's result, which would need a \
                 sliced broadcast"
                    .to_string()
            } else {
                "a Zarr scan is joined with a side that can't be sent to every worker: that \
                 needs an inner join, and on its build side rows held in memory, at most \
                 1 MB, or files whose rows take at most lambda.broadcast_bytes in memory"
                    .to_string()
            };
        }
        if parent.children().len() > 1 {
            return format!(
                "a Zarr scan is under a {}, which can't be cut by chunk",
                parent.name()
            );
        }
    }
    if !above.iter().any(|n| n.is::<AggregateExec>()) {
        let written = above.iter().any(|n| {
            n.downcast_ref::<DataSinkExec>()
                .is_some_and(|sink| sink.sink().downcast_ref::<ZarrDataSink>().is_some())
        });
        return if written {
            "the Zarr output's chunks don't each lie in one invocation's share of the \
             source, so two workers would write one"
                .to_string()
        } else {
            "a Zarr scan has no aggregate above it, so every row it reads would come back \
             here: COPY them to Zarr, which the workers write"
                .to_string()
        };
    }
    let blocking: Vec<_> = above
        .iter()
        .rev()
        .take_while(|n| !n.is::<AggregateExec>())
        .map(|n| n.name())
        .collect();
    format!(
        "the nodes between a Zarr scan and its aggregate don't work chunk by chunk: {}",
        blocking.join(", ")
    )
}

fn contains_fan_out(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<MapLambdaZarrExec>() || plan.children().into_iter().any(contains_fan_out)
}

/// The most a broadcast build side of rows held here may hold, which travel in
/// the fragment, in every dispatcher's request, under Lambda's 6 MB request
/// limit.
const BROADCAST_ROWS_BYTES: usize = 1024 * 1024;

/// The probe side of an inner join whose build side every worker can be sent
/// whole. An inner join matches each probe row against the whole build side
/// alone, so it works row by row on the probe side; an outer join would repeat
/// its unmatched build rows on every worker.
fn broadcast_probe(
    node: &Arc<dyn ExecutionPlan>,
    broadcast_bytes: usize,
) -> Option<&Arc<dyn ExecutionPlan>> {
    if let Some(join) = node.downcast_ref::<NestedLoopJoinExec>() {
        let sent = broadcastable(join.left(), broadcast_bytes);
        return (*join.join_type() == JoinType::Inner && sent).then_some(join.right());
    }
    if let Some(join) = node.downcast_ref::<CrossJoinExec>() {
        return broadcastable(join.left(), broadcast_bytes).then_some(join.right());
    }
    let join = node.downcast_ref::<HashJoinExec>()?;
    let sent = match join.partition_mode() {
        PartitionMode::CollectLeft => {
            broadcastable(join.left(), broadcast_bytes) || !joined_axes(join).is_empty()
        }
        PartitionMode::Partitioned => !joined_axes(join).is_empty(),
        PartitionMode::Auto => false,
    };
    (*join.join_type() == JoinType::Inner && sent).then_some(join.right())
}

/// A join's build side read from another Zarr scan: each invocation reads the
/// box of it that its own chunks can join with.
#[derive(Debug, Clone)]
struct JoinedScan {
    scan: Arc<dyn ExecutionPlan>,
    axes: Vec<Axis>,
    /// The most an invocation's box of it may take in memory.
    max_bytes: usize,
}

/// A coordinate a join equates between its probe scan and a joined scan.
#[derive(Debug, Clone, PartialEq)]
struct Axis {
    probe: String,
    build: String,
    /// Added to the probe's values to meet the build's, in the decoded unit,
    /// microseconds for a time: one join key for each.
    offsets: Vec<f64>,
}

/// A join key: a Zarr scan's coordinate, cast losslessly if at all, plus any
/// of `offsets`.
struct Key {
    coord: String,
    offsets: Vec<f64>,
}

/// The joins between `partial` and the `probe` scan whose build sides are Zarr
/// scans, in the order they are passed.
fn joined_scans(
    partial: &Arc<dyn ExecutionPlan>,
    probe: &Arc<dyn ExecutionPlan>,
    max_bytes: usize,
) -> Vec<JoinedScan> {
    let mut joined = vec![];
    let mut node = partial;
    while !Arc::ptr_eq(node, probe) {
        if let Some(join) = node.downcast_ref::<HashJoinExec>() {
            let axes = joined_axes(join);
            if let (false, Some(scan)) = (axes.is_empty(), row_source(join.left())) {
                joined.push(JoinedScan {
                    scan: scan.clone(),
                    axes,
                    max_bytes,
                });
            }
            node = join.right();
            continue;
        }
        match node.children()[..] {
            [child] => node = child,
            _ => break,
        }
    }
    joined
}

/// The coordinates an inner hash join equates between a Zarr scan on its build
/// side, read row by row, and one on its probe side; none unless it equates
/// every coordinate the two share, as an invocation would otherwise read all
/// of the build's along the others. A probe's time may be shifted by
/// intervals.
fn joined_axes(join: &HashJoinExec) -> Vec<Axis> {
    let coords = |plan: Option<&Arc<dyn ExecutionPlan>>| -> Vec<ZarrArrayMeta> {
        plan.and_then(|p| p.downcast_ref::<ZarrExec>())
            .and_then(ZarrExec::store_meta)
            .map(|meta| meta.coords.clone())
            .unwrap_or_default()
    };
    let build = coords(row_source(join.left()));
    if build.is_empty() {
        return vec![];
    }
    let probe = coords(probe_scan(join.right()));
    let is_time = |coords: &[ZarrArrayMeta], name: &str| {
        coords.iter().any(|c| {
            c.name == name
                && c.cf_time_attrs
                    .as_ref()
                    .is_some_and(|attrs| attrs.parse().is_ok())
        })
    };
    let axes: Vec<Axis> = join
        .on()
        .iter()
        .filter_map(|(build_key, probe_key)| {
            let build_key = scan_key(join.left(), build_key.downcast_ref::<Column>()?.index())?;
            let probe_key = scan_key(join.right(), probe_key.downcast_ref::<Column>()?.index())?;
            let shifted = probe_key.offsets != [0.0];
            if build_key.offsets != [0.0]
                || shifted
                    && !(is_time(&probe, &probe_key.coord) && is_time(&build, &build_key.coord))
            {
                return None;
            }
            Some(Axis {
                probe: probe_key.coord,
                build: build_key.coord,
                offsets: probe_key.offsets,
            })
        })
        .collect();
    let unjoined = build
        .iter()
        .filter(|c| probe.iter().any(|p| p.name == c.name))
        .any(|c| !axes.iter().any(|a| a.build == c.name));
    if unjoined {
        return vec![];
    }
    axes
}

/// The Zarr scan whose chunks `plan` is dealt by, through its joins' probe
/// sides.
fn probe_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    let mut node = plan;
    loop {
        if node.is::<ZarrExec>() {
            return Some(node);
        }
        if let Some(join) = node.downcast_ref::<HashJoinExec>() {
            node = join.right();
            continue;
        }
        if let Some(join) = node.downcast_ref::<CrossJoinExec>() {
            node = join.right();
            continue;
        }
        match node.children()[..] {
            [child] => node = child,
            _ => return None,
        }
    }
}

/// The Zarr scan `plan` reads row by row, with no aggregate or join between.
fn row_source(plan: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    let mut node = plan;
    loop {
        if node.is::<ZarrExec>() {
            return Some(node);
        }
        let rowwise = node.is::<ProjectionExec>()
            || node.is::<FilterExec>()
            || node.is::<RepartitionExec>()
            || node.is::<CoalescePartitionsExec>()
            || node.is::<CooperativeExec>();
        match node.children()[..] {
            [child] if rowwise => node = child,
            _ => return None,
        }
    }
}

/// The key `plan`'s column `index` is, if a coordinate of the Zarr scan under
/// `plan`.
fn scan_key(plan: &Arc<dyn ExecutionPlan>, mut index: usize) -> Option<Key> {
    let mut node = plan;
    loop {
        if let Some(scan) = node.downcast_ref::<ZarrExec>() {
            let schema = node.schema();
            let name = schema.field(index).name();
            let coords = &scan.store_meta()?.coords;
            return coords.iter().any(|c| c.name == *name).then(|| Key {
                coord: name.clone(),
                offsets: vec![0.0],
            });
        }
        if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
            return expr_key(&projection.expr()[index].expr, projection.input());
        }
        if let Some(join) = node.downcast_ref::<CrossJoinExec>() {
            index = index.checked_sub(join.left().schema().fields().len())?;
            node = join.right();
            continue;
        }
        if let Some(filter) = node.downcast_ref::<FilterExec>() {
            if let Some(projection) = filter.projection() {
                index = projection[index];
            }
        } else if !(node.is::<RepartitionExec>()
            || node.is::<CoalescePartitionsExec>()
            || node.is::<CooperativeExec>())
        {
            return None;
        }
        let [child] = node.children()[..] else {
            return None;
        };
        node = child;
    }
}

/// The key `expr` over `input`'s rows is: a column, a lossless cast of one,
/// or one plus intervals.
fn expr_key(expr: &Arc<dyn PhysicalExpr>, input: &Arc<dyn ExecutionPlan>) -> Option<Key> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        return scan_key(input, column.index());
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let from = cast.expr().data_type(&input.schema()).ok()?;
        return lossless(&from, cast.cast_type()).then(|| expr_key(cast.expr(), input))?;
    }
    let sum = expr
        .downcast_ref::<BinaryExpr>()
        .filter(|b| *b.op() == Operator::Plus)?;
    let key = expr_key(sum.left(), input)?;
    let by = offsets(sum.right(), input)?;
    let mut offsets: Vec<f64> = key
        .offsets
        .iter()
        .flat_map(|a| by.iter().map(move |b| a + b))
        .collect();
    offsets.sort_by(f64::total_cmp);
    offsets.dedup();
    Some(Key { offsets, ..key })
}

fn lossless(from: &DataType, to: &DataType) -> bool {
    let from = match from {
        DataType::Dictionary(_, values) => values,
        other => other,
    };
    // dropping UTC leaves the instant as it was
    let utc = |zone: &Option<Arc<str>>| zone.as_deref().is_none_or(|z| z == "UTC" || z == "+00:00");
    match (from, to) {
        (DataType::Timestamp(_, a), DataType::Timestamp(_, b)) => a == b || utc(a) && utc(b),
        _ => from == to,
    }
}

/// The intervals `expr` gives over `input`'s rows, in microseconds: a
/// literal, or a column of rows held here.
fn offsets(expr: &Arc<dyn PhysicalExpr>, input: &Arc<dyn ExecutionPlan>) -> Option<Vec<f64>> {
    let values = match (
        expr.downcast_ref::<Literal>(),
        expr.downcast_ref::<Column>(),
    ) {
        (Some(literal), _) => vec![literal.value().clone()],
        (_, Some(column)) => held_values(input, column.index())?,
        _ => return None,
    };
    values.iter().map(micros).collect()
}

fn micros(interval: &ScalarValue) -> Option<f64> {
    const DAY: f64 = 86_400e6;
    Some(match interval {
        ScalarValue::IntervalMonthDayNano(Some(v)) if v.months == 0 => {
            v.days as f64 * DAY + v.nanoseconds as f64 / 1e3
        }
        ScalarValue::IntervalDayTime(Some(v)) => v.days as f64 * DAY + v.milliseconds as f64 * 1e3,
        ScalarValue::DurationSecond(Some(v)) => *v as f64 * 1e6,
        ScalarValue::DurationMillisecond(Some(v)) => *v as f64 * 1e3,
        ScalarValue::DurationMicrosecond(Some(v)) => *v as f64,
        ScalarValue::DurationNanosecond(Some(v)) => *v as f64 / 1e3,
        _ => return None,
    })
}

/// The values of `plan`'s column `index`, if it passes them on from rows held
/// here, such as a `VALUES` list's, through the build side of cross joins.
fn held_values(plan: &Arc<dyn ExecutionPlan>, index: usize) -> Option<Vec<ScalarValue>> {
    if let Some(join) = plan.downcast_ref::<CrossJoinExec>() {
        return (index < join.left().schema().fields().len())
            .then(|| held_values(join.left(), index))?;
    }
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        let column = projection.expr()[index].expr.downcast_ref::<Column>()?;
        return held_values(projection.input(), column.index());
    }
    if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        let memory = source.data_source().downcast_ref::<MemorySourceConfig>()?;
        let index = match memory.projection() {
            Some(projection) => projection[index],
            None => index,
        };
        let mut values = vec![];
        for batch in memory.partitions().iter().flatten() {
            let column = batch.column(index);
            for row in 0..column.len() {
                values.push(ScalarValue::try_from_array(column, row).ok()?);
            }
        }
        return Some(values);
    }
    let passes_on = plan.is::<RepartitionExec>()
        || plan.is::<CoalescePartitionsExec>()
        || plan.is::<CooperativeExec>();
    match plan.children()[..] {
        [child] if passes_on => held_values(child, index),
        _ => None,
    }
}

/// `fragment` with each join whose build side is a Zarr scan collecting that
/// side whole, as an invocation's box of it is small, rather than
/// repartitioning both sides by key for this process's cores.
fn collect_left(fragment: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    let unhashed = |plan: &Arc<dyn ExecutionPlan>| match plan.downcast_ref::<RepartitionExec>() {
        Some(r) if matches!(r.partitioning(), Partitioning::Hash(..)) => r.input().clone(),
        _ => plan.clone(),
    };
    fragment
        .transform_up(|node| {
            let Some(join) = node
                .downcast_ref::<HashJoinExec>()
                .filter(|j| *j.partition_mode() == PartitionMode::Partitioned)
                .filter(|j| !joined_axes(j).is_empty())
            else {
                return Ok(Transformed::no(node));
            };
            let build = Arc::new(CoalescePartitionsExec::new(unhashed(join.left())));
            let collected = join
                .builder()
                .with_new_children(vec![build, unhashed(join.right())])?
                .with_partition_mode(PartitionMode::CollectLeft)
                .build_exec()?;
            Ok(Transformed::yes(collected))
        })
        .map(|t| t.data)
}

/// Whether `build` reads only rows held here, few enough to travel in the
/// fragment, or files whose rows take at most `broadcast_bytes` in memory, for
/// every worker to read whole itself.
fn broadcastable(build: &Arc<dyn ExecutionPlan>, broadcast_bytes: usize) -> bool {
    let Some(source) = source_under(build) else {
        return false;
    };
    let data = source.data_source();
    if let Some(memory) = data.downcast_ref::<MemorySourceConfig>() {
        let bytes: usize = memory
            .partitions()
            .iter()
            .flatten()
            .map(RecordBatch::get_array_memory_size)
            .sum();
        return bytes <= BROADCAST_ROWS_BYTES;
    }
    data.is::<FileScanConfig>()
        && decoded_bytes(source).is_some_and(|bytes| bytes <= broadcast_bytes)
}

/// The bytes `source`'s rows take in memory, from its statistics, if they tell:
/// decoded, not as stored, as Parquet can encode repetitive columns, such as
/// regridding weights, dozens of times smaller.
fn decoded_bytes(source: &DataSourceExec) -> Option<usize> {
    let stats = source.partition_statistics(None).ok()?;
    if let Some(&bytes) = stats.total_byte_size.get_value() {
        return Some(bytes);
    }
    // A column the scan computes, such as a join key's cast, has no size of its
    // own, but a fixed width.
    let rows = *stats.num_rows.get_value()?;
    let schema = source.schema();
    stats
        .column_statistics
        .iter()
        .zip(schema.fields())
        .map(|(column, field)| {
            column.byte_size.get_value().copied().or_else(|| {
                let width = match field.data_type() {
                    DataType::Dictionary(key, _) => key.primitive_width(),
                    other => other.primitive_width(),
                };
                Some(rows * width?)
            })
        })
        .sum()
}

/// The source `plan` reads, if it reads only one, through nodes that pass its
/// rows on.
fn source_under(plan: &Arc<dyn ExecutionPlan>) -> Option<&DataSourceExec> {
    if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        return Some(source);
    }
    let passes_on = plan.is::<ProjectionExec>()
        || plan.is::<FilterExec>()
        || plan.is::<RepartitionExec>()
        || plan.is::<CoalescePartitionsExec>()
        || plan.is::<CooperativeExec>();
    let [child] = plan.children()[..] else {
        return None;
    };
    if passes_on {
        source_under(child)
    } else {
        None
    }
}

/// The object stores of the files `plan` reads, as URLs such as `s3://bucket/`.
fn file_stores(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
    let mut stores = vec![];
    let _ = plan.apply(|node| {
        if let Some(files) = node
            .downcast_ref::<DataSourceExec>()
            .and_then(|source| source.data_source().downcast_ref::<FileScanConfig>())
        {
            stores.push(files.object_store_url.as_str().to_string());
        }
        Ok(TreeNodeRecursion::Continue)
    });
    stores
}

/// The scan's partitions cut into single chunks, less any that read nothing.
fn chunks(scan: &ZarrExec) -> Option<Vec<PartitionSpec>> {
    // No partitions means one unsliced read: VirtualiZarr, Icechunk, or a
    // limit, whose pushdown is only right over the whole table.
    if scan.limit().is_some() || scan.partitions().is_empty() {
        return None;
    }
    let meta = scan.store_meta()?;
    let var = meta.data_vars.first()?;
    // Unknown chunking (0) leaves each partition whole.
    let chunk_len = var
        .chunks
        .as_ref()
        .and_then(|c| c.first().copied())
        .unwrap_or(0);
    let inner = inner_axes(scan, meta, var);
    Some(
        scan.partitions()
            .iter()
            .flat_map(|spec| {
                let boxes = inner_boxes(&spec.extra, &inner);
                split_selection(&spec.outer, chunk_len, usize::MAX)
                    .into_iter()
                    .flat_map(move |outer| {
                        boxes.clone().into_iter().map(move |extra| PartitionSpec {
                            outer: outer.clone(),
                            extra,
                        })
                    })
            })
            .filter(|spec| !spec.is_empty())
            .collect(),
    )
}

/// The inner axes to cut at chunk boundaries, as `(coordinate index, extent,
/// chunk length)`: those of more than one chunk, and unfiltered, since a
/// partition's selection on an axis replaces the filter's.
fn inner_axes(
    scan: &ZarrExec,
    meta: &ZarrStoreMeta,
    var: &ZarrArrayMeta,
) -> Vec<(usize, u64, u64)> {
    let (Some(dims), Some(chunks)) = (&var.dimensions, &var.chunks) else {
        return vec![];
    };
    (1..var.shape.len())
        .filter_map(|axis| {
            let name = dims.get(axis)?;
            let filtered = scan.coord_filters().is_some_and(|f| f.get(name).is_some());
            let coord = meta.coords.iter().position(|c| &c.name == name)?;
            let (extent, chunk) = (var.shape[axis], *chunks.get(axis)?);
            (!filtered && chunk > 0 && extent > chunk).then_some((coord, extent, chunk))
        })
        .collect()
}

/// Each chunk-aligned box within `extra`, over the `inner` axes.
fn inner_boxes(
    extra: &[(usize, CoordSelection)],
    inner: &[(usize, u64, u64)],
) -> Vec<Vec<(usize, CoordSelection)>> {
    let mut boxes = vec![extra.to_vec()];
    for &(coord, extent, chunk) in inner {
        let whole = CoordSelection::Range(0, extent as usize);
        boxes = boxes
            .into_iter()
            .flat_map(|b| {
                let selected = b
                    .iter()
                    .find(|(c, _)| *c == coord)
                    .map_or(&whole, |(_, s)| s);
                split_selection(selected, chunk, usize::MAX)
                    .into_iter()
                    .map(|piece| {
                        let mut b = b.clone();
                        b.retain(|(c, _)| *c != coord);
                        b.push((coord, piece));
                        b
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
    }
    boxes
}

/// Whether the partial aggregate groups by every inner axis its chunks are cut
/// along, so that no group spans two tiles.
fn groups_within_tiles(partial: &AggregateExec, scan: &ZarrExec, chunks: &[PartitionSpec]) -> bool {
    let (Some(meta), Some(first)) = (scan.store_meta(), chunks.first()) else {
        return false;
    };
    let keys: Vec<&str> = partial
        .group_expr()
        .expr()
        .iter()
        .filter_map(|(expr, _)| expr.downcast_ref::<Column>().map(Column::name))
        .collect();
    partial.group_expr().is_single()
        && !first.extra.is_empty()
        && first.extra.iter().all(|(coord, _)| {
            meta.coords
                .get(*coord)
                .is_some_and(|c| keys.contains(&c.name.as_str()))
        })
}

/// Whether the partial aggregate groups by the outer coordinate, or a cast of
/// it, over chunks cut along no other axis. Each of the coordinate's values
/// then lies in one chunk, so no group spans two, however they are dealt.
fn groups_within_chunks(
    partial: &AggregateExec,
    scan: &ZarrExec,
    chunks: &[PartitionSpec],
) -> bool {
    let Some(outer) = scan
        .store_meta()
        .and_then(|meta| meta.data_vars.first())
        .and_then(|var| var.dimensions.as_ref()?.first())
    else {
        return false;
    };
    partial.group_expr().is_single()
        && chunks.iter().all(|spec| spec.extra.is_empty())
        && partial
            .group_expr()
            .expr()
            .iter()
            .any(|(expr, _)| column_name(expr) == Some(outer.as_str()))
}

/// The column `expr` reads, if it is one, or a cast of one.
fn column_name(expr: &Arc<dyn PhysicalExpr>) -> Option<&str> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        return Some(column.name());
    }
    let inner = match expr.downcast_ref::<CastExpr>() {
        Some(cast) => cast.expr(),
        None => expr.downcast_ref::<TryCastExpr>()?.expr(),
    };
    column_name(inner)
}

/// `chunks` dealt a tile at a time, through time, in runs of at most
/// `per_invocation`, and whether each tile took only one.
fn by_tile(
    mut chunks: Vec<PartitionSpec>,
    per_invocation: usize,
) -> (Vec<Vec<PartitionSpec>>, bool) {
    let tile = |spec: &PartitionSpec| {
        let mut key: Vec<_> = spec.extra.iter().map(|(c, s)| (*c, s.as_range())).collect();
        key.sort();
        key
    };
    chunks.sort_by_cached_key(tile);
    let stacks: Vec<_> = chunks.chunk_by(|a, b| tile(a) == tile(b)).collect();
    let whole = stacks.iter().all(|stack| stack.len() <= per_invocation);
    let runs = stacks
        .into_iter()
        .flat_map(|stack| stack.chunks(per_invocation).map(<[_]>::to_vec))
        .collect();
    (runs, whole)
}

/// What a fan-out did.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub chunks: usize,
    pub invocations: usize,
    /// From dispatching to the last output.
    pub elapsed_s: f64,
    /// With `return_stats`: each invocation's record, as map-lambda keeps it.
    pub invocation_stats: Vec<Value>,
    /// With `return_stats`: each worker's `index`, `timings_ms`,
    /// `s3_requests` and `chunks_in_flight`; or with a combine, each
    /// state's `indexes`, with its workers' `timings_ms` and `s3_requests`
    /// summed.
    pub workers: Vec<Value>,
    /// Each dispatcher's record.
    pub dispatcher_stats: Vec<Value>,
    /// With `return_stats`: each frame from the dispatchers as it arrived
    /// here, `at_s` from dispatching, with its workers' result `bytes`.
    pub arrivals: Vec<Value>,
}

/// A fragment run on map-lambda workers, one invocation per run of chunks,
/// yielding every worker's batches in one partition as they arrive.
#[derive(Debug)]
pub struct MapLambdaZarrExec {
    schema: SchemaRef,
    plan: Arc<dyn ExecutionPlan>,
    fragment: String,
    runs: Vec<Vec<PartitionSpec>>,
    /// Whether no group's rows span two runs.
    whole_groups: bool,
    /// Whether the fragment ends in the final aggregate.
    finished: bool,
    /// What the dispatchers merge the workers' partial rows with, if they do.
    reduce: Option<String>,
    /// The joins' build sides read from other Zarr scans.
    joined: Vec<JoinedScan>,
    fan_out: Arc<FanOut>,
    properties: Arc<PlanProperties>,
    report: Arc<Mutex<Report>>,
}

impl MapLambdaZarrExec {
    pub fn new(
        plan: Arc<dyn ExecutionPlan>,
        runs: Vec<Vec<PartitionSpec>>,
        whole_groups: bool,
        fan_out: Arc<FanOut>,
    ) -> Result<Self> {
        let schema = plan.schema();
        let fragment = split::encode_plan(plan.clone())?;
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            schema,
            plan,
            fragment,
            runs,
            whole_groups,
            finished: false,
            reduce: None,
            joined: vec![],
            fan_out,
            properties,
            report: Default::default(),
        })
    }

    /// The fragment each worker runs, as base64 datafusion-proto bytes.
    pub fn fragment(&self) -> &str {
        &self.fragment
    }

    /// The fragment each worker runs, before it is bound to its chunks.
    pub fn fragment_plan(&self) -> &Arc<dyn ExecutionPlan> {
        &self.plan
    }

    /// Each invocation's chunks.
    pub fn inputs(&self) -> Vec<Value> {
        self.runs
            .iter()
            .map(|run| {
                serde_json::to_value(split::compact::compact(run))
                    .expect("partition specs serialize")
            })
            .collect()
    }

    /// The scan the runs are chunks of: the one no join reads as its build side.
    fn probe(&self) -> Option<Arc<dyn ExecutionPlan>> {
        split::zarr_scans(&self.plan)
            .ok()?
            .into_iter()
            .find(|scan| !self.joined.iter().any(|j| Arc::ptr_eq(&j.scan, scan)))
    }

    fn chunks(&self) -> usize {
        self.runs.iter().map(Vec::len).sum()
    }

    /// Whether each of `spec`'s chunks lies in one invocation's share, so that one
    /// worker writes it: along each axis invocations split, the tiles' inner axes
    /// or else the outer, the grid is the whole source axis, and its chunks divide
    /// the source's.
    fn tiles_chunks(&self, spec: &SkeletonSpec) -> bool {
        let Some(tile) = self.runs.first().and_then(|run| run.first()) else {
            return false;
        };
        let probe = self.probe();
        let Some(meta) = probe
            .as_ref()
            .and_then(|scan| scan.downcast_ref::<ZarrExec>())
            .and_then(ZarrExec::store_meta)
        else {
            return false;
        };
        let Some(var) = meta.data_vars.first() else {
            return false;
        };
        let outer = var.dimensions.as_ref().and_then(|dims| dims.first());
        let Some(outer) = outer.and_then(|name| meta.coords.iter().position(|c| &c.name == name))
        else {
            return false;
        };
        // Invocations split the inner axes a tile is cut along, and the outer
        // too if a tile's chunks span invocations, or else the outer alone.
        let mut split: Vec<usize> = tile.extra.iter().map(|(coord, _)| *coord).collect();
        if split.is_empty() || self.tiles_span_runs() {
            split.push(outer);
        }
        split.iter().all(|coord| {
            let name = &meta.coords[*coord].name;
            let source_chunk = var
                .dimensions
                .as_ref()
                .and_then(|dims| dims.iter().position(|d| d == name))
                .and_then(|axis| var.chunks.as_ref()?.get(axis).copied());
            let extent = meta.coords[*coord].shape.first().copied();
            let axis = spec.coords.iter().position(|c| &c.name == name);
            match (source_chunk, extent, axis) {
                (Some(source_chunk), Some(extent), Some(axis)) => {
                    spec.coords[axis].values.len() as u64 == extent
                        && source_chunk % spec.chunks[axis] == 0
                }
                _ => false,
            }
        })
    }

    /// Whether some tile's chunks are in more than one run.
    fn tiles_span_runs(&self) -> bool {
        let key = |selection: &CoordSelection| match selection {
            CoordSelection::Range(start, end) => (false, vec![*start, *end]),
            CoordSelection::Indices(indices) => (true, indices.clone()),
        };
        let mut run_of = HashMap::new();
        self.runs.iter().enumerate().any(|(i, run)| {
            run.iter().any(|spec| {
                let tile: Vec<_> = spec.extra.iter().map(|(c, s)| (*c, key(s))).collect();
                *run_of.entry(tile).or_insert(i) != i
            })
        })
    }

    /// The runs re-dealt so that each invocation writes whole chunks of `spec`
    /// along the outer axis, about as many hours each as before, for a fan-out
    /// split along it alone. A source chunk an output chunk boundary crosses is
    /// read by both sides of it: reads may overlap, writes must not.
    fn runs_by_output_chunk(&self, spec: &SkeletonSpec) -> Option<Vec<Vec<PartitionSpec>>> {
        let specs: Vec<&PartitionSpec> = self.runs.iter().flatten().collect();
        if specs.iter().any(|s| !s.extra.is_empty()) {
            return None;
        }
        let ranges = specs.iter().map(|s| s.as_range()).collect::<Option<_>>()?;
        let (chunks, per_run) = self.output_time_chunks(spec, ranges)?;
        Some(
            chunks
                .chunks(per_run)
                .map(|run| {
                    run.iter()
                        .flatten()
                        .map(|outer| PartitionSpec {
                            outer: outer.clone(),
                            extra: vec![],
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// The outer `ranges` the runs read cut at `spec`'s chunks along it, in
    /// order, and how many of those chunks a run is to take, to read about as
    /// many hours of a tile as before; if `spec` has the outer axis, and the
    /// ranges select its indexes with no gaps.
    fn output_time_chunks(
        &self,
        spec: &SkeletonSpec,
        mut ranges: Vec<(usize, usize)>,
    ) -> Option<(Vec<Vec<CoordSelection>>, usize)> {
        let probe = self.probe()?;
        let meta = probe.downcast_ref::<ZarrExec>()?.store_meta()?;
        let outer = meta.data_vars.first()?.dimensions.as_ref()?.first()?;
        let axis = spec.coords.iter().position(|c| &c.name == outer)?;
        let chunk = usize::try_from(spec.chunks[axis]).ok().filter(|&c| c > 0)?;
        ranges.sort();
        ranges.dedup();
        let first = ranges.first()?.0;
        // The output's axis is the scan's selection, so output index i is source
        // index first + i, but only if the selection has no gaps.
        let selected: usize = ranges.iter().map(|(s, e)| e - s).sum();
        let contiguous = ranges.windows(2).all(|w| w[0].1 == w[1].0);
        if !contiguous || selected != spec.coords[axis].values.len() {
            return None;
        }
        let mut by_chunk: BTreeMap<usize, Vec<CoordSelection>> = BTreeMap::new();
        for (start, end) in ranges {
            let mut at = start;
            while at < end {
                let k = (at - first) / chunk;
                let to = end.min(first + (k + 1) * chunk);
                by_chunk
                    .entry(k)
                    .or_default()
                    .push(CoordSelection::Range(at, to));
                at = to;
            }
        }
        // a tile's hours in the first run, which a run of no tiles has all of
        let run = self.runs.first()?;
        let tile = &run.first()?.extra;
        let per_run: usize = run
            .iter()
            .filter(|s| &s.extra == tile)
            .filter_map(|s| s.as_range())
            .map(|(s, e)| e - s)
            .sum();
        Some((by_chunk.into_values().collect(), (per_run / chunk).max(1)))
    }

    /// The runs re-dealt so that each invocation writes one of `spec`'s chunks
    /// over the inner axes the tiles are cut along, for a fan-out dealt by
    /// tile: through all of time, or if the output has a time axis, through a
    /// run of its chunks along it, about as many hours as before. Each reads its
    /// box from whichever tiles it overlaps: a cell lies in one box, so its
    /// groups stay whole.
    fn runs_by_output_box(&self, spec: &SkeletonSpec) -> Option<Vec<Vec<PartitionSpec>>> {
        let specs: Vec<&PartitionSpec> = self.runs.iter().flatten().collect();
        let first = specs.first()?;
        if first.extra.is_empty() {
            return None;
        }
        let probe = self.probe()?;
        let meta = probe.downcast_ref::<ZarrExec>()?.store_meta()?;
        // The output's chunks along each axis the tiles are cut on, which must
        // cover the whole source axis, so that output and source indexes agree.
        let mut axes: Vec<(usize, Vec<(usize, usize)>)> = vec![];
        for (coord, _) in &first.extra {
            let name = &meta.coords[*coord].name;
            let extent = usize::try_from(*meta.coords[*coord].shape.first()?).ok()?;
            let axis = spec.coords.iter().position(|c| &c.name == name)?;
            let chunk = usize::try_from(spec.chunks[axis]).ok().filter(|&c| c > 0)?;
            if spec.coords[axis].values.len() != extent {
                return None;
            }
            let pieces = (0..extent)
                .step_by(chunk)
                .map(|start| (start, extent.min(start + chunk)))
                .collect();
            axes.push((*coord, pieces));
        }
        let mut outers: Vec<CoordSelection> = vec![];
        for spec in &specs {
            if !outers.contains(&spec.outer) {
                outers.push(spec.outer.clone());
            }
        }
        let has_time = meta
            .data_vars
            .first()?
            .dimensions
            .as_ref()?
            .first()
            .is_some_and(|outer| spec.coords.iter().any(|c| &c.name == outer));
        let times: Vec<Vec<CoordSelection>> = if has_time {
            let ranges = outers.iter().map(|o| o.as_range()).collect::<Option<_>>()?;
            let (chunks, per_run) = self.output_time_chunks(spec, ranges)?;
            chunks.chunks(per_run).map(<[_]>::concat).collect()
        } else {
            vec![outers]
        };
        let mut boxes: Vec<Vec<(usize, CoordSelection)>> = vec![vec![]];
        for (coord, pieces) in axes {
            boxes = boxes
                .into_iter()
                .flat_map(|b| {
                    pieces.iter().map(move |&(start, end)| {
                        let mut b = b.clone();
                        b.push((coord, CoordSelection::Range(start, end)));
                        b
                    })
                })
                .collect();
        }
        Some(
            boxes
                .iter()
                .flat_map(|extra| {
                    times.iter().map(move |outers| {
                        outers
                            .iter()
                            .map(|outer| PartitionSpec {
                                outer: outer.clone(),
                                extra: extra.clone(),
                            })
                            .collect()
                    })
                })
                .collect(),
        )
    }

    pub fn report(&self) -> Report {
        self.report.lock().expect("unpoisoned").clone()
    }

    /// What every invocation is sent besides its chunks.
    fn template(&self) -> Map<String, Value> {
        let mut template =
            Map::from_iter([("plan".to_string(), Value::from(self.fragment.clone()))]);
        if let Some(n) = self.fan_out.settings.chunks_in_flight {
            template.insert("chunks_in_flight".into(), n.into());
        }
        template
    }
}

impl DisplayAs for MapLambdaZarrExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let function = &self.fan_out.settings.worker_function;
        let (chunks, invocations) = (self.chunks(), self.runs.len());
        let combiner = if self.reduce.is_some() {
            split::COMBINER
        } else {
            "none"
        };
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "MapLambdaZarrExec: function={function}, chunks={chunks}, \
                 invocations={invocations}, combiner={combiner}",
            ),
            DisplayFormatType::TreeRender => write!(
                f,
                "function={function}\nchunks={chunks}\ninvocations={invocations}\n\
                 combiner={combiner}",
            ),
        }
    }
}

fn external(e: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

fn to_value(record: &impl Serialize) -> Value {
    serde_json::to_value(record).expect("records serialize")
}

/// What arrived from the dispatchers at once: each response with the
/// inputs it answers, a worker's output for one, or a state for those its
/// dispatcher merged.
struct Arrival {
    succeeded: usize,
    responses: Vec<(Vec<usize>, Box<RawValue>)>,
    invocation_stats: Vec<Value>,
    dispatcher_stats: Vec<Value>,
}

impl From<Batch> for Arrival {
    fn from(batch: Batch) -> Self {
        Self {
            succeeded: batch.succeeded,
            responses: batch
                .outputs
                .into_iter()
                .map(|(index, output)| (vec![index], output))
                .collect(),
            invocation_stats: batch.invocation_stats.iter().map(to_value).collect(),
            dispatcher_stats: batch.dispatcher_stats.iter().map(to_value).collect(),
        }
    }
}

impl From<CombineBatch> for Arrival {
    fn from(batch: CombineBatch) -> Self {
        Self {
            succeeded: batch.succeeded,
            responses: batch
                .states
                .into_iter()
                .map(|state| (state.indexes, state.state))
                .collect(),
            invocation_stats: batch.invocation_stats.iter().map(to_value).collect(),
            dispatcher_stats: batch.dispatcher_stats.iter().map(to_value).collect(),
        }
    }
}

/// Each run's chunks of `plan`'s scans, a list a scan in `split::zarr_scans`'s
/// order: the probe's chunks, and each joined scan's box of them. A run whose
/// rows join none of a joined scan's is left out.
async fn joined_inputs(
    plan: &Arc<dyn ExecutionPlan>,
    runs: &[Vec<PartitionSpec>],
    joined: &[JoinedScan],
) -> Result<Vec<Value>> {
    let scans = split::zarr_scans(plan)?;
    let joined_at =
        |scan: &Arc<dyn ExecutionPlan>| joined.iter().position(|j| Arc::ptr_eq(&j.scan, scan));
    let [probe] = scans
        .iter()
        .filter(|scan| joined_at(scan).is_none())
        .collect::<Vec<_>>()[..]
    else {
        return Err(DataFusionError::Plan(
            "a fragment's joins have no single probe scan".into(),
        ));
    };
    let names: Vec<String> = joined
        .iter()
        .flat_map(|j| j.axes.iter().map(|a| a.probe.clone()))
        .collect();
    let probe = Grid::read(probe, &names).await?;
    let mut grids = vec![];
    for j in joined {
        let names: Vec<String> = j.axes.iter().map(|a| a.build.clone()).collect();
        grids.push(Grid::read(&j.scan, &names).await?);
    }
    let mut inputs = vec![];
    'runs: for run in runs {
        let mut lists = vec![];
        for scan in &scans {
            let Some(i) = joined_at(scan) else {
                lists.push(run.clone());
                continue;
            };
            let boxes = joined_boxes(run, &probe, &grids[i], &joined[i].axes)?;
            if boxes.is_empty() {
                continue 'runs;
            }
            let row: usize = scan
                .schema()
                .fields()
                .iter()
                .map(|field| match field.data_type() {
                    DataType::Dictionary(key, _) => key.primitive_width(),
                    other => other.primitive_width(),
                })
                .sum::<Option<usize>>()
                .unwrap_or(0);
            let bytes = boxes.iter().map(|b| grids[i].cells(b)).sum::<usize>() * row;
            if bytes > joined[i].max_bytes {
                return Err(DataFusionError::Plan(format!(
                    "an invocation would read {} MB of a joined Zarr scan, at most \
                     lambda.broadcast_bytes: join it on more of its coordinates",
                    bytes / 1_000_000
                )));
            }
            lists.push(boxes);
        }
        inputs.push(Value::from(
            lists
                .iter()
                .map(|list| to_value(&split::compact::compact(list)))
                .collect::<Vec<_>>(),
        ));
    }
    Ok(inputs)
}

/// A scan's store: its metadata, and the values of the coordinates needed,
/// times in microseconds since 1970, so that two stores' compare.
struct Grid {
    meta: ZarrStoreMeta,
    values: HashMap<String, Vec<f64>>,
    // searched, not scanned: a whole record's hours, for each of tens of
    // thousands of runs
    orders: HashMap<String, Order>,
}

impl Grid {
    async fn read(scan: &Arc<dyn ExecutionPlan>, names: &[String]) -> Result<Self> {
        let scan = scan.downcast_ref::<ZarrExec>().expect("a Zarr scan");
        let meta = scan
            .store_meta()
            .ok_or_else(|| DataFusionError::Plan(format!("{} has no metadata", scan.path())))?
            .clone();
        let mut values = HashMap::new();
        for name in names {
            if values.contains_key(name) {
                continue;
            }
            let mut read = split::coordinate_values(scan, name).await?;
            let unit = meta
                .coords
                .iter()
                .find(|c| c.name == *name)
                .and_then(|c| c.cf_time_attrs.as_ref())
                .and_then(|attrs| attrs.parse().ok());
            if let Some(unit) = unit {
                let scale = if unit.is_nanoseconds {
                    1e-3
                } else {
                    unit.multiplier_us as f64
                };
                for v in &mut read {
                    *v = unit.epoch_offset_us as f64 + *v * scale;
                }
            }
            values.insert(name.clone(), read);
        }
        let orders = values.iter().map(|(n, v)| (n.clone(), order(v))).collect();
        Ok(Self {
            meta,
            values,
            orders,
        })
    }

    /// The coordinate index that a partition's outer selection is on.
    fn outer(&self) -> Option<usize> {
        let name = self.meta.data_vars.first()?.dimensions.as_ref()?.first()?;
        self.meta.coords.iter().position(|c| &c.name == name)
    }

    /// The cells of its first variable that `spec` selects.
    fn cells(&self, spec: &PartitionSpec) -> usize {
        let Some(dims) = self
            .meta
            .data_vars
            .first()
            .and_then(|v| v.dimensions.as_ref())
        else {
            return 0;
        };
        dims.iter()
            .map(|name| {
                let coord = self.index(name);
                match self.selection(spec, coord) {
                    Some(CoordSelection::Range(start, end)) => end - start,
                    Some(CoordSelection::Indices(indices)) => indices.len(),
                    None => coord
                        .and_then(|c| self.meta.coords[c].shape.first().copied())
                        .unwrap_or(0) as usize,
                }
            })
            .product()
    }

    /// `spec`'s selection on `coord`, if it makes one.
    fn selection<'a>(
        &self,
        spec: &'a PartitionSpec,
        coord: Option<usize>,
    ) -> Option<&'a CoordSelection> {
        if coord == self.outer() {
            return Some(&spec.outer);
        }
        spec.extra
            .iter()
            .find(|(c, _)| Some(*c) == coord)
            .map(|(_, selection)| selection)
    }

    fn index(&self, name: &str) -> Option<usize> {
        self.meta.coords.iter().position(|c| c.name == name)
    }

    /// The values `spec` selects of coordinate `name`, all of them if it
    /// selects none.
    fn selected(&self, spec: &PartitionSpec, name: &str) -> Vec<f64> {
        let values = &self.values[name];
        match self.selection(spec, self.index(name)) {
            Some(CoordSelection::Range(start, end)) => values[*start..*end].to_vec(),
            Some(CoordSelection::Indices(indices)) => indices.iter().map(|&i| values[i]).collect(),
            None => values.clone(),
        }
    }
}

/// The boxes of `build`'s store that `run`'s chunks of `probe`'s can join with:
/// along each of `axes`, the build's values within the span of those the
/// probe's chunk selects, shifted by each of its offsets, and along any other
/// axis, all of it. A chunk with no such values along some axis joins nothing,
/// so has no box.
fn joined_boxes(
    run: &[PartitionSpec],
    probe: &Grid,
    build: &Grid,
    axes: &[Axis],
) -> Result<Vec<PartitionSpec>> {
    let mut boxes: Vec<PartitionSpec> = vec![];
    let Some(outer) = build.outer() else {
        return Ok(boxes);
    };
    'chunks: for spec in run {
        let extent =
            |coord: usize| build.meta.coords[coord].shape.first().copied().unwrap_or(0) as usize;
        let mut joined = PartitionSpec {
            outer: CoordSelection::Range(0, extent(outer)),
            extra: vec![],
        };
        for axis in axes {
            let span = probe
                .selected(spec, &axis.probe)
                .into_iter()
                .filter(|v| !v.is_nan())
                .fold(None, |span: Option<(f64, f64)>, v| {
                    Some(span.map_or((v, v), |(low, high)| (low.min(v), high.max(v))))
                });
            let Some((low, high)) = span else {
                continue 'chunks;
            };
            let values = &build.values[&axis.build];
            let within = within(values, build.orders[&axis.build], low, high, &axis.offsets);
            if within.is_empty() {
                continue 'chunks;
            }
            let selection = selecting(within);
            let Some(coord) = build.index(&axis.build) else {
                continue 'chunks;
            };
            if coord == outer {
                joined.outer = selection;
            } else {
                joined.extra.push((coord, selection));
            }
        }
        joined.extra.sort_by_key(|(coord, _)| *coord);
        match boxes.iter_mut().find(|b| b.extra == joined.extra) {
            Some(same) => same.outer = union(&same.outer, &joined.outer),
            None => boxes.push(joined),
        }
    }
    for (i, a) in boxes.iter().enumerate() {
        for b in &boxes[i + 1..] {
            let overlap = overlaps(&a.outer, &b.outer)
                && a.extra
                    .iter()
                    .zip(&b.extra)
                    .all(|((_, a), (_, b))| overlaps(a, b));
            if overlap {
                return Err(DataFusionError::Plan(
                    "an invocation's chunks join overlapping boxes of a joined Zarr scan, \
                     whose rows would be joined twice"
                        .into(),
                ));
            }
        }
    }
    Ok(boxes)
}

#[derive(Clone, Copy, PartialEq)]
enum Order {
    Ascending,
    Descending,
    Neither,
}

fn order(values: &[f64]) -> Order {
    if values.windows(2).all(|w| w[0] < w[1]) {
        Order::Ascending
    } else if values.windows(2).all(|w| w[0] > w[1]) {
        Order::Descending
    } else {
        Order::Neither
    }
}

/// The indices of `values`, in `order`, within `[low, high]` shifted by any of
/// `offsets`, ascending.
fn within(values: &[f64], order: Order, low: f64, high: f64, offsets: &[f64]) -> Vec<usize> {
    let mut found = vec![];
    for o in offsets {
        let (low, high) = (low + o, high + o);
        let range = if order == Order::Ascending {
            values.partition_point(|v| *v < low)..values.partition_point(|v| *v <= high)
        } else if order == Order::Descending {
            values.partition_point(|v| *v > high)..values.partition_point(|v| *v >= low)
        } else {
            found.extend((0..values.len()).filter(|&i| (low..=high).contains(&values[i])));
            continue;
        };
        found.extend(range);
    }
    found.sort_unstable();
    found.dedup();
    found
}

fn indices(selection: &CoordSelection) -> Vec<usize> {
    match selection {
        CoordSelection::Range(start, end) => (*start..*end).collect(),
        CoordSelection::Indices(indices) => indices.clone(),
    }
}

fn union(a: &CoordSelection, b: &CoordSelection) -> CoordSelection {
    let mut all = indices(a);
    all.extend(indices(b));
    all.sort_unstable();
    all.dedup();
    selecting(all)
}

/// A selection of `indices`, ascending: a range if they are one.
fn selecting(indices: Vec<usize>) -> CoordSelection {
    match indices[..] {
        [first, .., last] if last - first + 1 == indices.len() => {
            CoordSelection::Range(first, last + 1)
        }
        [only] => CoordSelection::Range(only, only + 1),
        _ => CoordSelection::Indices(indices),
    }
}

fn overlaps(a: &CoordSelection, b: &CoordSelection) -> bool {
    let b = indices(b);
    indices(a).iter().any(|i| b.contains(i))
}

fn decode(output: &Value) -> Result<Vec<RecordBatch>> {
    let batches = output["batches"].as_str().ok_or_else(|| {
        DataFusionError::Execution(format!("a fragment answered without batches: {output}"))
    })?;
    split::decode_batches(batches)
}

impl ExecutionPlan for MapLambdaZarrExec {
    fn name(&self) -> &str {
        "MapLambdaZarrExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // With a joined scan, its boxes are found once the coordinates are read.
        let inputs = self.joined.is_empty().then(|| self.inputs());
        *self.report.lock().expect("unpoisoned") = Report {
            chunks: self.chunks(),
            invocations: inputs.as_ref().map_or(0, Vec::len),
            ..Report::default()
        };
        if inputs.as_ref().is_some_and(Vec::is_empty) {
            return Ok(self.answer(futures::stream::empty().boxed()));
        }
        let joined = (self.plan.clone(), self.runs.clone(), self.joined.clone());

        let settings = &self.fan_out.settings;
        let locations = if settings.endpoint_url.is_some() {
            vec![]
        } else {
            let files = file_stores(&self.plan);
            if files.iter().any(|store| store.starts_with("file:")) {
                return Err(DataFusionError::Plan(
                    "a join's build side reads local files, which the workers can't: put \
                     them in S3"
                        .to_string(),
                ));
            }
            split::zarr_scans(&self.plan)?
                .iter()
                .filter_map(|scan| scan.downcast_ref::<ZarrExec>())
                .map(|scan| scan.path().to_string())
                .chain(files)
                .collect()
        };
        let (fan_out, template) = (self.fan_out.clone(), self.template());
        let combine = self.reduce.clone().map(|plan| Combine {
            combiner: split::COMBINER.into(),
            config: json!({ "plan": plan }),
        });
        let combined = combine.is_some();
        let on_progress = self.fan_out.on_progress.clone();
        let total = Arc::new(AtomicUsize::new(inputs.as_ref().map_or(0, Vec::len)));
        let (planned, counted) = (self.report.clone(), total.clone());
        // The workers each write their own chunks to a store made here first.
        let store = self
            .plan
            .downcast_ref::<DataSinkExec>()
            .and_then(|sink| sink.sink().downcast_ref::<ZarrDataSink>())
            .map(|zarr| (zarr.store_path().to_string(), zarr.spec().clone()));
        let run = futures::stream::once(async move {
            region::check(&locations).await?;
            let inputs = match inputs {
                Some(inputs) => inputs,
                None => {
                    let (plan, runs, joined) = joined;
                    let inputs = joined_inputs(&plan, &runs, &joined).await?;
                    counted.store(inputs.len(), Ordering::Relaxed);
                    planned.lock().expect("unpoisoned").invocations = inputs.len();
                    inputs
                }
            };
            if inputs.is_empty() {
                return Ok(futures::stream::empty().boxed());
            }
            let total = counted.load(Ordering::Relaxed);
            if let Some((path, spec)) = store {
                tokio::task::spawn_blocking(move || create_skeleton(&path, &spec))
                    .await
                    .map_err(external)?
                    .map_err(DataFusionError::External)?;
            }
            if let Some(on_progress) = &on_progress {
                on_progress(0, total);
            }
            let (session, worker) = (&fan_out.session, &fan_out.settings.worker_function);
            let arrivals = match combine {
                None => session
                    .map_unordered(worker, inputs, "partitions", template)
                    .map_err(external)?
                    .map_ok(Arrival::from)
                    .boxed(),
                Some(combine) => session
                    .map_combine(worker, inputs, "partitions", template, combine)
                    .map_err(external)?
                    .map_ok(Arrival::from)
                    .boxed(),
            };
            Ok::<_, DataFusionError>(arrivals.map_err(external).boxed())
        })
        .try_flatten();
        let report = self.report.clone();
        let return_stats = self.fan_out.settings.return_stats;
        let on_progress = self.fan_out.on_progress.clone();
        let mut succeeded = 0;
        let started = Instant::now();
        let batches = run
            .and_then(move |arrival| {
                if let (Some(on_progress), 1..) = (&on_progress, arrival.succeeded) {
                    succeeded += arrival.succeeded;
                    on_progress(succeeded, total.load(Ordering::Relaxed));
                }
                let decoded = arrival
                    .responses
                    .into_iter()
                    .map(|(indexes, response)| {
                        let response: Value =
                            serde_json::from_str(response.get()).map_err(external)?;
                        Ok((indexes, response))
                    })
                    .collect::<Result<Vec<_>>>()
                    .and_then(|responses| {
                        let mut report = report.lock().expect("unpoisoned");
                        report.elapsed_s = started.elapsed().as_secs_f64();
                        report.dispatcher_stats.extend(arrival.dispatcher_stats);
                        if return_stats {
                            let bytes: usize = responses
                                .iter()
                                .map(|(_, response)| {
                                    response["batches"].as_str().map_or(0, str::len)
                                })
                                .sum();
                            let at_s = report.elapsed_s;
                            report.arrivals.push(json!({"at_s": at_s, "bytes": bytes}));
                            report.invocation_stats.extend(arrival.invocation_stats);
                            report
                                .workers
                                .extend(responses.iter().map(|(indexes, response)| {
                                    let mut worker = json!({
                                        "timings_ms": response["timings_ms"],
                                        "s3_requests": response["s3_requests"],
                                    });
                                    if combined {
                                        worker["indexes"] = json!(indexes);
                                    } else {
                                        worker["index"] = json!(indexes[0]);
                                        worker["chunks_in_flight"] =
                                            response["chunks_in_flight"].clone();
                                    }
                                    worker
                                }));
                        }
                        responses
                            .iter()
                            .map(|(_, response)| decode(response))
                            .collect::<Result<Vec<_>>>()
                    })
                    .map(|batches| futures::stream::iter(batches.into_iter().flatten().map(Ok)));
                futures::future::ready(decoded)
            })
            .try_flatten();
        Ok(self.answer(batches.boxed()))
    }
}

impl MapLambdaZarrExec {
    /// The workers' batches, or when they write, one count of the rows they
    /// wrote, as the sink they each ran would answer.
    fn answer(
        &self,
        batches: BoxStream<'static, Result<RecordBatch>>,
    ) -> SendableRecordBatchStream {
        let schema = self.schema.clone();
        if !self.plan.is::<DataSinkExec>() {
            return Box::pin(RecordBatchStreamAdapter::new(schema, batches));
        }
        let total = futures::stream::once({
            let schema = schema.clone();
            async move {
                let written = batches
                    .try_fold(0, |total, batch| async move {
                        let counts = batch.column(0).as_primitive::<UInt64Type>();
                        Ok(total + counts.iter().flatten().sum::<u64>())
                    })
                    .await?;
                let count = Arc::new(UInt64Array::from(vec![written]));
                Ok(RecordBatch::try_new(schema, vec![count])?)
            }
        });
        Box::pin(RecordBatchStreamAdapter::new(schema, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array(name: &str, shape: &[u64], dims: &[&str]) -> ZarrArrayMeta {
        ZarrArrayMeta {
            name: name.into(),
            data_type: "float64".into(),
            shape: shape.to_vec(),
            chunks: None,
            coord_min_max: None,
            cf_time_attrs: None,
            dimensions: Some(dims.iter().map(|d| d.to_string()).collect()),
        }
    }

    /// A store of `var` over `axes`, each a coordinate with its values, the
    /// first the outer.
    fn grid(var: &str, axes: &[(&str, Vec<f64>)]) -> Grid {
        let dims: Vec<&str> = axes.iter().map(|(name, _)| *name).collect();
        let shape: Vec<u64> = axes.iter().map(|(_, v)| v.len() as u64).collect();
        let mut coords: Vec<_> = axes
            .iter()
            .map(|(name, v)| array(name, &[v.len() as u64], &[name]))
            .collect();
        coords.sort_by(|a, b| a.name.cmp(&b.name));
        Grid {
            meta: ZarrStoreMeta {
                coords,
                data_vars: vec![array(var, &shape, &dims)],
                total_rows: 0,
            },
            values: axes
                .iter()
                .map(|(name, v)| (name.to_string(), v.clone()))
                .collect(),
            orders: axes
                .iter()
                .map(|(name, v)| (name.to_string(), order(v)))
                .collect(),
        }
    }

    fn steps(start: f64, step: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| start + step * i as f64).collect()
    }

    /// ERA5-like: 10° cells, latitude descending; and a climatology of its
    /// 30°N to 50°N, 240°E to 290°E, by hour.
    fn era5_and_climatology() -> (Grid, Grid) {
        let era5 = grid(
            "t2m",
            &[
                ("time", steps(0.0, 1.0, 100)),
                ("latitude", steps(90.0, -10.0, 19)),
                ("longitude", steps(0.0, 10.0, 36)),
            ],
        );
        let clim = grid(
            "clim_t",
            &[
                ("hour", steps(0.0, 1.0, 24)),
                ("latitude", steps(50.0, -10.0, 3)),
                ("longitude", steps(240.0, 10.0, 6)),
            ],
        );
        (era5, clim)
    }

    fn axes() -> Vec<Axis> {
        ["latitude", "longitude"]
            .map(|a| Axis {
                probe: a.to_string(),
                build: a.to_string(),
                offsets: vec![0.0],
            })
            .to_vec()
    }

    fn tile(
        era5: &Grid,
        time: (usize, usize),
        lat: (usize, usize),
        lon: (usize, usize),
    ) -> PartitionSpec {
        PartitionSpec {
            outer: CoordSelection::Range(time.0, time.1),
            extra: vec![
                (
                    era5.index("latitude").unwrap(),
                    CoordSelection::Range(lat.0, lat.1),
                ),
                (
                    era5.index("longitude").unwrap(),
                    CoordSelection::Range(lon.0, lon.1),
                ),
            ],
        }
    }

    #[test]
    fn a_tiles_chunks_join_the_box_of_the_other_store_over_its_cells() {
        let (era5, clim) = era5_and_climatology();
        // 50°N to 10°N by 250°E to 300°E, over two pencils of time
        let run = [
            tile(&era5, (0, 50), (4, 9), (25, 31)),
            tile(&era5, (50, 100), (4, 9), (25, 31)),
        ];

        let boxes = joined_boxes(&run, &era5, &clim, &axes()).unwrap();

        // every hour of 50°N to 30°N by 250°E to 290°E: the one box for both
        assert_eq!(
            boxes,
            [PartitionSpec {
                outer: CoordSelection::Range(0, 24),
                extra: vec![
                    (clim.index("latitude").unwrap(), CoordSelection::Range(0, 3)),
                    (
                        clim.index("longitude").unwrap(),
                        CoordSelection::Range(1, 6)
                    ),
                ],
            }]
        );
    }

    #[test]
    fn a_whole_globe_chunk_joins_all_of_the_other_store() {
        let (era5, clim) = era5_and_climatology();
        let run = [PartitionSpec {
            outer: CoordSelection::Range(0, 24),
            extra: vec![],
        }];

        let boxes = joined_boxes(&run, &era5, &clim, &axes()).unwrap();

        assert_eq!(
            boxes,
            [PartitionSpec {
                outer: CoordSelection::Range(0, 24),
                extra: vec![
                    (clim.index("latitude").unwrap(), CoordSelection::Range(0, 3)),
                    (
                        clim.index("longitude").unwrap(),
                        CoordSelection::Range(0, 6)
                    ),
                ],
            }]
        );
    }

    #[test]
    fn a_tile_outside_the_other_store_joins_nothing() {
        let (era5, clim) = era5_and_climatology();
        // the tropics
        let run = [tile(&era5, (0, 50), (8, 11), (25, 31))];

        assert_eq!(joined_boxes(&run, &era5, &clim, &axes()).unwrap(), []);
    }

    #[test]
    fn values_within_the_span_but_not_contiguous_are_picked_out() {
        let (era5, _) = era5_and_climatology();
        let clim = grid(
            "clim_t",
            &[
                ("hour", steps(0.0, 1.0, 24)),
                ("latitude", vec![50.0, -50.0, 40.0]),
                ("longitude", steps(240.0, 10.0, 6)),
            ],
        );
        let run = [tile(&era5, (0, 50), (4, 6), (24, 30))];

        let [joined] = &joined_boxes(&run, &era5, &clim, &axes()).unwrap()[..] else {
            panic!("one box");
        };

        assert_eq!(
            joined.extra[0],
            (
                clim.index("latitude").unwrap(),
                CoordSelection::Indices(vec![0, 2])
            )
        );
    }

    #[test]
    fn a_chunk_joins_the_hours_its_leads_reach() {
        let hour = 3.6e9;
        let era5 = grid(
            "t2m",
            &[
                ("time", steps(0.0, hour, 100)),
                ("latitude", steps(90.0, -10.0, 19)),
                ("longitude", steps(0.0, 10.0, 36)),
            ],
        );
        let mut axes = axes();
        axes.push(Axis {
            probe: "time".into(),
            build: "time".into(),
            offsets: [6.0, 24.0, 72.0].map(|h| h * hour).to_vec(),
        });
        let at = |t: usize| PartitionSpec {
            outer: CoordSelection::Range(t, t + 1),
            extra: vec![],
        };

        let [early] = &joined_boxes(&[at(10)], &era5, &era5, &axes).unwrap()[..] else {
            panic!("one box");
        };
        let [late] = &joined_boxes(&[at(90)], &era5, &era5, &axes).unwrap()[..] else {
            panic!("one box");
        };

        assert_eq!(early.outer, CoordSelection::Indices(vec![16, 34, 82]));
        assert_eq!(late.outer, CoordSelection::Range(96, 97));
        assert_eq!(joined_boxes(&[at(99)], &era5, &era5, &axes).unwrap(), []);
        let [both] = &joined_boxes(&[at(10), at(11)], &era5, &era5, &axes).unwrap()[..] else {
            panic!("the two hours' boxes merged");
        };
        assert_eq!(
            both.outer,
            CoordSelection::Indices(vec![16, 17, 34, 35, 82, 83])
        );
    }

    #[test]
    fn a_table_is_created_with_its_options_quoted() {
        let table = Table {
            location: "s3://b/it's".to_string(),
            options: BTreeMap::from([
                ("group".to_string(), "single/spatial".to_string()),
                ("anonymous".to_string(), "true".to_string()),
            ]),
        };
        assert_eq!(
            table.create("era5"),
            "CREATE EXTERNAL TABLE \"era5\" STORED AS ZARR LOCATION 's3://b/it''s' \
             OPTIONS ('anonymous' 'true', 'group' 'single/spatial')"
        );
        assert_eq!(
            Table {
                location: "/data/era5".to_string(),
                ..Table::default()
            }
            .create("e"),
            "CREATE EXTERNAL TABLE \"e\" STORED AS ZARR LOCATION '/data/era5'"
        );
    }
}

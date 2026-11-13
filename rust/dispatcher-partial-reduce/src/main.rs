use dispatcher_partial_reduce::PartialReduce;
use lambda_runtime::Error;

#[tokio::main]
async fn main() -> Result<(), Error> {
    map_lambda_dispatcher::run_with::<PartialReduce>().await
}

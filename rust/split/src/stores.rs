//! The object stores a session reads tables other than Zarr from, such as a
//! join's Parquet side, made as their URLs are first used.

use std::sync::Arc;

use datafusion::error::Result;
use datafusion::execution::object_store::{DefaultObjectStoreRegistry, ObjectStoreRegistry};
use object_store::aws::AmazonS3Builder;
use object_store::ObjectStore;
use url::Url;

/// DataFusion's registry, which knows only local files, that also makes a store
/// for any `s3://` bucket, with its credentials and region from the environment:
/// a worker's role, or the client's exported credentials.
#[derive(Debug, Default)]
pub struct Registry(DefaultObjectStoreRegistry);

impl ObjectStoreRegistry for Registry {
    fn register_store(
        &self,
        url: &Url,
        store: Arc<dyn ObjectStore>,
    ) -> Option<Arc<dyn ObjectStore>> {
        self.0.register_store(url, store)
    }

    fn deregister_store(&self, url: &Url) -> Result<Arc<dyn ObjectStore>> {
        self.0.deregister_store(url)
    }

    fn get_store(&self, url: &Url) -> Result<Arc<dyn ObjectStore>> {
        match self.0.get_store(url) {
            Err(_) if url.scheme() == "s3" => {
                let store: Arc<dyn ObjectStore> =
                    Arc::new(AmazonS3Builder::from_env().with_url(url.as_str()).build()?);
                self.0.register_store(url, store.clone());
                Ok(store)
            }
            found => found,
        }
    }
}

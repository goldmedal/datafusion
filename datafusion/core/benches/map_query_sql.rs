use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use parking_lot::Mutex;
use rand::prelude::ThreadRng;
use rand::Rng;
use tokio::runtime::Runtime;

use datafusion::prelude::SessionContext;
use datafusion_common::ScalarValue;
use datafusion_expr::Expr;
use datafusion_functions_array::map::{map, map_from_array};

mod data_utils;

fn keys(rng: &mut ThreadRng) -> Vec<String> {
    let mut keys = vec![];
    for _ in 0..1000 {
        keys.push(rng.gen_range(0..9999).to_string());
    }
    keys
}

fn values(rng: &mut ThreadRng) -> Vec<i32> {
    let mut values = vec![];
    for _ in 0..1000 {
        values.push(rng.gen_range(0..9999));
    }
    values
}

fn t_batch() -> RecordBatch {
    let c1: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    RecordBatch::try_from_iter(vec![("c1", c1)]).unwrap()
}

fn create_context() -> datafusion_common::Result<Arc<Mutex<SessionContext>>> {
    let ctx = SessionContext::new();
    ctx.register_batch("t", t_batch())?;
    Ok(Arc::new(Mutex::new(ctx)))
}

fn criterion_benchmark(c: &mut Criterion) {
    let ctx = create_context().unwrap();
    let rt = Runtime::new().unwrap();
    let df = rt.block_on(ctx.lock().table("t")).unwrap();

    let mut rng = rand::thread_rng();
    let keys = keys(&mut rng);
    let values = values(&mut rng);
    let mut key_buffer = Vec::new();
    let mut value_buffer = Vec::new();

    for i in 0..1000 {
        key_buffer.push(Expr::Literal(ScalarValue::Utf8(Some(keys[i].clone()))));
        value_buffer.push(Expr::Literal(ScalarValue::Int32(Some(values[i]))));
    }
    c.bench_function("map_1000", |b| {
        b.iter(|| {
            black_box(
                rt.block_on(
                    df.clone()
                        .select(vec![map(key_buffer.clone(), value_buffer.clone())])
                        .unwrap()
                        .collect(),
                )
                .unwrap(),
            );
        });
    });
    c.bench_function("map_one_1000", |b| {
        b.iter(|| {
            black_box(
                rt.block_on(
                    df.clone()
                        .select(vec![map_from_array(
                            key_buffer.clone(),
                            value_buffer.clone(),
                        )])
                        .unwrap()
                        .collect(),
                )
                .unwrap(),
            );
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
